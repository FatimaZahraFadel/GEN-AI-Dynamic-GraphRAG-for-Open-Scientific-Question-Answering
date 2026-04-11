"""
Stage 5 — Graph Builder: build a temporary in-memory knowledge graph.

The graph is built fresh for every user query from the entities and relations
produced by EntityExtractor.  It is never stored to a database or reused
across queries.  All methods receive and return the graph explicitly so that
no state leaks between pipeline runs.
"""

import re
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

import networkx as nx

from models.graph_node import Entity, Relation
from pipeline.embedding_service import EmbeddingService
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """
    Constructs a temporary directed knowledge graph from extracted entities
    and relations, and provides subgraph retrieval for the RAG context step.

    Design constraint
    -----------------
    The graph is **never** stored as an instance attribute.  Every method
    that produces or modifies a graph accepts it as a parameter and returns
    it explicitly.  This enforces per-query isolation — a fresh
    ``nx.DiGraph()`` is created inside :meth:`build` at the start of each
    pipeline run and discarded when the answer has been generated.
    """

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embedding_service = embedding_service or EmbeddingService.get_instance()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        entities: List[Entity],
        relations: List[Relation],
        question: str = "",
    ) -> nx.DiGraph:
        """
        Main entry point: create a fresh graph and populate it.

        Creates a new ``nx.DiGraph``, adds all entity nodes, then adds all
        valid relation edges.  The completed graph is returned; it is never
        stored on the instance.

        Parameters
        ----------
        entities : list[Entity]
            Deduplicated entities from :class:`EntityExtractor`.
        relations : list[Relation]
            Relations from :class:`EntityExtractor`.

        Returns
        -------
        nx.DiGraph
            Populated knowledge graph for this query.
        """
        graph = nx.DiGraph()

        norm_entities, norm_relations = self._normalize_entities_relations(entities, relations)
        filtered_entities, filtered_relations = self._filter_entities_by_question(
            norm_entities,
            norm_relations,
            question=question,
            threshold=0.30,
        )

        self.add_entities(graph, filtered_entities)
        self.add_relations(graph, filtered_relations)
        self.optimize_for_query(graph, question)

        logger.info(
            f"Graph built: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges."
        )
        return graph

    def optimize_for_query(self, graph: nx.DiGraph, question: str) -> None:
        """
        Apply post-build cleanup and densification steps for query focus.

        Order
        -----
        1. Remove noisy generic nodes.
        2. Add co-occurrence edges (Step 7b).
        3. Add semantic bridge edges (Step 7a).
        4. Prune remaining isolated non-focus nodes.
        """
        if graph.number_of_nodes() == 0:
            return
        self._remove_generic_nodes(graph)
        self._add_cooccurrence_edges(graph, min_cooccurrences=2)
        self._add_semantic_bridge_edges(graph, top_k_per_node=2, min_similarity=0.62)
        self._prune_isolated_non_focus_nodes(graph, question)

        n, e = graph.number_of_nodes(), graph.number_of_edges()
        density = e / max(n * (n - 1), 1)
        logger.info(
            "optimize_for_query: graph after densification — %d nodes, %d edges, "
            "density=%.4f",
            n, e, density,
        )

    def add_entities(self, graph: nx.DiGraph, entities: List[Entity]) -> None:
        """
        Add each entity as a node in the graph.

        Node attributes stored
        ----------------------
        - ``label``           — human-readable name (``entity.label``)
        - ``type``            — entity category (``entity.type``)
        - ``source_paper_id`` — originating paper ID

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to mutate (modified in-place).
        entities : list[Entity]
            Entities to add as nodes.
        """
        for entity in entities:
            existing = graph.nodes.get(entity.id, {}) if entity.id in graph else {}
            node_embedding = existing.get("embedding")
            if node_embedding is None:
                node_embedding = self._embedding_service.encode_text(entity.label)
            node_weight = float(getattr(entity, "intent_weight", existing.get("node_weight", 1.0)))
            graph.add_node(
                entity.id,
                label=entity.label,
                type=entity.type,
                source_paper_id=entity.source_paper_id,
                embedding=node_embedding,
                node_weight=max(float(existing.get("node_weight", 1.0)), node_weight),
            )

        logger.info(f"add_entities: added {len(entities)} node(s).")

    def add_relations(
        self, graph: nx.DiGraph, relations: List[Relation]
    ) -> None:
        """
        Add each relation as a directed edge between existing nodes.

        An edge is only added when **both** ``source_id`` and ``target_id``
        already exist in the graph as nodes.  Relations that reference unknown
        nodes are counted and logged but otherwise silently skipped.

        Edge attributes stored
        ----------------------
        - ``relation_type``   — type string (e.g. ``"affects"``)
        - ``source_paper_id`` — originating paper ID

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to mutate (modified in-place).
        relations : list[Relation]
            Relations to add as directed edges.
        """
        added = 0
        skipped = 0

        for rel in relations:
            if rel.source_id in graph and rel.target_id in graph:
                edge_weight = float(getattr(rel, "intent_weight", 1.0))
                if graph.has_edge(rel.source_id, rel.target_id):
                    existing = graph.get_edge_data(rel.source_id, rel.target_id) or {}
                    edge_weight = max(float(existing.get("edge_weight", 1.0)), edge_weight)
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    source_paper_id=rel.source_paper_id,
                    edge_weight=edge_weight,
                )
                added += 1
            else:
                skipped += 1

        logger.info(
            f"add_relations: added {added} edge(s), "
            f"skipped {skipped} (missing nodes)."
        )

    def get_subgraph(
        self,
        graph: nx.DiGraph,
        entity_ids: List[str],
        depth: int = 2,
    ) -> nx.DiGraph:
        """
        Retrieve a neighbourhood subgraph around a set of seed entity IDs.

        Starting from each seed node, the graph is traversed (BFS) up to
        ``depth`` hops in both directions (predecessors and successors) on
        the directed graph.  All discovered nodes and the edges between them
        are collected into a new ``nx.DiGraph``.

        Parameters
        ----------
        graph : nx.DiGraph
            The full knowledge graph for this query.
        entity_ids : list[str]
            Seed entity IDs to start the traversal from.
        depth : int
            Maximum number of hops from each seed node. Defaults to 2.

        Returns
        -------
        nx.DiGraph
            Subgraph containing the neighbourhood of the seed nodes.
        """
        visited: set[str] = set()

        for seed in entity_ids:
            if seed not in graph:
                logger.warning(
                    f"get_subgraph: seed node '{seed}' not found in graph — skipped."
                )
                continue

            # BFS over undirected view so we traverse both in- and out-edges
            queue: deque[tuple[str, int]] = deque([(seed, 0)])
            visited.add(seed)

            while queue:
                node, current_depth = queue.popleft()
                if current_depth >= depth:
                    continue
                # Explore successors and predecessors
                neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))

        subgraph = graph.subgraph(visited).copy()
        logger.info(
            f"get_subgraph: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges "
            f"(seeds={entity_ids}, depth={depth})."
        )
        return subgraph

    def graph_summary(self, graph: nx.DiGraph) -> dict:
        """
        Return a statistical summary of the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The knowledge graph to summarise.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``num_nodes`` (int) — total node count
            - ``num_edges`` (int) — total edge count
            - ``node_types`` (dict[str, int]) — entity type -> node count
            - ``most_connected_nodes`` (list[dict]) — top-5 nodes by total
              degree (in + out), each entry:
              ``{"id": ..., "label": ..., "type": ..., "degree": ...}``
        """
        # Count nodes per entity type
        node_types: Dict[str, int] = {}
        for _, attrs in graph.nodes(data=True):
            entity_type = attrs.get("type", "Unknown")
            node_types[entity_type] = node_types.get(entity_type, 0) + 1

        # Top-5 nodes by undirected degree (in-degree + out-degree)
        degree_list = [
            (node, graph.in_degree(node) + graph.out_degree(node))
            for node in graph.nodes()
        ]
        degree_list.sort(key=lambda x: x[1], reverse=True)

        most_connected: List[dict] = []
        for node_id, degree in degree_list[:5]:
            attrs = graph.nodes[node_id]
            most_connected.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id),
                    "type": attrs.get("type", "Unknown"),
                    "degree": degree,
                }
            )

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "node_types": node_types,
            "most_connected_nodes": most_connected,
        }

    # ------------------------------------------------------------------
    # Query-focused graph optimisation helpers
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        t = re.sub(r"[^a-z0-9\s\-]", " ", text.lower()).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def _canonical_entity_label(self, label: str) -> str:
        t = self._normalize_text(label)
        aliases = {
            "wheat rust": "rust",
            "rust disease": "rust",
            "leaf rust": "rust",
            "stem rust": "rust",
            "stripe rust": "rust",
            "fusarium head blight": "fusarium blight",
            "powdery mildew disease": "powdery mildew",
        }
        if t in aliases:
            return aliases[t]
        t = re.sub(r"\b(disease|condition|infection|pathogen|pathogens)\b", "", t).strip()
        t = re.sub(r"\s+", " ", t)
        return t or self._normalize_text(label)

    def _normalize_entities_relations(
        self,
        entities: List[Entity],
        relations: List[Relation],
        similarity_threshold: float = 0.82,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Step 6 — Entity normalization via embedding similarity clustering.

        Merges near-duplicate entities using a two-pass strategy:
        1. Canonical text normalization (regex + alias map).
        2. Embedding similarity clustering (union-find) within the same type —
           pairs above ``similarity_threshold`` are merged into one node.

        This is fully general and domain-agnostic: no hardcoded synonyms are
        needed beyond the small alias map for known exact duplicates.
        """
        if not entities:
            return entities, relations

        before_count = len(entities)
        canonical_to_entity: Dict[str, Entity] = {}
        old_to_new: Dict[str, str] = {}

        for entity in entities:
            canonical = self._canonical_entity_label(entity.label)
            if canonical not in canonical_to_entity:
                canonical_to_entity[canonical] = Entity(
                    id=re.sub(r"\s+", "_", canonical),
                    label=canonical,
                    type=entity.type,
                    source_paper_id=entity.source_paper_id,
                )
            old_to_new[entity.id] = canonical_to_entity[canonical].id

        merged = list(canonical_to_entity.values())
        if len(merged) > 1:
            labels = [e.label for e in merged]
            embs = self._embedding_service.encode_batch(labels)
            parent = list(range(len(merged)))

            def _find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def _union(i: int, j: int) -> None:
                ri = _find(i)
                rj = _find(j)
                if ri != rj:
                    parent[rj] = ri

            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            norm_embs = embs / norms

            merge_count = 0
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i].type != merged[j].type:
                        continue
                    sim = float(norm_embs[i] @ norm_embs[j])
                    if sim >= similarity_threshold:
                        _union(i, j)
                        merge_count += 1

            rep_to_entity: Dict[int, Entity] = {}
            for i, ent in enumerate(merged):
                rep = _find(i)
                rep_to_entity.setdefault(rep, ent)

            # Build mapping from all canonical ids to representative ids.
            cid_to_rep: Dict[str, str] = {}
            for i, ent in enumerate(merged):
                rep = _find(i)
                cid_to_rep[ent.id] = rep_to_entity[rep].id

            for old, cid in list(old_to_new.items()):
                old_to_new[old] = cid_to_rep.get(cid, cid)

            merged = list(rep_to_entity.values())
            logger.info(
                "_normalize_entities_relations: %d entities → %d after merging "
                "(%d embedding-cluster merge(s), threshold=%.2f)",
                before_count, len(merged), merge_count, similarity_threshold,
            )

        new_relations: List[Relation] = []
        seen_rel = set()
        for rel in relations:
            src = old_to_new.get(rel.source_id, rel.source_id)
            dst = old_to_new.get(rel.target_id, rel.target_id)
            if src == dst:
                continue
            key = (src, dst, rel.relation_type)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            new_relations.append(
                Relation(
                    source_id=src,
                    target_id=dst,
                    relation_type=rel.relation_type,
                    source_paper_id=rel.source_paper_id,
                )
            )
        return merged, new_relations

    def _filter_entities_by_question(
        self,
        entities: List[Entity],
        relations: List[Relation],
        question: str,
        threshold: float,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Step 5 — Query-aware entity filtering (fully embedding-driven).

        No hardcoded domain keyword lists.  The adaptive threshold is derived
        from the *mean* similarity of all entities to the question, so the
        cut-off self-adjusts to any topic.

        Algorithm
        ---------
        1. Embed question and all entity labels.
        2. Compute cosine similarity for each entity.
        3. Compute adaptive threshold = max(base_threshold, mean_sim - 0.05).
        4. Keep entities above adaptive threshold.
        5. Always keep the top-5 entities by similarity to guarantee a
           non-empty graph even for very short/abstract questions.
        """
        if not entities:
            return entities, relations
        if not question.strip():
            return entities, relations

        question_emb = self._embedding_service.encode_text(question)
        labels = [e.label for e in entities]
        ent_embs = self._embedding_service.encode_batch(labels)

        q_norm = question_emb / (np.linalg.norm(question_emb) + 1e-10)
        e_norm = ent_embs / (np.linalg.norm(ent_embs, axis=1, keepdims=True) + 1e-10)
        sims = (e_norm @ q_norm).astype(float)

        # Adaptive threshold: anchored to the mean similarity so it scales
        # across domains without any hardcoded terms.
        mean_sim = float(np.mean(sims))
        adaptive_threshold = max(threshold, mean_sim - 0.05)

        # Also keep direct token matches from the query (still query-driven).
        q_tokens = {tok for tok in self._normalize_text(question).split() if len(tok) > 3}

        kept_ids: set = set()
        for idx, entity in enumerate(entities):
            sim = float(sims[idx])
            label_n = self._normalize_text(entity.label)
            direct_match = any(tok in label_n for tok in q_tokens)
            if sim >= adaptive_threshold or direct_match:
                kept_ids.add(entity.id)

        # Safety net: always keep top-5 most similar entities
        top5_indices = np.argsort(sims)[::-1][:5]
        for idx in top5_indices:
            kept_ids.add(entities[idx].id)

        filtered_entities = [e for e in entities if e.id in kept_ids]
        filtered_relations = [
            r for r in relations if r.source_id in kept_ids and r.target_id in kept_ids
        ]

        removed = len(entities) - len(filtered_entities)
        logger.info(
            "_filter_entities_by_question: mean_sim=%.4f adaptive_threshold=%.4f "
            "kept=%d removed=%d",
            mean_sim, adaptive_threshold, len(filtered_entities), removed,
        )
        return filtered_entities, filtered_relations

    def _is_focus_node(self, attrs: Dict) -> bool:
        """
        Return True for nodes that represent substantive scientific entities.
        Uses entity type rather than hardcoded domain keywords.
        """
        ntype = str(attrs.get("type", ""))
        return ntype in {
            "Disease / Condition",
            "Cause / Factor",
            "Effect / Outcome",
            "Treatment / Method",
        }

    def _remove_generic_nodes(self, graph: nx.DiGraph) -> None:
        """
        Drop overly generic nodes that are isolated from focus nodes.

        'Generic' is defined by a short fixed list of universal noise labels
        that appear in any domain (e.g. 'microorganisms', 'bacteria' as
        stand-alone concepts without any relation to domain focus nodes).
        Focus nodes are identified by entity type, not by domain keywords.
        """
        generic_labels = {
            "climate change",
            "microorganisms",
            "microorganism",
            "bacteria",
            "microbes",
            "soil microbes",
        }
        focus_nodes = {nid for nid, attrs in graph.nodes(data=True) if self._is_focus_node(attrs)}
        to_remove = []

        for nid, attrs in graph.nodes(data=True):
            label_n = self._normalize_text(str(attrs.get("label", "")))
            if label_n not in generic_labels:
                continue
            neigh = set(graph.predecessors(nid)) | set(graph.successors(nid))
            if not any(n in focus_nodes for n in neigh):
                to_remove.append(nid)

        if to_remove:
            graph.remove_nodes_from(to_remove)

    def _add_semantic_bridge_edges(
        self,
        graph: nx.DiGraph,
        top_k_per_node: int,
        min_similarity: float,
    ) -> None:
        """
        Step 7a — Reconnect fragmented components with embedding-similarity bridges.

        For each node, adds edges to the top-k most similar nodes whose
        similarity exceeds ``min_similarity``, provided no edge already exists.
        Edge weight is set to the cosine similarity score for downstream ranking.
        """
        node_ids = list(graph.nodes())
        if len(node_ids) < 3:
            return

        labels = [graph.nodes[n].get("label", n) for n in node_ids]
        embs = self._embedding_service.encode_batch(labels)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        norm_embs = embs / norms

        sim_matrix = norm_embs @ norm_embs.T
        added = 0
        for i, nid in enumerate(node_ids):
            ranked = np.argsort(sim_matrix[i])[::-1]
            local_added = 0
            for j in ranked:
                if i == j:
                    continue
                score = float(sim_matrix[i, j])
                if score < min_similarity:
                    break
                other = node_ids[j]
                if graph.has_edge(nid, other) or graph.has_edge(other, nid):
                    continue
                graph.add_edge(
                    nid,
                    other,
                    relation_type="semantically_related",
                    source_paper_id="semantic_bridge",
                    edge_weight=round(score, 4),
                )
                local_added += 1
                added += 1
                if local_added >= top_k_per_node:
                    break

        if added:
            logger.info("Added %d semantic bridge edge(s) for graph connectivity.", added)

    def _add_cooccurrence_edges(
        self,
        graph: nx.DiGraph,
        min_cooccurrences: int = 2,
    ) -> None:
        """
        Step 7b — Add edges for entities that co-occur in multiple papers.

        Two nodes that appear in ≥ ``min_cooccurrences`` papers together gain
        a 'co_occurs_with' edge.  This strengthens connections that are
        evidenced by multiple independent sources — a reliable signal across
        any domain.
        """
        from collections import defaultdict
        paper_to_nodes: Dict[str, List[str]] = defaultdict(list)
        for nid, attrs in graph.nodes(data=True):
            pid = attrs.get("source_paper_id", "")
            if pid:
                paper_to_nodes[pid].append(nid)

        cooccur_count: Dict[tuple, int] = defaultdict(int)
        for nodes_in_paper in paper_to_nodes.values():
            for i in range(len(nodes_in_paper)):
                for j in range(i + 1, len(nodes_in_paper)):
                    pair = (nodes_in_paper[i], nodes_in_paper[j])
                    cooccur_count[pair] += 1

        added = 0
        for (n1, n2), count in cooccur_count.items():
            if count < min_cooccurrences:
                continue
            if n1 not in graph or n2 not in graph:
                continue
            if graph.has_edge(n1, n2) or graph.has_edge(n2, n1):
                # Update edge weight if already present
                edge_data = graph.get_edge_data(n1, n2) or graph.get_edge_data(n2, n1) or {}
                existing_weight = edge_data.get("edge_weight", 1.0)
                if graph.has_edge(n1, n2):
                    graph[n1][n2]["edge_weight"] = round(existing_weight + 0.1 * count, 4)
                else:
                    graph[n2][n1]["edge_weight"] = round(existing_weight + 0.1 * count, 4)
                continue
            graph.add_edge(
                n1, n2,
                relation_type="co_occurs_with",
                source_paper_id="cooccurrence",
                edge_weight=round(0.5 + 0.1 * count, 4),
            )
            added += 1

        if added:
            logger.info(
                "_add_cooccurrence_edges: added %d co-occurrence edge(s) "
                "(min_cooccurrences=%d).", added, min_cooccurrences,
            )

    def _prune_isolated_non_focus_nodes(self, graph: nx.DiGraph, question: str) -> None:
        """
        Remove isolated (degree-0) nodes that are not query-relevant.

        Uses query token matching and entity type instead of hardcoded domain
        keyword lists — making this fully domain-agnostic.
        """
        # Focus on tokens from the query itself (domain-agnostic)
        q_tokens = {tok for tok in self._normalize_text(question).split() if len(tok) > 3}

        # Focus types that should always be preserved even if isolated
        focus_types = {
            "Disease / Condition",
            "Environment / Location",
            "Cause / Factor",
        }

        to_remove = []
        for nid, attrs in graph.nodes(data=True):
            if graph.degree(nid) > 0:
                continue
            label_n = self._normalize_text(str(attrs.get("label", "")))
            ntype = str(attrs.get("type", ""))
            if ntype in focus_types:
                continue
            if any(tok in label_n for tok in q_tokens):
                continue
            to_remove.append(nid)

        if to_remove:
            logger.info("_prune_isolated_non_focus_nodes: removing %d isolated noise node(s).", len(to_remove))
            graph.remove_nodes_from(to_remove)


# ---------------------------------------------------------------------------
# Full pipeline smoke-test:
# DomainDetector -> PaperRetriever -> PaperFilter -> EntityExtractor -> GraphBuilder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    from dotenv import load_dotenv
    load_dotenv()

    from pipeline.domain_detector import DomainDetector
    from pipeline.paper_retriever import PaperRetriever
    from pipeline.paper_filter import PaperFilter
    from pipeline.entity_extractor import EntityExtractor

    QUESTION = "What fungal diseases affect wheat crops in humid environments?"
    TOP_K = 3

    print()
    print("=" * 70)
    print(f"  Question : {QUESTION}")
    print("=" * 70)

    # Stage 1 — domain
    domain = DomainDetector().classify(QUESTION, method="keyword")
    print(f"\n  [1] Domain     : {domain}")

    # Stage 2 — retrieve
    raw_papers = PaperRetriever(top_n=20).retrieve(QUESTION, domain)
    print(f"  [2] Retrieved  : {len(raw_papers)} papers")

    # Stage 3 — filter
    filtered = PaperFilter().filter(raw_papers, QUESTION, top_k=TOP_K)
    print(f"  [3] Filtered   : {len(filtered)} papers")

    # Stage 4 — extract
    entities, relations = EntityExtractor().extract(filtered)
    print(f"  [4] Entities   : {len(entities)}   Relations: {len(relations)}")

    # Stage 5 — build graph
    builder = GraphBuilder()
    graph = builder.build(entities, relations)

    # --- Graph summary ---
    summary = builder.graph_summary(graph)
    print()
    print("=" * 70)
    print("  GRAPH SUMMARY")
    print("=" * 70)
    print(f"  Nodes : {summary['num_nodes']}")
    print(f"  Edges : {summary['num_edges']}")
    print()
    print("  Node types:")
    for etype, count in sorted(summary["node_types"].items()):
        print(f"    {count:3d}  {etype}")
    print()
    print("  Most connected nodes (top 5):")
    for entry in summary["most_connected_nodes"]:
        print(
            f"    [{entry['degree']:2d} edges]  {entry['label']:<30}  "
            f"({entry['type']})"
        )

    # --- Subgraph retrieval ---
    seeds = ["wheat", "fungal_disease"]
    subgraph = builder.get_subgraph(graph, seeds, depth=2)

    print()
    print("=" * 70)
    print(f"  SUBGRAPH  (seeds={seeds}, depth=2)")
    print("=" * 70)
    print(f"  Nodes : {subgraph.number_of_nodes()}")
    print(f"  Edges : {subgraph.number_of_edges()}")
    print()

    if subgraph.number_of_nodes() > 0:
        print("  Nodes in subgraph:")
        for node_id, attrs in subgraph.nodes(data=True):
            print(f"    - {attrs.get('label', node_id):<30} [{attrs.get('type', '?')}]")
        print()
        print("  Edges in subgraph:")
        for src, dst, attrs in subgraph.edges(data=True):
            src_label = subgraph.nodes[src].get("label", src)
            dst_label = subgraph.nodes[dst].get("label", dst)
            print(f"    {src_label}  --[{attrs.get('relation_type', '?')}]-->  {dst_label}")
    else:
        print("  (no nodes matched — check that entity IDs exist in the graph)")

    print()
