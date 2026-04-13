"""
Stage 6 — Graph Retriever: retrieve a query-relevant subgraph for LLM context.

Given the user question and the dynamically built knowledge graph for the
current query, this module:
  1. Identifies the graph nodes most semantically similar to the question
     (seed entities) using sentence-transformer embeddings.
  2. Expands those seeds into a subgraph via BFS traversal (GraphBuilder).
  3. Serialises the subgraph into a structured triple-text ready for the LLM.
"""

import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from config.settings import EMBEDDING_MODEL
from pipeline.embedding_service import EmbeddingService
from pipeline.graph_builder import GraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

_TOP_K_SEEDS = 5
_HOP_DEPTH = 2
# Minimum cosine similarity for a node to qualify as a seed entity.
# 0.35 keeps only nodes with a meaningful relationship to the query,
# reducing subgraph noise from unrelated BFS expansions. The fallback
# in extract_seed_entities still uses ranked_indices when no node clears
# this threshold, so sparse graphs are handled gracefully.
_MIN_SEED_SIMILARITY = 0.35


def compute_graph_confidence(graph: nx.DiGraph, seed_scores: Dict[str, float]) -> float:
    """
    Compute a calibrated graph confidence score in [0, 1].

    Signals used:
    - graph density
    - average node degree
    - number of edges
    - average seed similarity scores
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if num_nodes == 0:
        return 0.0

    density = float(nx.density(graph))
    avg_degree = float(sum(dict(graph.degree()).values()) / max(num_nodes, 1))
    degree_norm = avg_degree / (avg_degree + 2.0)
    edge_norm = min(num_edges / 20.0, 1.0)

    if seed_scores:
        seed_mean = float(np.mean(list(seed_scores.values())))
        seed_norm = max(0.0, min(1.0, (seed_mean + 1.0) / 2.0))
    else:
        seed_norm = 0.0

    confidence = (0.25 * density) + (0.25 * degree_norm) + (0.20 * edge_norm) + (0.30 * seed_norm)
    return max(0.0, min(1.0, confidence))


class GraphRetriever:
    """
    Retrieves the most query-relevant subgraph from a dynamically built
    knowledge graph and converts it to LLM-ready context text.

    Seed nodes are selected by cosine similarity between the question
    embedding and each node's label embedding.  The seeds are then expanded
    with :meth:`GraphBuilder.get_subgraph` (BFS up to ``depth`` hops).

    The graph is never stored as an instance attribute — it is always
    received as a parameter and the subgraph is returned explicitly.

    Attributes
    ----------
    embedding_model_name : str
        Name of the sentence-transformers model used for all embeddings.
    _embedding_service : EmbeddingService
        Shared embedding service used across modules.
    _graph_builder : GraphBuilder
        Instance used to delegate subgraph extraction.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """
        Initialise the graph retriever.

        Parameters
        ----------
        embedding_model_name : str
            Sentence-transformers model. Defaults to
            ``config.settings.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
        """
        self.embedding_model_name = embedding_model_name
        self._embedding_service = (
            embedding_service or EmbeddingService.get_instance(embedding_model_name)
        )
        self._graph_builder = GraphBuilder(embedding_service=self._embedding_service)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        graph: nx.DiGraph,
        plan: Dict | None = None,
        enable_reasoning_controller: bool = True,
        runtime_metrics: Dict[str, float] | None = None,
    ) -> Dict:
        """
        Main entry point: retrieve the query-relevant subgraph and context.

        Calls :meth:`extract_seed_entities`, :meth:`retrieve_subgraph`, and
        :meth:`subgraph_to_text` in sequence and packages the results.

        Parameters
        ----------
        question : str
            Natural-language question from the user.
        graph : nx.DiGraph
            Full knowledge graph built by :class:`GraphBuilder` for this query.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``seed_entities``  (list[str])  — top-5 node IDs chosen as seeds
            - ``subgraph``       (nx.DiGraph) — BFS-expanded neighbourhood
            - ``context_text``   (str)        — triple-formatted text for LLM
            - ``num_nodes``      (int)        — nodes in subgraph
            - ``num_edges``      (int)        — edges in subgraph
        """
        if runtime_metrics is not None:
            runtime_metrics["graph_retrieval_calls"] = runtime_metrics.get("graph_retrieval_calls", 0) + 1

        seed_ids, seed_scores = self.extract_seed_entities(question, graph)
        subgraph = self.retrieve_subgraph(graph, seed_ids, depth=_HOP_DEPTH)
        reasoning_paths = []
        reasoning_steps = []
        if enable_reasoning_controller:
            reasoning_paths = self.extract_ranked_paths(subgraph, seed_ids, seed_scores, top_k=5)
            reasoning_steps = self.build_reasoning_steps(question, graph, seed_scores)

        validation = self.validate_subgraph(question, subgraph, reasoning_paths, plan=plan)
        context_text = self.subgraph_to_text(subgraph, reasoning_paths=reasoning_paths)
        graph_confidence = compute_graph_confidence(subgraph, seed_scores)
        refinement_hint = self.build_refinement_hint(question=question, plan=plan, validation=validation)

        if runtime_metrics is not None:
            runtime_metrics["seed_nodes"] = runtime_metrics.get("seed_nodes", 0) + len(seed_ids)

        # Step 11 — consolidated diagnostic log
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        subgraph_density = n_edges / max(n_nodes * (n_nodes - 1), 1)
        n_paths = len(reasoning_paths)
        logger.info(
            "[GraphRetriever diagnostics] seeds=%d | subgraph: nodes=%d edges=%d "
            "density=%.4f | reasoning_paths=%d | graph_confidence=%.4f | "
            "mean_relevance=%.4f | is_valid=%s",
            len(seed_ids),
            n_nodes,
            n_edges,
            subgraph_density,
            n_paths,
            graph_confidence,
            validation.get("mean_relevance", 0.0),
            validation.get("is_valid", False),
        )

        return {
            "seed_entities": seed_ids,
            "seed_scores": seed_scores,
            "subgraph": subgraph,
            "reasoning_paths": reasoning_paths,
            "reasoning_steps": reasoning_steps,
            "graph_confidence": graph_confidence,
            "context_text": context_text,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "validation": validation,
            "needs_focus_expansion": not validation.get("is_valid", False),
            "refinement_hint": refinement_hint,
        }

    def build_refinement_hint(self, question: str, plan: Dict | None, validation: Dict) -> str:
        """Create an adaptive query refinement hint when graph evidence is insufficient."""
        plan = plan or {}
        entities = [str(e) for e in (plan.get("entities") or [])]
        dims = [str(d) for d in (plan.get("dimensions") or [])]
        intent = " ".join([str(i) for i in (plan.get("intent_signals") or [])]).strip()

        hint_parts = [question]
        if entities:
            hint_parts.append("focus entities: " + ", ".join(entities[:4]))
        if dims:
            hint_parts.append("analysis dimensions: " + ", ".join(dims[:3]))
        if intent:
            hint_parts.append("intent: " + intent)
        return " | ".join(hint_parts)

    def _node_importance(self, graph: nx.DiGraph, nid: str) -> float:
        """
        Compute a normalised structural importance score for a node.

        Combines in-degree, out-degree, and edge weights to produce a score
        in [0, 1].  Nodes that are highly connected and connected via strong
        edges score higher.  Domain-agnostic — uses graph topology only.
        """
        in_deg = graph.in_degree(nid)
        out_deg = graph.out_degree(nid)
        degree = in_deg + out_deg
        if degree == 0:
            return 0.0

        # Sum of edge weights for incident edges
        weight_sum = 0.0
        for _, _, attrs in graph.in_edges(nid, data=True):
            weight_sum += float(attrs.get("edge_weight", 1.0))
        for _, _, attrs in graph.out_edges(nid, data=True):
            weight_sum += float(attrs.get("edge_weight", 1.0))

        avg_weight = weight_sum / max(degree, 1)
        # Normalise degree by a soft cap (20 edges = full score)
        degree_norm = min(degree / 20.0, 1.0)
        return 0.5 * degree_norm + 0.5 * min(avg_weight, 1.0)

    def _hub_penalty(self, graph: nx.DiGraph, nid: str) -> float:
        """
        Return penalty factor in [0,1] for high-degree hub nodes.

        Higher values indicate generic hub-like nodes that should be down-weighted
        during retrieval to promote specific evidence paths.
        """
        num_nodes = graph.number_of_nodes()
        if num_nodes <= 2:
            return 0.0
        degree = float(graph.degree(nid))
        # Soft-normalized by graph size; robust across sparse/dense runs.
        degree_norm = min(degree / max((num_nodes - 1), 1), 1.0)
        return degree_norm

    def extract_seed_entities(
        self, question: str, graph: nx.DiGraph
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Step 8 — Hybrid seed selection: similarity + node importance.

        Select the top-k graph nodes using a hybrid score:
            hybrid_score = 0.70 * embedding_similarity + 0.30 * node_importance

        This favours nodes that are both semantically close to the query AND
        structurally central in the graph — producing more connected subgraphs.
        No hardcoded domain keywords are used for boosting.

        Parameters
        ----------
        question : str
            User question to use as the similarity anchor.
        graph : nx.DiGraph
            The full knowledge graph for this query.

        Returns
        -------
        tuple[list[str], dict[str, float]]
            Seed node IDs and their raw similarity scores.
        """
        if graph.number_of_nodes() == 0:
            logger.warning("extract_seed_entities: graph is empty — no seeds.")
            return [], {}

        node_ids = list(graph.nodes())
        node_embeddings: List[np.ndarray] = []
        missing_ids: List[str] = []
        missing_labels: List[str] = []

        for nid in node_ids:
            embedding = graph.nodes[nid].get("embedding")
            if embedding is None:
                missing_ids.append(nid)
                missing_labels.append(graph.nodes[nid].get("label", nid))
            else:
                node_embeddings.append(np.asarray(embedding, dtype=float))

        if missing_ids:
            computed = self._embedding_service.encode_batch(missing_labels)
            missing_iter = iter(computed)
            rebuilt_embeddings: List[np.ndarray] = []
            for nid in node_ids:
                existing = graph.nodes[nid].get("embedding")
                if existing is None:
                    emb = np.asarray(next(missing_iter), dtype=float)
                    graph.nodes[nid]["embedding"] = emb
                    rebuilt_embeddings.append(emb)
                else:
                    rebuilt_embeddings.append(np.asarray(existing, dtype=float))
            node_embeddings = rebuilt_embeddings

        if not node_embeddings:
            logger.warning("extract_seed_entities: node embeddings unavailable.")
            return [], {}

        query_emb = self._embedding_service.encode_text(question)
        node_embs = np.vstack(node_embeddings)      # shape (n_nodes, dim)

        # Cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        n_norms = node_embs / (np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-10)
        sim_scores = n_norms @ q_norm               # shape (n_nodes,)

        # Step 8 — Hybrid score: similarity + structural importance
        hybrid_scores = []
        importance_scores = []
        hub_penalties = []
        for i, nid in enumerate(node_ids):
            imp = self._node_importance(graph, nid)
            hub_pen = self._hub_penalty(graph, nid)
            importance_scores.append(imp)
            hub_penalties.append(hub_pen)
            # Penalty-based retrieval: reduce score of generic hub nodes.
            hybrid = (0.72 * float(sim_scores[i])) + (0.28 * imp) - (0.20 * hub_pen)
            hybrid_scores.append(hybrid)

        # Rank by hybrid score and keep top-k
        top_k = min(_TOP_K_SEEDS, len(node_ids))
        ranked_indices = np.argsort(np.asarray(hybrid_scores))[::-1][:top_k]

        filtered_ranked = [i for i in ranked_indices if float(sim_scores[i]) >= _MIN_SEED_SIMILARITY]
        if not filtered_ranked:
            filtered_ranked = list(ranked_indices)

        seed_ids = [node_ids[i] for i in filtered_ranked]
        seed_scores = {node_ids[i]: float(sim_scores[i]) for i in filtered_ranked}

        # Step 11 diagnostic logging
        logger.info("extract_seed_entities (hybrid scoring): top seeds selected:")
        for i in ranked_indices:
            logger.info(
                "  sim=%.4f  imp=%.4f  hybrid=%.4f  |  %s",
                float(sim_scores[i]),
                importance_scores[i],
                hybrid_scores[i],
                graph.nodes[node_ids[i]].get("label", node_ids[i]),
            )

        return seed_ids, seed_scores

    def decompose_question(self, question: str, max_parts: int = 4) -> List[str]:
        """Heuristically split a complex question into 2–4 sub-questions."""
        chunks = [c.strip() for c in re.split(r"\?|\.|;|,|\band\b", question, flags=re.IGNORECASE) if c.strip()]
        # Keep informative segments only.
        chunks = [c for c in chunks if len(c.split()) >= 3]
        if not chunks:
            return [question]
        return chunks[:max_parts]

    def build_reasoning_steps(
        self,
        question: str,
        graph: nx.DiGraph,
        seed_scores: Dict[str, float],
    ) -> List[str]:
        """Build a lightweight multi-step reasoning chain over local graph neighborhoods."""
        steps: List[str] = []
        sub_questions = self.decompose_question(question)

        for idx, sub_q in enumerate(sub_questions, 1):
            local_seeds, _ = self.extract_seed_entities(sub_q, graph)
            local_seeds = local_seeds[:2]
            local_subgraph = self._graph_builder.get_subgraph(graph, local_seeds, depth=1)

            if local_subgraph.number_of_edges() > 0:
                src, dst, attrs = next(iter(local_subgraph.edges(data=True)))
                src_label = local_subgraph.nodes[src].get("label", src)
                dst_label = local_subgraph.nodes[dst].get("label", dst)
                rel = attrs.get("relation_type", "related_to")
                step_text = (
                    f"Step {idx}: For '{sub_q}', the graph links "
                    f"{src_label} --[{rel}]--> {dst_label}."
                )
            elif local_seeds:
                labels = [graph.nodes[s].get("label", s) for s in local_seeds if s in graph]
                step_text = f"Step {idx}: For '{sub_q}', key entities are {', '.join(labels)}."
            else:
                step_text = f"Step {idx}: For '{sub_q}', evidence is limited in the retrieved graph."
            steps.append(step_text)

        conf = compute_graph_confidence(graph, seed_scores)
        steps.append(f"Conclusion: Graph evidence confidence is {conf:.2f}.")
        return steps

    def extract_ranked_paths(
        self,
        subgraph: nx.DiGraph,
        seed_ids: List[str],
        seed_scores: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Step 9 — Multi-hop reasoning path extraction and ranking.

        Extracts paths of length ≥ 2 between seed pairs and ranks them by a
        composite score:
            score = 0.25 * length_score       (prefer longer paths — more reasoning)
                  + 0.35 * node_relevance      (average seed similarity along path)
                  + 0.25 * edge_weight_score   (average edge weight along path)
                  + 0.15 * paper_freq_score    (multi-paper evidence support)

        No hardcoded domain terms — ranking is fully query and graph-driven.
        """
        if subgraph.number_of_nodes() == 0:
            return []

        undirected = subgraph.to_undirected()
        candidates: List[Dict] = []
        usable_seeds = [s for s in seed_ids if s in undirected]

        for src, dst in combinations(usable_seeds[:6], 2):
            if not nx.has_path(undirected, src, dst):
                continue
            try:
                path = nx.shortest_path(undirected, source=src, target=dst)
            except nx.NetworkXNoPath:
                continue

            if len(path) < 3 or len(path) > 5:
                continue

            edge_weights: List[float] = []
            node_sims: List[float] = []
            paper_ids: set = set()

            for node in path:
                node_sims.append(seed_scores.get(node, 0.0))

            for a, b in zip(path[:-1], path[1:]):
                edge_data = subgraph.get_edge_data(a, b) or subgraph.get_edge_data(b, a) or {}
                edge_weights.append(float(edge_data.get("edge_weight", 1.0)))
                pid = edge_data.get("source_paper_id", "")
                if pid and pid not in ("semantic_bridge", "cooccurrence"):
                    paper_ids.add(pid)

            # Prefer longer paths (more reasoning hops = richer explanation)
            length_score = min((len(path) - 1) / 4.0, 1.0)  # cap at 4 hops
            # Normalise mean similarity from [-1,1] → [0,1]
            node_relevance = (float(np.mean(node_sims)) + 1.0) / 2.0
            # Average edge weight (already in [0,1] range)
            edge_weight_score = float(np.mean(edge_weights)) if edge_weights else 0.5
            # Paper support: reward paths evidenced by multiple real papers
            paper_freq_score = min(len(paper_ids) / max(len(path) - 1, 1), 1.0)
            # Penalize hub-heavy paths so specific paths are preferred.
            path_hub_penalty = float(np.mean([self._hub_penalty(subgraph, n) for n in path]))

            total = (
                0.25 * length_score
                + 0.35 * node_relevance
                + 0.25 * edge_weight_score
                + 0.15 * paper_freq_score
                - 0.15 * path_hub_penalty
            )

            label_path = [subgraph.nodes[n].get("label", n) for n in path]
            candidates.append(
                {
                    "nodes": path,
                    "labels": label_path,
                    "score": float(total),
                    "length": len(path),
                    "paper_frequency": len(paper_ids),
                    "edge_weight_avg": round(edge_weight_score, 4),
                    "node_relevance_avg": round(node_relevance, 4),
                }
            )

        # Fallback: allow 1-hop evidence if no multi-hop path exists.
        if not candidates:
            for src, dst in combinations(usable_seeds[:5], 2):
                if not nx.has_path(undirected, src, dst):
                    continue
                try:
                    path = nx.shortest_path(undirected, source=src, target=dst)
                except nx.NetworkXNoPath:
                    continue
                if len(path) != 2:
                    continue
                label_path = [subgraph.nodes[n].get("label", n) for n in path]
                candidates.append(
                    {
                        "nodes": path,
                        "labels": label_path,
                        "score": 0.2,
                        "length": len(path),
                        "paper_frequency": 0,
                        "edge_weight_avg": 0.5,
                        "node_relevance_avg": 0.5,
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if candidates:
            logger.info(
                "extract_ranked_paths: %d candidates — top path score=%.4f, "
                "length=%d, paper_freq=%d",
                len(candidates),
                candidates[0]["score"],
                candidates[0]["length"],
                candidates[0]["paper_frequency"],
            )

        return candidates[:top_k]

    def retrieve_subgraph(
        self,
        graph: nx.DiGraph,
        seed_entity_ids: List[str],
        depth: int = _HOP_DEPTH,
    ) -> nx.DiGraph:
        """
        Expand seed nodes into a subgraph via BFS traversal.

        Delegates to :meth:`GraphBuilder.get_subgraph`.  If the result is
        empty (seeds not found), falls back to returning the full graph so
        that answer generation always has some context.

        Parameters
        ----------
        graph : nx.DiGraph
            The full knowledge graph for this query.
        seed_entity_ids : list[str]
            Starting node IDs for BFS expansion.
        depth : int
            Maximum number of hops from each seed. Defaults to 2.

        Returns
        -------
        nx.DiGraph
            Expanded subgraph, or the full graph if no seeds matched.
        """
        subgraph = self._graph_builder.get_subgraph(graph, seed_entity_ids, depth)

        if subgraph.number_of_nodes() == 0:
            logger.warning(
                "retrieve_subgraph: subgraph is empty — "
                "falling back to full graph as context."
            )
            return graph.copy()

        # Expand one more hop for isolated seed nodes so context stays connected.
        isolated_seed_count = 0
        for seed in seed_entity_ids:
            if seed in subgraph and subgraph.degree(seed) == 0:
                isolated_seed_count += 1
        if isolated_seed_count > 0:
            subgraph = self._graph_builder.get_subgraph(graph, seed_entity_ids, min(depth + 1, 2))

        logger.info(
            f"retrieve_subgraph: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges retrieved."
        )
        return subgraph

    def validate_subgraph(
        self,
        question: str,
        subgraph: nx.DiGraph,
        reasoning_paths: List[Dict],
        plan: Dict | None = None,
    ) -> Dict:
        """
        Step 10 — Context quality validation (domain-agnostic).

        Checks:
        1. Minimum node count (≥ 3 nodes).
        2. Minimum edge count (≥ 2 edges) — ensures meaningful paths exist.
        3. At least one multi-hop reasoning path (length ≥ 3).
        4. Embedding-based relevance: the mean cosine similarity between the
           question and node labels must exceed a minimum threshold.

        If any check fails, ``is_valid=False`` signals the pipeline to trigger
        a fallback (extra retrieval pass or abstract supplementation).

        No hardcoded domain terms — all checks are structural or embedding-based.
        """
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        has_multihop = any(int(p.get("length", 0)) >= 3 for p in reasoning_paths)

        # Embedding-based relevance check
        mean_relevance = 0.0
        if num_nodes > 0:
            labels = [
                attrs.get("label", nid)
                for nid, attrs in subgraph.nodes(data=True)
            ]
            q_emb = self._embedding_service.encode_text(question)
            node_embs = self._embedding_service.encode_batch(labels)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
            n_norms = node_embs / (np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-10)
            sims = n_norms @ q_norm
            mean_relevance = float(np.mean(sims))

        plan = plan or {}
        requirements = plan.get("graph_requirements") or {}
        min_nodes = int(requirements.get("min_nodes", 3))
        min_edges = int(requirements.get("min_edges", 2))
        min_entity_types = int(requirements.get("min_entity_types", 1))

        entity_types_present = {attrs.get("type", "Unknown") for _, attrs in subgraph.nodes(data=True)}
        reasoning_type = str(plan.get("type", "definition"))
        required_entities = list(plan.get("required_entity_types") or [])

        def _count_matching_types(type_names: List[str]) -> int:
            return sum(1 for _, attrs in subgraph.nodes(data=True) if str(attrs.get("type", "")) in set(type_names))

        method_types = {"Method / Intervention"}
        effect_types = {"Effect / Outcome"}
        concept_types = {"Concept / Entity"}
        cause_types = {"Cause / Factor"}

        complexity = str(plan.get("complexity", "moderate"))
        relevance_threshold = 0.20
        if complexity == "complex":
            relevance_threshold = 0.16
        if num_nodes <= 5 or num_edges <= 4:
            relevance_threshold = min(relevance_threshold, 0.14)

        type_specific_ok = True
        missing_requirements: List[str] = []

        if reasoning_type in {"method_process", "method/process", "method"}:
            method_count = _count_matching_types(list(method_types))
            type_specific_ok = method_count >= 4
            if not type_specific_ok:
                missing_requirements.append(f"insufficient_method_nodes:{method_count}")
        elif reasoning_type == "comparison":
            plan_entities = [str(e).strip().lower() for e in (plan.get("entities") or []) if str(e).strip()]
            labels = {str(attrs.get("label", nid)).strip().lower() for nid, attrs in subgraph.nodes(data=True)}
            type_specific_ok = all(any(entity in label for label in labels) for entity in plan_entities) if plan_entities else len(entity_types_present) >= 2
            if not type_specific_ok:
                missing_requirements.append("missing_comparison_entities")
        elif reasoning_type in {"causal_mechanism", "causal/mechanism", "causal"}:
            cause_count = _count_matching_types(list(cause_types))
            effect_count = _count_matching_types(list(effect_types))
            type_specific_ok = cause_count >= 1 and effect_count >= 1 and has_multihop
            if not type_specific_ok:
                missing_requirements.append("missing_causal_chain")
        elif reasoning_type == "definition":
            concept_count = _count_matching_types(list(concept_types))
            type_specific_ok = concept_count >= 1 and len(entity_types_present) >= 2
            if not type_specific_ok:
                missing_requirements.append("missing_definition_context")
        else:
            type_specific_ok = True

        checks = {
            "has_min_nodes": num_nodes >= 3,
            "has_min_edges": num_edges >= 2,
            "has_multihop_paths": has_multihop,
            "mean_relevance_ok": mean_relevance >= relevance_threshold,
            "plan_min_nodes": num_nodes >= min_nodes,
            "plan_min_edges": num_edges >= min_edges,
            "plan_entity_type_diversity": len(entity_types_present) >= min_entity_types,
            "plan_type_specific": type_specific_ok,
        }
        is_valid = all(checks.values())
        if not checks["plan_min_nodes"]:
            missing_requirements.append("insufficient_nodes")
        if not checks["plan_min_edges"]:
            missing_requirements.append("insufficient_edges")
        if not checks["plan_entity_type_diversity"]:
            missing_requirements.append("insufficient_entity_type_diversity")
        if not checks["has_multihop_paths"]:
            missing_requirements.append("missing_multihop_paths")
        if not checks["plan_type_specific"] and "plan_type_specific" not in missing_requirements:
            missing_requirements.append("plan_type_specific")

        logger.info(
            "validate_subgraph: nodes=%d edges=%d multihop=%s "
            "mean_relevance=%.4f is_valid=%s",
            num_nodes, num_edges, has_multihop, mean_relevance, is_valid,
        )

        return {
            "question": question,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "has_multihop_paths": has_multihop,
            "mean_relevance": round(mean_relevance, 4),
            "relevance_threshold": round(relevance_threshold, 4),
            "checks": checks,
            "missing_requirements": missing_requirements,
            "is_valid": is_valid,
        }

    def subgraph_to_text(self, subgraph: nx.DiGraph, reasoning_paths: List[Dict] | None = None) -> str:
        """
        Serialise the subgraph into a structured triple-text for the LLM.

        Each directed edge is formatted as a readable triple:
        ``[source_label] --[relation_type]--> [target_label]``

        Triples are grouped by ``relation_type`` under labelled sections.
        Isolated nodes (no edges) are listed separately so they are not lost.

        Parameters
        ----------
        subgraph : nx.DiGraph
            Retrieved subgraph to serialise.

        Returns
        -------
        str
            Formatted context string ready to inject into the LLM prompt.
            Returns an empty string if the subgraph has no nodes.
        """
        if subgraph.number_of_nodes() == 0:
            logger.warning("subgraph_to_text: empty subgraph — returning empty context.")
            return ""

        # Group triples by relation type
        grouped: Dict[str, List[str]] = defaultdict(list)
        nodes_with_edges: set = set()

        for src, dst, attrs in subgraph.edges(data=True):
            relation = attrs.get("relation_type", "related_to")
            src_label = subgraph.nodes[src].get("label", src)
            dst_label = subgraph.nodes[dst].get("label", dst)
            triple = f"  {src_label} --[{relation}]--> {dst_label}"
            grouped[relation].append(triple)
            nodes_with_edges.update([src, dst])

        lines: List[str] = ["=== Knowledge Graph Context ===", ""]

        # Emit top reasoning paths first.
        if reasoning_paths:
            lines.append("[top_reasoning_paths]")
            for i, path_info in enumerate(reasoning_paths, 1):
                path_text = " -> ".join(path_info.get("labels", []))
                lines.append(f"  Path {i} (score={path_info.get('score', 0.0):.2f}): {path_text}")
            lines.append("")

        # Emit grouped triples
        for relation_type in sorted(grouped.keys()):
            lines.append(f"[{relation_type}]")
            lines.extend(grouped[relation_type])
            lines.append("")

        # Emit isolated nodes (entities with no edges in the subgraph)
        isolated = [
            nid for nid in subgraph.nodes() if nid not in nodes_with_edges
        ]
        if isolated:
            lines.append("[relevant entities (no direct relations)]")
            for nid in isolated:
                attrs = subgraph.nodes[nid]
                label = attrs.get("label", nid)
                etype = attrs.get("type", "Unknown")
                lines.append(f"  {label}  ({etype})")
            lines.append("")

        lines.append("=== End of Context ===")
        context_text = "\n".join(lines)

        logger.info(
            f"subgraph_to_text: serialised {subgraph.number_of_edges()} triples "
            f"across {len(grouped)} relation type(s)."
        )
        return context_text

# ---------------------------------------------------------------------------
# Full pipeline smoke-test:
# DomainDetector -> PaperRetriever -> PaperFilter ->
# EntityExtractor -> GraphBuilder -> GraphRetriever
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
    TOP_K = 5

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

    # Stage 4 — extract entities
    entities, relations = EntityExtractor().extract(filtered)
    print(f"  [4] Entities   : {len(entities)}   Relations: {len(relations)}")

    # Stage 5 — build graph
    builder = GraphBuilder()
    graph = builder.build(entities, relations)
    print(f"  [5] Graph      : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Stage 6 — retrieve subgraph + context
    retriever = GraphRetriever()
    result = retriever.retrieve(QUESTION, graph)

    print()
    print("=" * 70)
    print("  SEED ENTITIES SELECTED")
    print("=" * 70)
    for seed_id in result["seed_entities"]:
        label = graph.nodes[seed_id].get("label", seed_id) if seed_id in graph else seed_id
        etype = graph.nodes[seed_id].get("type", "?") if seed_id in graph else "?"
        print(f"  - {label:<30}  [{etype}]  (id: {seed_id})")

    print()
    print("=" * 70)
    print(f"  SUBGRAPH: {result['num_nodes']} nodes / {result['num_edges']} edges")
    print("=" * 70)

    print()
    print("=" * 70)
    print("  CONTEXT TEXT (injected into LLM prompt)")
    print("=" * 70)
    print(result["context_text"])
