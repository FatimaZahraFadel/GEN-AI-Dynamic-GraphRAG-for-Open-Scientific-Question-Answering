"""
Stage 5 — Graph Builder: build a temporary in-memory knowledge graph.

The graph is built fresh for every user query from the entities and relations
produced by EntityExtractor.  It is never stored to a database or reused
across queries.  All methods receive and return the graph explicitly so that
no state leaks between pipeline runs.
"""

from collections import deque
from typing import Dict, List

import networkx as nx

from models.graph_node import Entity, Relation
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        entities: List[Entity],
        relations: List[Relation],
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

        self.add_entities(graph, entities)
        self.add_relations(graph, relations)

        logger.info(
            f"Graph built: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges."
        )
        return graph

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
            graph.add_node(
                entity.id,
                label=entity.label,
                type=entity.type,
                source_paper_id=entity.source_paper_id,
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
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    source_paper_id=rel.source_paper_id,
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
