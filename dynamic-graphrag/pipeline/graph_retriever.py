"""
Stage 6 — Graph Retriever: retrieve a query-relevant subgraph for LLM context.

Given the user question and the dynamically built knowledge graph for the
current query, this module:
  1. Identifies the graph nodes most semantically similar to the question
     (seed entities) using sentence-transformer embeddings.
  2. Expands those seeds into a subgraph via BFS traversal (GraphBuilder).
  3. Serialises the subgraph into a structured triple-text ready for the LLM.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL
from pipeline.graph_builder import GraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

_TOP_K_SEEDS = 5
_HOP_DEPTH = 2


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
    _model : SentenceTransformer or None
        Cached model instance; ``None`` until first use.
    _graph_builder : GraphBuilder
        Instance used to delegate subgraph extraction.
    """

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL) -> None:
        """
        Initialise the graph retriever.

        Parameters
        ----------
        embedding_model_name : str
            Sentence-transformers model. Defaults to
            ``config.settings.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
        """
        self.embedding_model_name = embedding_model_name
        self._model: SentenceTransformer | None = None
        self._graph_builder = GraphBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, graph: nx.DiGraph) -> Dict:
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
        seed_ids = self.extract_seed_entities(question, graph)
        subgraph = self.retrieve_subgraph(graph, seed_ids, depth=_HOP_DEPTH)
        context_text = self.subgraph_to_text(subgraph)

        return {
            "seed_entities": seed_ids,
            "subgraph": subgraph,
            "context_text": context_text,
            "num_nodes": subgraph.number_of_nodes(),
            "num_edges": subgraph.number_of_edges(),
        }

    def extract_seed_entities(
        self, question: str, graph: nx.DiGraph
    ) -> List[str]:
        """
        Select the top-5 graph nodes most semantically similar to the question.

        The question and every node label are embedded with
        ``all-MiniLM-L6-v2``.  Cosine similarity is computed between the
        question embedding and each node's label embedding.  The top-5 node
        IDs (by similarity) are returned as seed points for subgraph expansion.

        Parameters
        ----------
        question : str
            User question to use as the similarity anchor.
        graph : nx.DiGraph
            The full knowledge graph for this query.

        Returns
        -------
        list[str]
            Up to ``_TOP_K_SEEDS`` (5) node IDs ranked by descending similarity.
        """
        if graph.number_of_nodes() == 0:
            logger.warning("extract_seed_entities: graph is empty — no seeds.")
            return []

        model = self._get_model()

        node_ids = list(graph.nodes())
        node_labels = [
            graph.nodes[nid].get("label", nid) for nid in node_ids
        ]

        # Encode question + all labels in a single batch
        all_texts = [question] + node_labels
        embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

        query_emb = embeddings[0]       # shape (dim,)
        node_embs = embeddings[1:]      # shape (n_nodes, dim)

        # Cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        n_norms = node_embs / (np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-10)
        scores = n_norms @ q_norm       # shape (n_nodes,)

        # Rank by score and keep top-k
        top_k = min(_TOP_K_SEEDS, len(node_ids))
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        seed_ids = [node_ids[i] for i in ranked_indices]

        logger.info("extract_seed_entities: top seeds selected:")
        for i in ranked_indices:
            logger.info(
                f"  score={scores[i]:.4f}  |  id={node_ids[i]}  "
                f"label={graph.nodes[node_ids[i]].get('label', node_ids[i])}"
            )

        return seed_ids

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

        logger.info(
            f"retrieve_subgraph: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges retrieved."
        )
        return subgraph

    def subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """
        Return the cached SentenceTransformer, loading it on first call.

        Returns
        -------
        SentenceTransformer
            Loaded sentence-transformer model.
        """
        if self._model is None:
            logger.info(
                f"Loading SentenceTransformer '{self.embedding_model_name}'…"
            )
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model


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
