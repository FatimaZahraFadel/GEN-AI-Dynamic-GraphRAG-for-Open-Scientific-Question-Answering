"""
Dynamic GraphRAG — main entry point.

This module wires together all pipeline stages and demonstrates the full
end-to-end flow for answering a scientific question via graph-augmented
retrieval.  Each stage is called in sequence; intermediate results are
passed through to the next stage.

Pipeline stages
---------------
1. DomainDetector   — classify the query into one or more scientific domains
2. PaperRetriever   — fetch candidate papers from Semantic Scholar / OpenAlex
3. PaperFilter      — embed, score, and rank papers; keep top-N
4. EntityExtractor  — extract entities and relations from each paper abstract
5. GraphBuilder     — build a knowledge graph from extracted entities/relations
6. GraphRetriever   — retrieve a query-relevant subgraph
7. AnswerGenerator  — generate a grounded answer conditioned on the subgraph
"""

import os
import sys
import io
import warnings

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fix Windows Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# --- Import pipeline stages ---
from pipeline.domain_detector import DomainDetector
from pipeline.paper_retriever import PaperRetriever
from pipeline.paper_filter import PaperFilter
from pipeline.entity_extractor import EntityExtractor
from pipeline.graph_builder import GraphBuilder
from pipeline.graph_retriever import GraphRetriever
from pipeline.answer_generator import AnswerGenerator

# --- Import config ---
from config.settings import DOMAINS, TOP_N_PAPERS, MAX_GRAPH_NODES, EMBEDDING_MODEL

# --- Import logger ---
from utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(query: str) -> str:
    """
    Execute the full Dynamic GraphRAG pipeline for a single user query.

    Parameters
    ----------
    query : str
        Natural-language scientific question.

    Returns
    -------
    str
        Generated answer grounded in the retrieved knowledge graph.
    """

    logger.info("Starting Dynamic GraphRAG pipeline")
    logger.info(f"Query: {query}")

    # ------------------------------------------------------------------
    # Stage 1: Domain Detection
    # ------------------------------------------------------------------
    detector = DomainDetector()
    domain = detector.classify(query, method="keyword")
    logger.info(f"Detected domain: {domain}")

    # ------------------------------------------------------------------
    # Stage 2: Paper Retrieval
    # ------------------------------------------------------------------
    retriever = PaperRetriever(top_n=TOP_N_PAPERS)
    papers = retriever.retrieve(query, domain)
    logger.info(f"Retrieved {len(papers)} candidate papers")

    # ------------------------------------------------------------------
    # Stage 3: Paper Filtering
    # ------------------------------------------------------------------
    paper_filter = PaperFilter()
    filtered_papers = paper_filter.filter(papers, query, top_k=10)
    logger.info(f"Filtered to {len(filtered_papers)} papers")

    # ------------------------------------------------------------------
    # Stage 4: Entity Extraction
    # Pass the question so extraction is focused on relevant entities.
    # ------------------------------------------------------------------
    extractor = EntityExtractor()
    entities, relations = extractor.extract(filtered_papers, question=query)
    logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations")

    # ------------------------------------------------------------------
    # Stage 5: Graph Construction
    # ------------------------------------------------------------------
    builder = GraphBuilder()
    graph = builder.build(entities, relations)
    logger.info(
        f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # ------------------------------------------------------------------
    # Stage 6: Graph Retrieval
    # ------------------------------------------------------------------
    graph_retriever = GraphRetriever()
    retrieval = graph_retriever.retrieve(query, graph)
    logger.info(
        f"Retrieved subgraph: {retrieval['num_nodes']} nodes, "
        f"{retrieval['num_edges']} edges"
    )

    # ------------------------------------------------------------------
    # Stage 7: Answer Generation
    # Pass the subgraph so sparse-graph detection can supplement context.
    # ------------------------------------------------------------------
    generator = AnswerGenerator()
    answer_dict = generator.generate(
        question=query,
        context_text=retrieval["context_text"],
        papers=filtered_papers,
        subgraph=retrieval["subgraph"],
    )
    logger.info("Answer generated successfully")

    return generator.format_answer(answer_dict)


if __name__ == "__main__":
    sample_query = (
        "What fungal diseases affect wheat crops in humid environments?"
    )

    print()
    print("=" * 70)
    print("  DYNAMIC GraphRAG — Full Pipeline")
    print("=" * 70)
    print(f"  Question: {sample_query}")
    print()

    answer = run_pipeline(sample_query)
    print(answer)
