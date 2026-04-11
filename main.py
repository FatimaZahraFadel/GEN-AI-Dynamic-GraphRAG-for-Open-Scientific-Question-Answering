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
import time
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
from pipeline.embedding_service import EmbeddingService
from pipeline.paper_retriever import PaperRetriever
from pipeline.paper_filter import PaperFilter
from pipeline.entity_extractor import EntityExtractor
from pipeline.graph_builder import GraphBuilder
from pipeline.graph_retriever import GraphRetriever
from pipeline.answer_generator import AnswerGenerator

# --- Import config ---
from config.settings import (
    EMBEDDING_MODEL,
    ENTITY_EXTRACTION_MAX_WORKERS,
    FAST_MODE_TOP_K_PAPERS,
    FAST_MODE_TOP_N_PAPERS,
    GRAPH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_EXTRA_TOP_N,
    MAX_GRAPH_EXPANSION_ITERS,
    TOP_N_PAPERS,
)
from utils.session_state import SessionState

# --- Import logger ---
from utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_SERVICE = EmbeddingService.get_instance(EMBEDDING_MODEL)
_SESSION_STATE = SessionState()


def run_pipeline(
    query: str,
    fast_mode: bool = True,
    session_state: SessionState | None = None,
    enable_reasoning: bool = True,
    enable_confidence: bool = True,
    enable_expansion: bool = True,
    return_metrics: bool = True,
    # Backward-compatible aliases:
    return_details: bool | None = None,
    enable_reasoning_controller: bool | None = None,
    enable_confidence_retrieval: bool | None = None,
    enable_iterative_expansion: bool | None = None,
) -> str | dict:
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
    t_start = time.perf_counter()
    runtime_metrics: dict = {
        "llm_calls": 0,
        "retrieved_papers": 0,
        "graph_expansion_iterations": 0,
    }
    emb_before = EmbeddingService.get_metrics()

    # Compatibility mapping for older call sites.
    if enable_reasoning_controller is not None:
        enable_reasoning = enable_reasoning_controller
    if enable_confidence_retrieval is not None:
        enable_confidence = enable_confidence_retrieval
    if enable_iterative_expansion is not None:
        enable_expansion = enable_iterative_expansion

    session = session_state or _SESSION_STATE
    query_embedding = _EMBEDDING_SERVICE.encode_text(query)

    # ------------------------------------------------------------------
    # Stage 1: Domain Detection
    # ------------------------------------------------------------------
    detector = DomainDetector(embedding_service=_EMBEDDING_SERVICE)
    domain = detector.classify(query, method="keyword")
    logger.info(f"Detected domain: {domain}")

    same_domain = session.domain == domain
    reuse_graph = same_domain and session.should_reuse_graph(query_embedding)
    logger.info(
        "Session reuse decision: same_domain=%s, similarity=%.3f, reuse_graph=%s",
        same_domain,
        session.last_similarity,
        reuse_graph,
    )

    top_n = FAST_MODE_TOP_N_PAPERS if (fast_mode and reuse_graph) else TOP_N_PAPERS
    top_k = FAST_MODE_TOP_K_PAPERS if (fast_mode and reuse_graph) else 10

    # ------------------------------------------------------------------
    # Stage 2: Paper Retrieval
    # ------------------------------------------------------------------
    if (
        enable_confidence
        and fast_mode
        and reuse_graph
        and session.should_skip_retrieval()
    ):
        papers = session.get_cached_papers()
        logger.info("Fast mode enabled: skipped retrieval and reused %d cached papers.", len(papers))
    else:
        retriever = PaperRetriever(top_n=top_n)
        retrieved_papers = retriever.retrieve(query, domain)
        runtime_metrics["retrieved_papers"] += len(retrieved_papers)
        session.cache_papers(retrieved_papers)
        papers = session.get_cached_papers() if reuse_graph else retrieved_papers

    logger.info(f"Retrieved {len(papers)} candidate papers")

    # ------------------------------------------------------------------
    # Stage 3: Paper Filtering
    # ------------------------------------------------------------------
    paper_filter = PaperFilter(embedding_service=_EMBEDDING_SERVICE)
    filtered_papers = paper_filter.filter(papers, query, top_k=top_k)
    logger.info(f"Filtered to {len(filtered_papers)} papers")

    # ------------------------------------------------------------------
    # Stage 4: Entity Extraction
    # Pass the question so extraction is focused on relevant entities.
    # ------------------------------------------------------------------
    extractor = EntityExtractor()
    entities, relations = extractor.extract(
        filtered_papers,
        question=query,
        extraction_cache=session.extraction_cache,
        max_workers=ENTITY_EXTRACTION_MAX_WORKERS,
        runtime_metrics=runtime_metrics,
    )
    logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations")

    # ------------------------------------------------------------------
    # Stage 5: Graph Construction
    # ------------------------------------------------------------------
    builder = GraphBuilder(embedding_service=_EMBEDDING_SERVICE)
    if reuse_graph and session.graph is not None:
        graph = session.graph
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)
        logger.info("Graph incrementally updated from session cache.")
    else:
        graph = builder.build(entities, relations)

    logger.info(
        f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # ------------------------------------------------------------------
    # Stage 6: Graph Retrieval + confidence-aware iterative expansion
    # ------------------------------------------------------------------
    graph_retriever = GraphRetriever(embedding_service=_EMBEDDING_SERVICE)
    retrieval = graph_retriever.retrieve(
        query,
        graph,
        enable_reasoning_controller=enable_reasoning,
        runtime_metrics=runtime_metrics,
    )

    expansion_iters = 0
    retrieval_skips = int(
        bool(
            enable_confidence
            and fast_mode
            and reuse_graph
            and session.should_skip_retrieval()
        )
    )

    while (
        enable_confidence
        and enable_expansion
        and retrieval.get("graph_confidence", 0.0) < GRAPH_CONFIDENCE_THRESHOLD
        and expansion_iters < MAX_GRAPH_EXPANSION_ITERS
    ):
        logger.info(
            "Low graph confidence detected (%.3f < %.3f). Expansion iteration %d/%d.",
            retrieval.get("graph_confidence", 0.0),
            GRAPH_CONFIDENCE_THRESHOLD,
            expansion_iters + 1,
            MAX_GRAPH_EXPANSION_ITERS,
        )
        extra_retriever = PaperRetriever(top_n=LOW_CONFIDENCE_EXTRA_TOP_N)
        extra_papers = extra_retriever.retrieve(query, domain)
        runtime_metrics["retrieved_papers"] += len(extra_papers)
        session.cache_papers(extra_papers)

        candidate_pool = session.get_cached_papers() if session.get_cached_papers() else filtered_papers
        filtered_papers = paper_filter.filter(candidate_pool, query, top_k=max(top_k, FAST_MODE_TOP_K_PAPERS + 2))

        entities, relations = extractor.extract(
            filtered_papers,
            question=query,
            extraction_cache=session.extraction_cache,
            max_workers=ENTITY_EXTRACTION_MAX_WORKERS,
            runtime_metrics=runtime_metrics,
        )
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)

        retrieval = graph_retriever.retrieve(
            query,
            graph,
            enable_reasoning_controller=enable_reasoning,
            runtime_metrics=runtime_metrics,
        )
        expansion_iters += 1
        runtime_metrics["graph_expansion_iterations"] = expansion_iters

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
        reasoning_steps=retrieval.get("reasoning_steps", []),
        reasoning_paths=retrieval.get("reasoning_paths", []),
        runtime_metrics=runtime_metrics,
    )
    logger.info("Answer generated successfully")

    session.domain = domain
    session.graph = graph
    session.cache_papers(filtered_papers)
    session.record_query(query, query_embedding)

    emb_after = EmbeddingService.get_metrics()
    runtime_metrics["embedding_calls"] = emb_after["embedding_calls"] - emb_before["embedding_calls"]
    runtime_metrics["embedded_texts"] = emb_after["embedded_texts"] - emb_before["embedded_texts"]
    runtime_metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 4)

    graph_density = 0.0
    avg_degree = 0.0
    if graph.number_of_nodes() > 0:
        graph_density = retrieval["subgraph"].number_of_edges() / max(
            retrieval["subgraph"].number_of_nodes() * max(retrieval["subgraph"].number_of_nodes() - 1, 1),
            1,
        )
        avg_degree = sum(dict(retrieval["subgraph"].degree()).values()) / max(
            retrieval["subgraph"].number_of_nodes(),
            1,
        )

    result_payload = {
        "answer": answer_dict["answer"],
        "scores": {},
        "metrics": {
            "runtime": runtime_metrics.get("runtime_seconds", 0.0),
            "llm_calls": int(runtime_metrics.get("llm_calls", 0)),
            "embedding_calls": int(runtime_metrics.get("embedding_calls", 0)),
            "retrieved_papers": int(runtime_metrics.get("retrieved_papers", 0)),
            "graph_expansion_iters": int(runtime_metrics.get("graph_expansion_iterations", 0)),
        },
        "graph_metrics": {
            "num_nodes": retrieval["subgraph"].number_of_nodes(),
            "num_edges": retrieval["subgraph"].number_of_edges(),
            "density": round(graph_density, 6),
            "avg_degree": round(float(avg_degree), 4),
        },
        "confidence": float(retrieval.get("graph_confidence", 0.0)),
        # Extended details for app/evaluator compatibility
        "question": query,
        "domain": domain,
        "graph": graph,
        "filtered_papers": filtered_papers,
        "retrieval": retrieval,
        "answer_dict": answer_dict,
        "expansion_iters": expansion_iters,
        "retrieval_skips": retrieval_skips,
        "graph_confidence": retrieval.get("graph_confidence", 0.0),
        "runtime_metrics": runtime_metrics,
    }

    if return_metrics or bool(return_details):
        return result_payload

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
