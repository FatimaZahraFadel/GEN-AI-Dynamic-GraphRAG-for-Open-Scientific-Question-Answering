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
import re
import warnings

import numpy as np

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


def _extract_query_terms(question: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9\-]{4,}\b", (question or "").lower())
    stop = {
        "what", "which", "when", "where", "does", "that", "this", "from",
        "with", "into", "about", "their", "there", "these", "could", "would",
        "should", "might", "face", "problem", "problems", "made", "make",
    }
    return {token for token in tokens if token not in stop}


def _detect_query_intent(question: str) -> str:
    """Classify intent into fact/cause/process/solution using lightweight rules."""
    q = (question or "").lower()

    solution_keywords = ["fix", "improve", "reduce", "mitigate", "optimize", "optimization", "solution", "remedy"]
    list_keywords = ["what are", "list", "techniques", "methods", "strategies", "approaches", "best practices"]
    cause_keywords = ["why", "cause", "causes", "due to", "reason", "driver", "because"]
    process_keywords = ["how", "process", "steps", "workflow", "procedure", "implement"]

    if any(k in q for k in list_keywords):
        if any(k in q for k in solution_keywords):
            return "solution"
        return "list"
    if any(k in q for k in solution_keywords):
        return "solution"
    if any(k in q for k in cause_keywords):
        return "cause"
    if any(k in q for k in process_keywords):
        return "process"
    return "fact"


def _apply_graph_intent_bias(entities, relations, filtered_papers) -> None:
    """Attach intent-aware weights to entities/relations from supporting papers."""
    paper_weight: dict[str, float] = {}
    for paper in filtered_papers:
        pid = paper.paper_id
        if not pid:
            continue
        intent_score = float(getattr(paper, "_intent_score", 0.0))
        rel_score = float(getattr(paper, "relevance_score", 0.0))
        paper_weight[pid] = 1.0 + 0.6 * min(max(intent_score, 0.0), 1.0) + 0.3 * min(max(rel_score, 0.0), 1.0)

    for entity in entities:
        setattr(entity, "intent_weight", paper_weight.get(entity.source_paper_id, 1.0))

    for relation in relations:
        setattr(relation, "intent_weight", paper_weight.get(relation.source_paper_id, 1.0))


def _strict_relevance_gate(
    question: str,
    entities,
    relations,
    embedding_service: EmbeddingService,
    strict: bool,
):
    """Prune entities/relations that drift away from the user question."""
    if not strict or len(entities) <= 4:
        return entities, relations

    query_terms = _extract_query_terms(question)
    entity_labels = [entity.label for entity in entities]
    embeddings = embedding_service.encode_batch([question] + entity_labels)
    query_emb = embeddings[0]
    entity_embs = embeddings[1:]

    query_norm = query_emb / (float((query_emb @ query_emb) ** 0.5) + 1e-10)
    entity_norms = entity_embs / (
        (np.linalg.norm(entity_embs, axis=1, keepdims=True) + 1e-10)
    )
    semantic_scores = (entity_norms @ query_norm).astype(float)
    semantic_scores = (semantic_scores + 1.0) / 2.0

    lexical_scores = []
    for label in entity_labels:
        label_tokens = set(re.findall(r"\b[a-z0-9\-]{4,}\b", label.lower()))
        if not query_terms:
            lexical_scores.append(0.0)
            continue
        overlap = len(label_tokens & query_terms)
        lexical_scores.append(overlap / max(len(query_terms), 1))

    combined_scores = [0.7 * s + 0.3 * l for s, l in zip(semantic_scores, lexical_scores)]

    base_threshold = 0.32 if strict else 0.24
    percentile_threshold = float(np.percentile(np.array(combined_scores, dtype=float), 55 if strict else 40))
    threshold = max(base_threshold, percentile_threshold)

    ranked_indices = sorted(range(len(entities)), key=lambda i: combined_scores[i], reverse=True)
    kept_indices = [i for i in ranked_indices if combined_scores[i] >= threshold]

    if len(kept_indices) < 5:
        kept_indices = ranked_indices[:min(5, len(ranked_indices))]

    if query_terms:
        for i, label in enumerate(entity_labels):
            label_text = label.lower()
            if any(term in label_text for term in query_terms):
                if i not in kept_indices:
                    kept_indices.append(i)

    kept_indices = sorted(set(kept_indices), key=lambda i: combined_scores[i], reverse=True)
    kept_ids = {entities[i].id for i in kept_indices}
    filtered_entities = [entity for entity in entities if entity.id in kept_ids]
    filtered_relations = [
        relation for relation in relations
        if relation.source_id in kept_ids and relation.target_id in kept_ids
    ]

    logger.info(
        "Strict relevance gate: kept %d/%d entities, %d/%d relations (threshold=%.3f).",
        len(filtered_entities),
        len(entities),
        len(filtered_relations),
        len(relations),
        threshold,
    )

    return filtered_entities, filtered_relations


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
    subdomain = detector.classify_subdomain(query, domain)
    intent = _detect_query_intent(query)
    logger.info(f"Detected domain: {domain}")
    logger.info(f"Detected subdomain: {subdomain}")
    logger.info(f"Detected intent: {intent}")

    retrieval_domain = domain if subdomain == "general" else f"{domain} {subdomain}"

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
    # Stage 2: Paper Retrieval (with LLM query expansion + dual retrieval)
    # ------------------------------------------------------------------
    if (
        enable_confidence
        and fast_mode
        and reuse_graph
        and session.should_skip_retrieval()
    ):
        papers = session.get_cached_papers()
        logger.info("Fast mode enabled: skipped retrieval and reused %d cached papers.", len(papers))
        expanded_query = query  # no expansion needed when skipping retrieval
    else:
        # Query expansion is embedded inside PaperRetriever.retrieve()
        # (Step 1 & 2).  We capture the expanded query for logging/expansion
        # loop re-use by creating the retriever once and calling expand_query.
        retriever = PaperRetriever(top_n=top_n, use_query_expansion=True)
        expanded_query = retriever.expand_query(query)
        logger.info("Expanded query for pipeline: '%s'", expanded_query[:120])
        retrieved_papers = retriever.retrieve(query, retrieval_domain, expanded_question=expanded_query)
        runtime_metrics["retrieved_papers"] += len(retrieved_papers)
        session.cache_papers(retrieved_papers)
        papers = session.get_cached_papers() if reuse_graph else retrieved_papers

    logger.info(f"Retrieved {len(papers)} candidate papers")
    logger.info("[Step 11 diagnostics] original_query='%s'", query[:80])
    logger.info("[Step 11 diagnostics] expanded_query='%s'", expanded_query[:120])

    # ------------------------------------------------------------------
    # Stage 3: Paper Filtering
    # ------------------------------------------------------------------
    paper_filter = PaperFilter(embedding_service=_EMBEDDING_SERVICE)
    filtered_papers = paper_filter.filter(papers, query, intent=intent, top_k=top_k)
    evidence_assessment = paper_filter.assess_evidence_consistency(query, filtered_papers)
    logger.info(f"Filtered to {len(filtered_papers)} papers")
    logger.info(
        "Intent-relevant papers after filter: %d",
        paper_filter.last_intent_relevant_count,
    )
    logger.info(
        "Evidence consistency: score=%.3f, consistent=%s",
        evidence_assessment.get("consistency_score", 0.0),
        evidence_assessment.get("is_consistent", False),
    )

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

    strict_gate = bool(
        (not evidence_assessment.get("is_consistent", False))
        or evidence_assessment.get("consistency_score", 0.0) < 0.25
    )
    if strict_gate:
        entities, relations = _strict_relevance_gate(
            query,
            entities,
            relations,
            _EMBEDDING_SERVICE,
            strict=True,
        )
        logger.info(
            "Strict gate active: low_confidence_mode=%s, evidence_consistent=%s",
            True,
            evidence_assessment.get("is_consistent", False),
        )

    _apply_graph_intent_bias(entities, relations, filtered_papers)

    # ------------------------------------------------------------------
    # Stage 5: Graph Construction
    # ------------------------------------------------------------------
    builder = GraphBuilder(embedding_service=_EMBEDDING_SERVICE)
    if reuse_graph and session.graph is not None:
        graph = session.graph
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)
        builder.optimize_for_query(graph, query)
        logger.info("Graph incrementally updated from session cache.")
    else:
        graph = builder.build(entities, relations, question=query)

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

    low_confidence_mode = bool(
        strict_gate
        or retrieval.get("graph_confidence", 0.0) < GRAPH_CONFIDENCE_THRESHOLD
        or retrieval.get("needs_focus_expansion", False)
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
        and (
            retrieval.get("graph_confidence", 0.0) < GRAPH_CONFIDENCE_THRESHOLD
            or retrieval.get("needs_focus_expansion", False)
        )
        and expansion_iters < MAX_GRAPH_EXPANSION_ITERS
    ):
        # Use the LLM-expanded query from Stage 2 (domain-agnostic).
        # If validation failed and we need focus, rely on the already-expanded
        # query rather than any hardcoded keywords.
        if retrieval.get("needs_focus_expansion", False):
            logger.info(
                "Graph validation failed; using LLM-expanded query for next retrieval pass: '%s'",
                expanded_query[:100],
            )

        logger.info(
            "Low graph confidence detected (%.3f < %.3f). Expansion iteration %d/%d.",
            retrieval.get("graph_confidence", 0.0),
            GRAPH_CONFIDENCE_THRESHOLD,
            expansion_iters + 1,
            MAX_GRAPH_EXPANSION_ITERS,
        )
        extra_retriever = PaperRetriever(top_n=LOW_CONFIDENCE_EXTRA_TOP_N)
        extra_papers = extra_retriever.retrieve(expanded_query, retrieval_domain)
        runtime_metrics["retrieved_papers"] += len(extra_papers)
        session.cache_papers(extra_papers)

        candidate_pool = session.get_cached_papers() if session.get_cached_papers() else filtered_papers
        filtered_papers = paper_filter.filter(
            candidate_pool,
            expanded_query,
            intent=intent,
            top_k=max(top_k, FAST_MODE_TOP_K_PAPERS + 2),
        )
        evidence_assessment = paper_filter.assess_evidence_consistency(expanded_query, filtered_papers)

        entities, relations = extractor.extract(
            filtered_papers,
            question=expanded_query,
            extraction_cache=session.extraction_cache,
            max_workers=ENTITY_EXTRACTION_MAX_WORKERS,
            runtime_metrics=runtime_metrics,
        )
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)
        builder.optimize_for_query(graph, expanded_query)

        retrieval = graph_retriever.retrieve(
            expanded_query,
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
        intent=intent,
        evidence_assessment=evidence_assessment,
        low_confidence_mode=low_confidence_mode,
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
        "expanded_query": expanded_query,
        "domain": domain,
        "subdomain": subdomain,
        "intent": intent,
        "graph": graph,
        "filtered_papers": filtered_papers,
        "retrieval": retrieval,
        "answer_dict": answer_dict,
        "evidence_assessment": evidence_assessment,
        "low_confidence_mode": low_confidence_mode,
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
