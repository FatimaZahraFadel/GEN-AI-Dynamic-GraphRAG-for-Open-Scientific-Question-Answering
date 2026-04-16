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
7. AnswerGenerator   — generate a grounded LLM answer conditioned on the subgraph
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
from pipeline.embedding_service import EmbeddingService
from pipeline.paper_retriever import PaperRetriever
from pipeline.paper_filter import PaperFilter
from pipeline.entity_extractor import EntityExtractor
from pipeline.graph_builder import GraphBuilder
from pipeline.graph_retriever import GraphRetriever
from pipeline.answer_generator import AnswerGenerator
from pipeline.answer_compiler import compile_answer, assess_graph_quality, CompileResult
from pipeline.adaptive_retry import (
    plan_retry_actions,
    merge_graphs,
    format_retry_summary,
)
from pipeline.domain_detector import DomainDetector
from pipeline.query_planner import QueryPlanner
from pipeline.confidence_scorer import score_confidence_and_coverage

# --- Import config ---
from config.settings import (
    ENTITY_EXTRACTION_FAST_MODEL,
    ENTITY_EXTRACTION_MAX_CONCURRENCY,
    ENTITY_EXTRACTION_MODEL,
    EMBEDDING_MODEL,
    ENTITY_EXTRACTION_MAX_WORKERS,
    FAST_MODE_TOP_K_PAPERS,
    FAST_MODE_TOP_N_PAPERS,
    GRAPH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_EXTRA_TOP_N,
    MAX_GRAPH_EXPANSION_ITERS,
    TOP_N_PAPERS,
    USE_FAST_EXTRACTOR_IN_FAST_MODE,
)
from utils.session_state import SessionState

# --- Import logger ---
from utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDING_SERVICE = EmbeddingService.get_instance(EMBEDDING_MODEL)
_SESSION_STATE = SessionState()
_MODEL_HEALTH_CACHE: dict[str, bool] = {}
_DOMAIN_DETECTOR = DomainDetector(embedding_service=_EMBEDDING_SERVICE)

_ADAPTIVE_ANSWER_TOP_K_TARGET = 7
_ADAPTIVE_EXTRACTION_SUCCESS_THRESHOLD = 0.80


def _precheck_extractor_model(model_name: str) -> bool:
    """Check extractor model availability once per process."""
    cached = _MODEL_HEALTH_CACHE.get(model_name)
    if cached is not None:
        return cached

    healthy, message = EntityExtractor.precheck_model_health(model_name)
    _MODEL_HEALTH_CACHE[model_name] = healthy
    if healthy:
        logger.info("Extractor model precheck passed for '%s'.", model_name)
    else:
        logger.warning("Extractor model precheck failed for '%s': %s", model_name, message)
    return healthy


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
    top_n_override: int | None = None,
    top_k_override: int | None = None,
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
    logger.info("Query: %s", query)
    t_start = time.perf_counter()
    runtime_metrics: dict = {
        "llm_calls": 0,
        "retrieved_papers": 0,
        "graph_expansion_iterations": 0,
        "planning_seconds": 0.0,
        "domain_detection_seconds": 0.0,
        "retrieval_seconds": 0.0,
        "filtering_seconds": 0.0,
        "entity_extraction_seconds": 0.0,
        "graph_build_seconds": 0.0,
        "graph_retrieval_seconds": 0.0,
        "answer_generation_seconds": 0.0,
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
    # Stage 1: Understand + Plan
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    planner = QueryPlanner()
    plan_obj = planner.plan(query)
    plan = plan_obj.to_dict()
    runtime_metrics["planning_seconds"] += time.perf_counter() - t_stage
    runtime_metrics["llm_calls"] += planner.llm_calls

    plan_type_to_intent = {
        "definition": "fact",
        "comparison": "comparison",
        "method/process": "process",
        "causal/mechanism": "cause",
        "optimization/recommendation": "solution",
    }
    intent = plan_type_to_intent.get(plan.get("type", "definition"), _detect_query_intent(query))

    t_domain = time.perf_counter()
    try:
        domain = _DOMAIN_DETECTOR.classify_robust(query)
        subdomain = _DOMAIN_DETECTOR.classify_subdomain(query, domain)
    except Exception as exc:
        logger.warning("Domain detection failed (%s); falling back to embedding classification.", exc)
        try:
            domain = _DOMAIN_DETECTOR.embedding_classify(query)
            subdomain = domain.lower()
        except Exception:
            domain = _DOMAIN_DETECTOR.domains[0] if _DOMAIN_DETECTOR.domains else "general"
            subdomain = domain.lower()
    runtime_metrics["domain_detection_seconds"] = time.perf_counter() - t_domain
    retrieval_domain = domain

    logger.info("Planning completed: type=%s, entities=%d, complexity=%s", plan.get("type"), len(plan.get("entities", [])), plan.get("complexity"))
    logger.info("Domain detection completed: domain=%s, subdomain=%s", domain, subdomain)

    same_domain = session.domain == domain
    reuse_graph = same_domain and session.should_reuse_graph(query_embedding)
    logger.info(
        "Session reuse decision: same_domain=%s, similarity=%.3f, reuse_graph=%s",
        same_domain,
        session.last_similarity,
        reuse_graph,
    )

    top_n = top_n_override if top_n_override is not None else (
        FAST_MODE_TOP_N_PAPERS if (fast_mode and reuse_graph) else TOP_N_PAPERS
    )
    top_k = top_k_override if top_k_override is not None else (
        FAST_MODE_TOP_K_PAPERS if (fast_mode and reuse_graph) else 10
    )
    # Adaptive pruning control: reduce over-pruning for complex queries.
    if plan.get("complexity") == "complex":
        top_k = max(top_k, 7)
    elif plan.get("complexity") == "simple":
        top_k = max(4, min(top_k, 6))

    # ------------------------------------------------------------------
    # Stage 2: Paper Retrieval (with LLM query expansion + dual retrieval)
    # ------------------------------------------------------------------
    if (
        enable_confidence
        and fast_mode
        and reuse_graph
        and session.should_skip_retrieval()
    ):
        t_stage = time.perf_counter()
        papers = session.get_cached_papers()
        logger.info("Fast mode enabled: skipped retrieval and reused %d cached papers.", len(papers))
        expanded_query = query  # no expansion needed when skipping retrieval
        runtime_metrics["retrieval_seconds"] += time.perf_counter() - t_stage
    else:
        t_stage = time.perf_counter()
        # Query expansion is embedded inside PaperRetriever.retrieve()
        # (Step 1 & 2).  We capture the expanded query for logging/expansion
        # loop re-use by creating the retriever once and calling expand_query.
        retriever = PaperRetriever(top_n=top_n, use_query_expansion=True)
        plan_queries = planner.build_retrieval_queries(plan_obj)
        planning_seed_query = " ".join([query] + plan_queries[:2]).strip()
        expanded_query = retriever.expand_query(planning_seed_query)
        logger.info("Expanded query for pipeline: '%s'", expanded_query[:120])
        retrieved_papers = retriever.retrieve(
            query,
            retrieval_domain,
            expanded_question=expanded_query,
            plan=plan,
        )
        runtime_metrics["retrieved_papers"] += len(retrieved_papers)
        session.cache_papers(retrieved_papers)
        papers = session.get_cached_papers() if reuse_graph else retrieved_papers
        runtime_metrics["retrieval_seconds"] += time.perf_counter() - t_stage

    retrieved_papers_snapshot = list(papers)

    logger.info("Retrieved %d candidate papers", len(papers))
    logger.info("[Step 11 diagnostics] original_query='%s'", query[:80])
    logger.info("[Step 11 diagnostics] expanded_query='%s'", expanded_query[:120])

    # ------------------------------------------------------------------
    # Stage 3: Paper Filtering
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    paper_filter = PaperFilter(embedding_service=_EMBEDDING_SERVICE)
    filtered_papers = paper_filter.filter(papers, query, intent=intent, top_k=top_k)
    evidence_assessment = paper_filter.assess_evidence_consistency(query, filtered_papers)
    runtime_metrics["filtering_seconds"] += time.perf_counter() - t_stage
    logger.info("Filtered to %d papers", len(filtered_papers))
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
    t_stage = time.perf_counter()
    extraction_model = (
        ENTITY_EXTRACTION_FAST_MODEL
        if (fast_mode and USE_FAST_EXTRACTOR_IN_FAST_MODE)
        else ENTITY_EXTRACTION_MODEL
    )
    if not _precheck_extractor_model(extraction_model):
        logger.warning(
            "Configured extractor model '%s' unavailable; using '%s'.",
            extraction_model,
            ENTITY_EXTRACTION_MODEL,
        )
        extraction_model = ENTITY_EXTRACTION_MODEL
    extractor = EntityExtractor(model=extraction_model)
    entities, relations = extractor.extract(
        filtered_papers,
        question=query,
        plan=plan,
        extraction_cache=session.extraction_cache,
        max_workers=max(ENTITY_EXTRACTION_MAX_WORKERS, ENTITY_EXTRACTION_MAX_CONCURRENCY),
        runtime_metrics=runtime_metrics,
    )
    runtime_metrics["entity_extraction_seconds"] += time.perf_counter() - t_stage
    logger.info("Extracted %d entities, %d relations", len(entities), len(relations))

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
    t_stage = time.perf_counter()
    builder = GraphBuilder(embedding_service=_EMBEDDING_SERVICE)
    if reuse_graph and session.graph is not None:
        graph = session.graph
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)
        builder.optimize_for_query(graph, query, plan=plan, query_entities=plan.get("entities", []))
        logger.info("Graph incrementally updated from session cache.")
    else:
        graph = builder.build(entities, relations, question=query, plan=plan)

    logger.info(
        f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    runtime_metrics["graph_build_seconds"] += time.perf_counter() - t_stage

    # ------------------------------------------------------------------
    # Stage 6: Graph Retrieval + confidence-aware iterative expansion
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()
    graph_retriever = GraphRetriever(embedding_service=_EMBEDDING_SERVICE)
    retrieval = graph_retriever.retrieve(
        query,
        graph,
        plan=plan,
        enable_reasoning_controller=enable_reasoning,
        runtime_metrics=runtime_metrics,
    )

    # Adaptive top-k for answer grounding only:
    # increase paper context size only when first-pass extraction quality is high
    # and the retrieved graph is valid. This improves answer authority without
    # forcing larger extraction loads on unstable runs.
    filtered_papers_for_answer = filtered_papers
    extraction_attempted = int(runtime_metrics.get("extraction_attempted", 0))
    extraction_successful = int(runtime_metrics.get("extraction_successful", 0))
    extraction_success_rate = (
        (extraction_successful / extraction_attempted)
        if extraction_attempted > 0
        else 1.0
    )
    can_expand_answer_context = (
        top_k < _ADAPTIVE_ANSWER_TOP_K_TARGET
        and extraction_success_rate >= _ADAPTIVE_EXTRACTION_SUCCESS_THRESHOLD
        and bool(retrieval.get("validation", {}).get("is_valid", False))
    )
    if can_expand_answer_context:
        filtered_papers_for_answer = paper_filter.filter(
            papers,
            query,
            intent=intent,
            top_k=_ADAPTIVE_ANSWER_TOP_K_TARGET,
        )
        runtime_metrics["adaptive_answer_top_k_applied"] = 1
        runtime_metrics["adaptive_answer_top_k"] = _ADAPTIVE_ANSWER_TOP_K_TARGET
        logger.info(
            "Adaptive answer top_k applied: %d -> %d (extraction_success_rate=%.2f, graph_valid=%s)",
            top_k,
            _ADAPTIVE_ANSWER_TOP_K_TARGET,
            extraction_success_rate,
            retrieval.get("validation", {}).get("is_valid", False),
        )
    else:
        runtime_metrics["adaptive_answer_top_k_applied"] = 0
        runtime_metrics["adaptive_answer_top_k"] = top_k
    runtime_metrics["graph_retrieval_seconds"] += time.perf_counter() - t_stage

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

    validation_state = retrieval.get("validation", {}) or {}
    confidence_needs_expansion = not bool(validation_state.get("is_valid", False))
    seen_refinement_queries: set[str] = set()

    max_expansion_iters = min(MAX_GRAPH_EXPANSION_ITERS, 2)
    while (
        enable_confidence
        and enable_expansion
        and (
            (
                retrieval.get("graph_confidence", 0.0) < GRAPH_CONFIDENCE_THRESHOLD
                and confidence_needs_expansion
            )
            or retrieval.get("needs_focus_expansion", False)
        )
        and expansion_iters < max_expansion_iters
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
            max_expansion_iters,
        )
        extra_retriever = PaperRetriever(top_n=LOW_CONFIDENCE_EXTRA_TOP_N)
        t_stage = time.perf_counter()
        refinement_query = retrieval.get("refinement_hint") or expanded_query
        refinement_key = (refinement_query or "").strip().lower()
        if refinement_key in seen_refinement_queries:
            logger.info("Stopping expansion loop: repeated refinement query '%s'.", refinement_query[:100])
            break
        seen_refinement_queries.add(refinement_key)

        extra_papers = extra_retriever.retrieve(refinement_query, retrieval_domain, plan=plan)
        runtime_metrics["retrieved_papers"] += len(extra_papers)
        session.cache_papers(extra_papers)
        runtime_metrics["retrieval_seconds"] += time.perf_counter() - t_stage

        candidate_pool = session.get_cached_papers() if session.get_cached_papers() else filtered_papers
        t_stage = time.perf_counter()
        filtered_papers = paper_filter.filter(
            candidate_pool,
            expanded_query,
            intent=intent,
            top_k=max(top_k, FAST_MODE_TOP_K_PAPERS + 2),
        )
        evidence_assessment = paper_filter.assess_evidence_consistency(expanded_query, filtered_papers)
        runtime_metrics["filtering_seconds"] += time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        entities, relations = extractor.extract(
            filtered_papers,
            question=expanded_query,
            plan=plan,
            extraction_cache=session.extraction_cache,
            max_workers=ENTITY_EXTRACTION_MAX_WORKERS,
            runtime_metrics=runtime_metrics,
        )
        runtime_metrics["entity_extraction_seconds"] += time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        builder.add_entities(graph, entities)
        builder.add_relations(graph, relations)
        builder.optimize_for_query(graph, expanded_query, plan=plan, query_entities=plan.get("entities", []))
        runtime_metrics["graph_build_seconds"] += time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        retrieval = graph_retriever.retrieve(
            expanded_query,
            graph,
            plan=plan,
            enable_reasoning_controller=enable_reasoning,
            runtime_metrics=runtime_metrics,
        )
        validation_state = retrieval.get("validation", {}) or {}
        confidence_needs_expansion = not bool(validation_state.get("is_valid", False))
        runtime_metrics["graph_retrieval_seconds"] += time.perf_counter() - t_stage
        expansion_iters += 1
        runtime_metrics["graph_expansion_iterations"] = expansion_iters

    logger.info(
        f"Retrieved subgraph: {retrieval['num_nodes']} nodes, "
        f"{retrieval['num_edges']} edges"
    )

    # ------------------------------------------------------------------
    # Stage 7: Answer Generation via AnswerGenerator (LLM-grounded)
    # The graph compiler is used only as a quality/coverage validator.
    # ------------------------------------------------------------------
    t_stage = time.perf_counter()

    # Limit papers to what the generator actually consumes so metrics are truthful.
    papers_for_answer = filtered_papers_for_answer[:5]

    generator = AnswerGenerator()
    answer_result = generator.generate(
        question=query,
        context_text=retrieval["context_text"],
        papers=papers_for_answer,
        plan=plan,
        subgraph=retrieval["subgraph"],
        reasoning_steps=retrieval.get("reasoning_steps", []),
        reasoning_paths=retrieval.get("reasoning_paths", []),
        intent=intent,
        evidence_assessment=evidence_assessment,
        low_confidence_mode=low_confidence_mode,
        runtime_metrics=runtime_metrics,
    )
    runtime_metrics["llm_calls"] = runtime_metrics.get("llm_calls", 0) + generator.llm_calls
    runtime_metrics["answer_generation_seconds"] = time.perf_counter() - t_stage

    # Run the compiler as a validation/quality helper (not the answer source).
    graph_quality = assess_graph_quality(retrieval["subgraph"], plan)

    logger.info(
        "Answer generated: model=%s papers=%d graph_quality=%s low_confidence=%s",
        answer_result.get("model", "unknown"),
        answer_result.get("num_papers_used", 0),
        graph_quality,
        answer_result.get("low_confidence", False),
    )

    answer_dict = {
        "answer": answer_result["answer"],
        "model": answer_result.get("model", "unknown"),
        "num_papers_used": answer_result.get("num_papers_used", len(papers_for_answer)),
        "low_confidence": answer_result.get("low_confidence", low_confidence_mode),
        "quality": graph_quality,
    }

    session.domain = domain
    session.graph = graph
    session.cache_papers(filtered_papers)
    session.record_query(query, query_embedding)

    emb_after = EmbeddingService.get_metrics()
    runtime_metrics["embedding_calls"] = emb_after["embedding_calls"] - emb_before["embedding_calls"]
    runtime_metrics["embedded_texts"] = emb_after["embedded_texts"] - emb_before["embedded_texts"]
    runtime_metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 4)

    confidence_report = score_confidence_and_coverage(
        subgraph=retrieval["subgraph"],
        papers=filtered_papers_for_answer,
        reasoning_paths=retrieval.get("reasoning_paths", []),
        seed_scores=retrieval.get("seed_scores", {}),
        query_plan=plan,
    )

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
        "confidence": float(
            confidence_report.get("confidence", retrieval.get("graph_confidence", 0.0))
        ),
        "coverage": float(
            confidence_report.get("coverage", 0.0)
        ),
        "warnings": confidence_report.get("warnings", []),
        # Extended details for app/evaluator compatibility
        "question": query,
        "query_plan": plan,
        "expanded_query": expanded_query,
        "domain": domain,
        "subdomain": subdomain,
        "intent": intent,
        "retrieved_papers": retrieved_papers_snapshot,
        "graph": graph,
        "filtered_papers": filtered_papers,
        "retrieval": retrieval,
        "answer_dict": answer_dict,
        "evidence_assessment": evidence_assessment,
        "low_confidence_mode": low_confidence_mode,
        "expansion_iters": expansion_iters,
        "retrieval_skips": retrieval_skips,
        "graph_confidence": retrieval.get("graph_confidence", 0.0),
        "confidence_report": confidence_report,
        "runtime_metrics": runtime_metrics,
        "quality": answer_dict.get("quality", "unknown"),
    }

    if return_metrics or bool(return_details):
        return result_payload

    return answer_dict["answer"]


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
