"""
Stage 4 — Entity Extractor: extract entities and relationships from paper abstracts.

For each Paper, the abstract is sent to a Groq LLM which returns a JSON object
containing a list of entities and a list of directed relations between them.
The results are converted into Entity and Relation dataclass instances that
feed directly into the graph-building stage.
"""

import json
import hashlib
import asyncio
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from groq import Groq
import requests

from config.settings import (
    ENTITY_EXTRACTION_MAX_WORKERS,
    ENTITY_EXTRACTION_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_EXTRACTION_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    USE_OLLAMA_EXTRACTION_FALLBACK,
    USE_OLLAMA_PRIMARY,
)
from models.graph_node import Entity, Relation
from models.paper import Paper
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Controlled vocabularies — domain-agnostic
# ---------------------------------------------------------------------------
# These types are intentionally broad so the same extractor works for
# Agriculture, Geoscience, Computer Science, Supply Chain, and Environment
# without requiring domain-specific prompts.
#
# Mapping examples (old → new):
#   Crop / Organism           → Concept / Entity
#   Disease / Condition       → Problem / Condition
#   Treatment / Method        → Method / Intervention
#   Environment / Location    → Context / Location
#   Cause / Factor            → Cause / Factor      (unchanged)
#   Effect / Outcome          → Effect / Outcome     (unchanged)

ENTITY_TYPES = [
    "Concept / Entity",      # organisms, materials, systems, datasets, models, genes, …
    "Problem / Condition",   # diseases, failures, risks, deficiencies, bugs, anomalies, …
    "Method / Intervention", # techniques, algorithms, treatments, protocols, tools, …
    "Context / Location",    # environments, regions, platforms, settings, domains, …
    "Cause / Factor",        # drivers, risk factors, inputs, constraints, stressors, …
    "Effect / Outcome",      # results, metrics, impacts, performance, consequences, …
]

RELATION_TYPES = [
    "affects",       # factor/entity changes another variable or system behaviour
    "mitigates",     # method/intervention reduces a problem or negative outcome
    "occurs_in",     # condition/process appears in a context/environment
    "caused_by",     # condition/outcome attributable to a factor
    "leads_to",      # upstream cause/mechanism produces a downstream outcome
    "detected_by",   # entity/condition identified by a method or signal
    "studied_in",    # concept is investigated within a context or domain
    "correlates_with",  # statistical or observed co-variation between entities
    "depends_on",    # entity or process requires another entity/condition
    "part_of",       # entity is a component or sub-system of another
]

_MAX_EXTRACTION_RETRIES = 2        # 2 attempts per prompt; reduces worst-case latency
_BASE_BACKOFF_SECONDS = 0.5        # 0.5s → 1.0s (was 1s → 2s → 4s)
_BACKOFF_JITTER_MAX_SECONDS = 0.3
_STRICT_JSON_RETRY_MAX = 1


def _slugify(text: str) -> str:
    """
    Convert a label to a lowercase, underscore-separated slug.

    All non-alphanumeric characters (except hyphens) are replaced by
    underscores; leading/trailing underscores are stripped; runs of
    multiple underscores are collapsed.

    Parameters
    ----------
    text : str
        Raw label string, e.g. ``"Rust Disease"``.

    Returns
    -------
    str
        Slug string, e.g. ``"rust_disease"``.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


class EntityExtractor:
    """
    Extracts scientific entities and their relationships from paper abstracts
    using a Groq LLM (llama3-8b-8192).

    The extractor sends each abstract through a structured JSON-extraction
    prompt.  Responses are parsed safely; papers whose LLM call fails or
    returns malformed JSON are skipped with a warning.

    Attributes
    ----------
    model : str
        Groq model identifier. Defaults to ``"llama3-8b-8192"``.
    _client : Groq
        Authenticated Groq API client (lazy-initialised on first use).
    """

    def __init__(self, model: str = ENTITY_EXTRACTION_MODEL) -> None:
        """
        Initialise the entity extractor.

        Parameters
        ----------
        model : str
            Groq model to use for extraction. Defaults to
            ``"llama3-8b-8192"``.
        """
        self.model = model
        self._fallback_model = ENTITY_EXTRACTION_MODEL
        self._client: Groq | None = None
        self._use_ollama_primary = USE_OLLAMA_PRIMARY
        self._ollama_fallback_enabled = USE_OLLAMA_EXTRACTION_FALLBACK
        self._ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self._ollama_model = OLLAMA_EXTRACTION_MODEL
        self._ollama_timeout = OLLAMA_TIMEOUT_SECONDS
        self._groq_rate_limited = False
        self.llm_calls: int = 0
        self._stats_lock = Lock()
        self._retry_count: int = 0
        self._json_retry_count: int = 0
        self._fallback_count: int = 0

    def _paper_cache_key(self, paper: Paper) -> str:
        """Build a stable cache key from paper identity or content fingerprint."""
        if paper.paper_id:
            return f"pid:{paper.paper_id}"
        payload = f"{paper.title}\n{paper.abstract or ''}".strip().encode("utf-8", errors="ignore")
        return f"fp:{hashlib.sha1(payload).hexdigest()}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        papers: List[Paper],
        question: str = "",
        plan: Dict | None = None,
        extraction_cache: Dict[str, Tuple[List[Entity], List[Relation]]] | None = None,
        max_workers: int = ENTITY_EXTRACTION_MAX_WORKERS,
        runtime_metrics: Dict[str, float] | None = None,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from all papers.

        Calls :meth:`extract_from_abstract` for each paper, deduplicates
        entities by ``id`` (keeping the first occurrence), and collects all
        relations.

        Parameters
        ----------
        papers : list[Paper]
            Filtered papers from the previous pipeline stage.
        question : str
            Original user question used to focus extraction. Optional but
            recommended — improves relevance of extracted entities.

        Returns
        -------
        tuple[list[Entity], list[Relation]]
            ``(all_entities, all_relations)`` — deduplicated entities and the
            full set of relations across all papers.
        """
        all_entities: List[Entity] = []
        all_relations: List[Relation] = []
        seen_entity_ids: set[str] = set()
        extraction_cache = extraction_cache if extraction_cache is not None else {}

        papers_to_extract: List[Paper] = []
        cache_hits = 0

        for paper in papers:
            cache_key = self._paper_cache_key(paper)
            if cache_key in extraction_cache:
                entities, relations = extraction_cache[cache_key]
                cache_hits += 1
            elif paper.paper_id and f"pid:{paper.paper_id}" in extraction_cache:
                entities, relations = extraction_cache[f"pid:{paper.paper_id}"]
                cache_hits += 1
            else:
                papers_to_extract.append(paper)
                continue

            for entity in entities:
                if entity.id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity.id)
            all_relations.extend(relations)

        if papers_to_extract:
            logger.info(
                "Entity extraction cache hits: %d/%d. Extracting %d new paper(s).",
                cache_hits,
                len(papers),
                len(papers_to_extract),
            )

        if runtime_metrics is not None:
            runtime_metrics["llm_calls"] = runtime_metrics.get("llm_calls", 0) + len(papers_to_extract)

        # Reset per-run extraction telemetry.
        self._retry_count = 0
        self._json_retry_count = 0
        self._fallback_count = 0
        self._groq_rate_limited = False

        # Parallel extraction for new papers.
        extracted_results: List[Tuple[str, List[Entity], List[Relation]]] = []
        worker_count = max(1, int(max_workers))

        if papers_to_extract and worker_count > 1:
            extracted_results = self._extract_async_concurrent(
                papers=papers_to_extract,
                question=question,
                plan=plan,
                max_concurrency=worker_count,
            )
        else:
            for paper in papers_to_extract:
                entities, relations = self.extract_from_abstract(paper, question, plan=plan)
                extracted_results.append((paper.paper_id, entities, relations))

        for paper_id, entities, relations in extracted_results:
            if paper_id:
                extraction_cache[f"pid:{paper_id}"] = (entities, relations)

            for paper in papers_to_extract:
                if paper.paper_id == paper_id:
                    extraction_cache[self._paper_cache_key(paper)] = (entities, relations)
                    break

            for entity in entities:
                if entity.id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity.id)

            all_relations.extend(relations)

        successful = sum(1 for _, ents, rels in extracted_results if len(ents) > 0 or len(rels) > 0)
        attempted = len(papers_to_extract)
        success_rate = (successful / attempted) if attempted > 0 else 1.0

        logger.info(
            "Extraction diagnostics: attempted=%d, successful=%d, success_rate=%.2f, retries=%d, json_retries=%d, fallback_used=%d",
            attempted,
            successful,
            success_rate,
            self._retry_count,
            self._json_retry_count,
            self._fallback_count,
        )

        if runtime_metrics is not None:
            runtime_metrics["extraction_attempted"] = runtime_metrics.get("extraction_attempted", 0) + attempted
            runtime_metrics["extraction_successful"] = runtime_metrics.get("extraction_successful", 0) + successful
            runtime_metrics["extraction_retries"] = runtime_metrics.get("extraction_retries", 0) + self._retry_count
            runtime_metrics["extraction_fallback_used"] = runtime_metrics.get("extraction_fallback_used", 0) + self._fallback_count

        logger.info(
            f"Extraction complete: {len(all_entities)} unique entities, "
            f"{len(all_relations)} relations across {len(papers)} papers."
        )
        return all_entities, all_relations

    def _extract_async_concurrent(
        self,
        papers: List[Paper],
        question: str,
        plan: Dict | None,
        max_concurrency: int,
    ) -> List[Tuple[str, List[Entity], List[Relation]]]:
        """Run extraction using async scheduling with bounded concurrency."""

        async def _runner() -> List[Tuple[str, List[Entity], List[Relation]]]:
            sem = asyncio.Semaphore(max(1, max_concurrency))

            async def _one(paper: Paper) -> Tuple[str, List[Entity], List[Relation]]:
                async with sem:
                    entities, relations = await asyncio.to_thread(self.extract_from_abstract, paper, question, plan)
                    return (paper.paper_id, entities, relations)

            tasks = [asyncio.create_task(_one(p)) for p in papers]
            return await asyncio.gather(*tasks)

        try:
            return asyncio.run(_runner())
        except RuntimeError:
            # Fallback for environments where an event loop is already running.
            with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
                futures = {executor.submit(self.extract_from_abstract, p, question, plan): p for p in papers}
                out: List[Tuple[str, List[Entity], List[Relation]]] = []
                for future in as_completed(futures):
                    paper = futures[future]
                    entities, relations = future.result()
                    out.append((paper.paper_id, entities, relations))
                return out

    def extract_from_abstract(
        self, paper: Paper, question: str = "", plan: Dict | None = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a single paper's abstract.

        Sends the abstract to the Groq LLM via :meth:`build_extraction_prompt`,
        parses the JSON response, and converts raw dicts into :class:`Entity`
        and :class:`Relation` objects.

        If the LLM call fails or returns unparseable JSON, the error is logged
        and ``([], [])`` is returned so the pipeline can continue.

        Parameters
        ----------
        paper : Paper
            The paper to process.
        question : str
            Original user question used to focus extraction on relevant
            entities and relations. Optional but improves quality.

        Returns
        -------
        tuple[list[Entity], list[Relation]]
            Entities and relations extracted from this paper's abstract.
        """
        if not paper.abstract or len(paper.abstract.strip()) < 50:
            logger.warning(
                f"Skipping paper '{paper.paper_id}': abstract too short."
            )
            return [], []

        # Truncate abstract to 1 500 characters before building the prompt.
        # Abstracts longer than this rarely add extraction value but consume
        # extra tokens, increasing TPM usage and rate-limit pressure.
        abstract_text = paper.abstract.strip()[:1500]

        full_prompt = self.build_extraction_prompt(abstract_text, question, plan=plan)
        # 600 max_tokens is enough for 8-15 entities + relations in JSON.
        raw_text = self._call_llm_with_retries(full_prompt, max_tokens=600, call_kind="full", paper_id=paper.paper_id)
        data = self._parse_json(raw_text) if raw_text else None

        # Retry once with stricter JSON instructions if parsing fails.
        if data is None:
            for _ in range(_STRICT_JSON_RETRY_MAX):
                with self._stats_lock:
                    self._json_retry_count += 1
                strict_prompt = self.build_strict_extraction_prompt(abstract_text, question, plan=plan)
                strict_raw = self._call_llm_with_retries(
                    strict_prompt,
                    max_tokens=500,
                    call_kind="strict-json",
                    paper_id=paper.paper_id,
                )
                data = self._parse_json(strict_raw) if strict_raw else None
                if data is not None:
                    break

        # Fallback extraction with simpler prompt if full extraction still fails.
        if data is None:
            with self._stats_lock:
                self._fallback_count += 1
            fallback_prompt = self.build_fallback_extraction_prompt(abstract_text, question, plan=plan)
            fallback_raw = self._call_llm_with_retries(
                fallback_prompt,
                max_tokens=350,
                call_kind="fallback",
                paper_id=paper.paper_id,
            )
            data = self._parse_json(fallback_raw) if fallback_raw else None

        if data is None:
            logger.warning(
                "Extraction failed after retries and fallback for paper '%s'; using deterministic heuristic extraction.",
                paper.paper_id,
            )
            return self._heuristic_extract_from_abstract(abstract_text, paper.paper_id, question, plan)

        entities = self._parse_entities(data.get("entities", []), paper.paper_id)
        relations = self._parse_relations(
            data.get("relations", []), paper.paper_id
        )

        logger.info(
            f"Paper '{paper.paper_id[:30]}': "
            f"{len(entities)} entities, {len(relations)} relations extracted."
        )
        return entities, relations

    def build_extraction_prompt(self, abstract: str, question: str = "", plan: Dict | None = None) -> str:
        """
        Build the structured JSON-extraction prompt for the LLM.

        The prompt lists the allowed entity types and relation types and
        instructs the model to return **only** a raw JSON object — no
        markdown, no explanation.  When a question is provided it is placed
        at the top of the prompt so the model prioritises entities and
        relations directly relevant to the question.

        Parameters
        ----------
        abstract : str
            The paper abstract to analyse.
        question : str
            Original user question. When non-empty, guides the extraction
            toward disease, treatment, cause, and effect entities most
            relevant to the question topic.

        Returns
        -------
        str
            Fully formatted prompt string.
        """
        entity_types_str = ", ".join(f'"{t}"' for t in ENTITY_TYPES)
        relation_types_str = ", ".join(f'"{t}"' for t in RELATION_TYPES)

        reasoning_type = str((plan or {}).get("type", "definition"))
        required_entity_types = ", ".join((plan or {}).get("required_entity_types", []))
        required_relation_types = ", ".join((plan or {}).get("required_relation_types", []))

        task_focus = {
            "method/process": "Prioritize actions, techniques, procedures, and step dependencies.",
            "causal/mechanism": "Prioritize causes, mechanisms, mediators, and downstream effects.",
            "comparison": "Prioritize measurable properties, comparative dimensions, and outcomes.",
            "optimization/recommendation": "Prioritize strategies, constraints, trade-offs, and expected impact.",
            "definition": "Prioritize core concepts, attributes, and contextual qualifiers.",
        }.get(reasoning_type, "Prioritize entities and relations most relevant to answering the query.")

        question_block = ""
        if question:
            question_block = (
                f"The user is trying to answer: {question}\n"
                f"IMPORTANT: Extract entities and relations that are directly relevant to this question intent.\n"
                f"Extract only entities and relations that are directly relevant to the user's query: {question}. Ignore peripheral administrative or tangential scientific data.\n"
                f"Prioritize terms that explain mechanisms, causes, methods, fixes, outcomes, and constraints when present.\n\n"
            )

        plan_block = (
            f"Reasoning type: {reasoning_type}\n"
            f"Extraction focus: {task_focus}\n"
            f"Required entity types: {required_entity_types or 'none specified'}\n"
            f"Preferred relation types: {required_relation_types or 'none specified'}\n\n"
        )

        return f"""You are an expert scientific information extraction system.

{question_block}{plan_block}Extract ALL entities and relationships from the abstract below.

Entity type guidance (domain-agnostic — use for any scientific field):
- Concept / Entity: organisms, materials, datasets, models, systems, genes, chemicals, algorithms, compounds
- Problem / Condition: diseases, failures, risks, deficiencies, anomalies, bottlenecks, disturbances
- Method / Intervention: techniques, algorithms, treatments, protocols, tools, procedures, experiments
- Context / Location: environments, regions, platforms, operating conditions, domains, ecosystems, settings
- Cause / Factor: drivers, risk factors, inputs, stressors, constraints, triggers, predictors
- Effect / Outcome: results, metrics, impacts, consequences, yields, accuracies, losses, improvements

Relation type guidance:
- affects: factor/entity changes another variable or system behaviour
- mitigates: method/intervention reduces a problem or negative outcome
- occurs_in: condition/process appears in a context/environment
- caused_by: condition/outcome attributable to a factor
- leads_to: upstream cause/mechanism produces a downstream outcome
- detected_by: entity/condition identified by a method or signal
- studied_in: concept is investigated within a context or domain
- correlates_with: statistical or observed co-variation between entities
- depends_on: entity or process requires another entity/condition
- part_of: entity is a component or sub-system of another

IMPORTANT: Maximize entity extraction. Include many entities (target 8-15), not just 1-3. Connect related entities with relations.

Return ONLY a valid JSON object in this exact format — no markdown, no explanation, no extra text:
{{
  "entities": [
    {{"label": "Wheat", "type": "Concept / Entity"}},
    {{"label": "Rust disease", "type": "Problem / Condition"}},
    {{"label": "Puccinia", "type": "Concept / Entity"}},
    {{"label": "Humid conditions", "type": "Context / Location"}},
    {{"label": "Fungicide application", "type": "Method / Intervention"}}
  ],
  "relations": [
    {{"source": "Rust disease", "target": "Wheat", "relation": "affects"}},
    {{"source": "Humid conditions", "target": "Rust disease", "relation": "leads_to"}},
    {{"source": "Fungicide application", "target": "Rust disease", "relation": "mitigates"}}
  ]
}}

Abstract:
{abstract}"""

    def build_strict_extraction_prompt(self, abstract: str, question: str = "", plan: Dict | None = None) -> str:
        """Build a strict JSON-only extraction prompt for parse-recovery retries."""
        question_line = f"Question: {question}\n" if question else ""
        reasoning_line = f"Reasoning type: {(plan or {}).get('type', 'definition')}\n"
        return f"""Return ONLY valid minified JSON (no markdown, no prose, no trailing text).
{question_line}{reasoning_line}
Schema:
{{
  "entities": [{{"label": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Allowed relation values: {', '.join(RELATION_TYPES)}
Allowed type values: {', '.join(ENTITY_TYPES)}

Abstract:
{abstract}
"""

    def build_fallback_extraction_prompt(self, abstract: str, question: str = "", plan: Dict | None = None) -> str:
        """Build a lightweight fallback extraction prompt with fewer outputs."""
        question_line = f"Question: {question}\n" if question else ""
        reasoning_line = f"Reasoning type: {(plan or {}).get('type', 'definition')}\n"
        return f"""Extract a minimal, reliable set of entities and relations.
{question_line}{reasoning_line}
Rules:
- Return only 4-8 high-confidence entities.
- Return only 0-6 high-confidence relations.
- If unsure, return fewer items rather than speculative ones.
- Output strictly valid JSON with this schema:
{{
  "entities": [{{"label": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Abstract:
{abstract}
"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_entities(
        self, raw_entities: list, paper_id: str
    ) -> List[Entity]:
        """
        Convert raw entity dicts from the LLM JSON into Entity objects.

        Entries missing a ``label`` field are silently skipped.  The ``type``
        field is accepted as-is; if absent it defaults to ``"Unknown"``.

        Parameters
        ----------
        raw_entities : list
            List of ``{"label": ..., "type": ...}`` dicts from the LLM.
        paper_id : str
            ID of the source paper (attached to every Entity).

        Returns
        -------
        list[Entity]
            Parsed and slugified Entity objects.
        """
        entities: List[Entity] = []
        for item in raw_entities:
            label = (item.get("label") or "").strip()
            if not label:
                continue
            entities.append(
                Entity(
                    id=_slugify(label),
                    label=label,
                    type=(item.get("type") or "Unknown").strip(),
                    source_paper_id=paper_id,
                )
            )
        return entities

    def _parse_relations(
        self, raw_relations: list, paper_id: str
    ) -> List[Relation]:
        """
        Convert raw relation dicts from the LLM JSON into Relation objects.

        Entries missing ``source``, ``target``, or ``relation`` are skipped.

        Parameters
        ----------
        raw_relations : list
            List of ``{"source": ..., "target": ..., "relation": ...}`` dicts.
        paper_id : str
            ID of the source paper (attached to every Relation).

        Returns
        -------
        list[Relation]
            Parsed Relation objects with slugified source/target IDs.
        """
        relations: List[Relation] = []
        for item in raw_relations:
            source = (item.get("source") or "").strip()
            target = (item.get("target") or "").strip()
            relation_type = (item.get("relation") or "").strip()
            if not (source and target and relation_type):
                continue
            relations.append(
                Relation(
                    source_id=_slugify(source),
                    target_id=_slugify(target),
                    relation_type=relation_type,
                    source_paper_id=paper_id,
                )
            )
        return relations

    def _parse_json(self, raw_text: str | None) -> dict | None:
        """Parse model output into JSON dictionary, returning None on failure."""
        if not raw_text:
            return None
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def _heuristic_extract_from_abstract(
        self,
        abstract: str,
        paper_id: str,
        question: str = "",
        plan: Dict | None = None,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Deterministic, domain-agnostic fallback extraction from the abstract text."""
        plan = plan or {}
        reasoning_type = str(plan.get("type", "definition"))

        def classify_type(phrase: str) -> str:
            text = phrase.lower()
            if any(tok in text for tok in ["method", "technique", "process", "approach", "protocol", "procedure", "strategy", "framework", "system", "technology", "tool"]):
                return "Method / Intervention"
            if any(tok in text for tok in ["effect", "outcome", "result", "impact", "improvement", "reduction", "yield", "performance", "efficiency", "accuracy", "stability", "selectivity", "quality"]):
                return "Effect / Outcome"
            if any(tok in text for tok in ["cause", "factor", "driver", "trigger", "mechanism", "constraint", "input", "stressor"]):
                return "Cause / Factor"
            if any(tok in text for tok in ["context", "environment", "condition", "setting", "platform", "region", "domain", "ecosystem"]):
                return "Context / Location"
            if reasoning_type in {"method_process", "method/process", "optimization", "optimization/recommendation"}:
                return "Method / Intervention"
            return "Concept / Entity"

        def classify_relation(sentence: str) -> str:
            text = sentence.lower()
            if any(tok in text for tok in ["leads to", "results in", "causes", "produces", "yields"]):
                return "leads_to"
            if any(tok in text for tok in ["reduces", "improves", "increases", "decreases", "enhances", "affects", "changes"]):
                return "affects"
            if any(tok in text for tok in ["using", "via", "through", "with", "by"]):
                return "depends_on"
            if any(tok in text for tok in ["in ", "within", "under", "during", "in the"]):
                return "occurs_in"
            return "correlates_with"

        def slug(text: str) -> str:
            text = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
            text = re.sub(r"\s+", " ", text).strip()
            return re.sub(r"\s", "_", text)

        sentence_candidates = re.split(r"(?<=[.!?])\s+", abstract)
        phrases: List[str] = []
        for sentence in sentence_candidates:
            if len(sentence.strip()) < 20:
                continue
            # Grab multi-word phrases that look like entities without relying on domain keywords.
            chunks = re.findall(r"\b[A-Za-z][A-Za-z0-9\-/]{2,}(?:\s+[A-Za-z][A-Za-z0-9\-/]{2,}){0,4}\b", sentence)
            for chunk in chunks:
                cleaned = chunk.strip().strip(",;:()[]{}").strip()
                if len(cleaned) < 3:
                    continue
                phrases.append(cleaned)

        # Deduplicate while preserving order.
        seen = set()
        entities: List[Entity] = []
        for phrase in phrases:
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    id=slug(phrase),
                    label=phrase,
                    type=classify_type(phrase),
                    source_paper_id=paper_id,
                )
            )
            if len(entities) >= 10:
                break

        relations: List[Relation] = []
        if len(entities) >= 2:
            for sentence in sentence_candidates:
                sentence_entities = [e for e in entities if e.label.lower() in sentence.lower()]
                if len(sentence_entities) < 2:
                    continue
                relation_type = classify_relation(sentence)
                for source, target in zip(sentence_entities[:-1], sentence_entities[1:]):
                    relations.append(
                        Relation(
                            source_id=source.id,
                            target_id=target.id,
                            relation_type=relation_type,
                            source_paper_id=paper_id,
                        )
                    )

        logger.info(
            "Heuristic extraction for paper '%s': %d entities, %d relations.",
            paper_id,
            len(entities),
            len(relations),
        )
        return entities, relations

    def _call_llm_with_retries(
        self,
        prompt: str,
        max_tokens: int,
        call_kind: str,
        paper_id: str,
    ) -> str | None:
        """Call LLM with exponential backoff and jitter, returning None on final failure."""
        if self._use_ollama_primary and self._can_use_ollama():
            ollama_text = self._call_ollama(
                prompt=prompt,
                max_tokens=max_tokens,
                paper_id=paper_id,
                call_kind=call_kind,
            )
            if ollama_text:
                return ollama_text
            logger.warning(
                "Ollama primary failed for paper '%s' (%s); falling back to Groq.",
                paper_id,
                call_kind,
            )

        if self._groq_rate_limited and self._can_use_ollama():
            ollama_text = self._call_ollama(
                prompt=prompt,
                max_tokens=max_tokens,
                paper_id=paper_id,
                call_kind=call_kind,
            )
            if ollama_text:
                return ollama_text

        try:
            client = self._get_client()
        except EnvironmentError as e:
            if self._can_use_ollama():
                logger.info(
                    "Groq unavailable for paper '%s' (%s): %s. Using Ollama fallback model '%s'.",
                    paper_id,
                    call_kind,
                    e,
                    self._ollama_model,
                )
                return self._call_ollama(prompt=prompt, max_tokens=max_tokens, paper_id=paper_id, call_kind=call_kind)
            logger.error("Groq unavailable and Ollama fallback disabled for paper '%s': %s", paper_id, e)
            return None

        active_model = self.model
        for attempt in range(_MAX_EXTRACTION_RETRIES):
            try:
                self.llm_calls += 1
                response = client.chat.completions.create(
                    model=active_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e).lower()
                if "rate_limit_exceeded" in msg or "tokens per day" in msg or "429" in msg:
                    self._groq_rate_limited = True
                    if self._can_use_ollama():
                        logger.warning(
                            "Groq rate limit for paper '%s' (%s); enabling extractor circuit breaker and trying Ollama fallback model '%s'.",
                            paper_id,
                            call_kind,
                            self._ollama_model,
                        )
                        ollama_text = self._call_ollama(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            paper_id=paper_id,
                            call_kind=call_kind,
                        )
                        if ollama_text:
                            return ollama_text
                    logger.warning(
                        "Groq rate limit for paper '%s' (%s); switching to heuristic extraction.",
                        paper_id,
                        call_kind,
                    )
                    return None
                if "model_decommissioned" in msg or "no longer supported" in msg:
                    if active_model != self._fallback_model:
                        logger.warning(
                            "Extractor model '%s' unavailable; retrying with fallback model '%s'.",
                            active_model,
                            self._fallback_model,
                        )
                        active_model = self._fallback_model
                        continue
                if attempt >= _MAX_EXTRACTION_RETRIES - 1:
                    logger.error(
                        "Groq API error (%s) for paper '%s': %s",
                        call_kind,
                        paper_id,
                        e,
                    )
                    return None

                wait_seconds = (
                    _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    + random.uniform(0.0, _BACKOFF_JITTER_MAX_SECONDS)
                )
                with self._stats_lock:
                    self._retry_count += 1
                logger.warning(
                    "Groq API error (%s) for paper '%s' on attempt %d/%d; retrying in %.2fs",
                    call_kind,
                    paper_id,
                    attempt + 1,
                    _MAX_EXTRACTION_RETRIES,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        return None

    def _can_use_ollama(self) -> bool:
        """Whether local Ollama fallback is enabled by configuration."""
        return bool((self._use_ollama_primary or self._ollama_fallback_enabled) and self._ollama_model)

    def _call_ollama(
        self,
        prompt: str,
        max_tokens: int,
        paper_id: str,
        call_kind: str,
    ) -> str | None:
        """Call local Ollama /api/generate and return generated text or None."""
        try:
            self.llm_calls += 1
            resp = requests.post(
                f"{self._ollama_base_url}/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self._ollama_timeout,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            text = (payload.get("response") or "").strip()
            if text:
                return text
            logger.warning(
                "Ollama returned empty response for paper '%s' (%s).",
                paper_id,
                call_kind,
            )
            return None
        except Exception as e:
            logger.warning(
                "Ollama fallback failed for paper '%s' (%s): %s",
                paper_id,
                call_kind,
                e,
            )
            return None

    def _get_client(self) -> Groq:
        """
        Return the cached Groq client, creating it on first call.

        Returns
        -------
        Groq
            Authenticated Groq API client.

        Raises
        ------
        EnvironmentError
            If ``GROQ_API_KEY`` is not set in the environment.
        """
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file."
                )
            self._client = Groq(api_key=api_key)
        return self._client

    @staticmethod
    def precheck_model_health(model: str) -> tuple[bool, str]:
        """Run a minimal request to verify provider model availability."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False, "GROQ_API_KEY is not set."

        try:
            client = Groq(api_key=api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "health check"}],
                temperature=0.0,
                max_tokens=1,
            )
            return True, "ok"
        except Exception as e:
            return False, str(e)


# ---------------------------------------------------------------------------
# Full pipeline smoke-test: DomainDetector -> Retriever -> Filter -> Extractor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    from pipeline.domain_detector import DomainDetector
    from pipeline.paper_retriever import PaperRetriever
    from pipeline.paper_filter import PaperFilter

    QUESTION = "What fungal diseases affect wheat crops in humid environments?"
    TOP_K = 3  # keep small to limit LLM calls during testing

    print()
    print("=" * 70)
    print(f"  Question : {QUESTION}")
    print("=" * 70)

    # Stage 1 — domain detection
    domain = DomainDetector().classify(QUESTION, method="keyword")
    print(f"\n  [1] Domain detected : {domain}")

    # Stage 2 — paper retrieval
    retriever = PaperRetriever(top_n=20)
    raw_papers = retriever.retrieve(QUESTION, domain)
    print(f"  [2] Papers retrieved: {len(raw_papers)}")

    # Stage 3 — filter + rank
    filtered = PaperFilter().filter(raw_papers, QUESTION, top_k=TOP_K)
    print(f"  [3] Papers after filter: {len(filtered)}")

    # Stage 4 — entity extraction
    extractor = EntityExtractor()
    all_entities, all_relations = extractor.extract(filtered)

    # Print results grouped by paper
    print()
    print("=" * 70)
    print("  EXTRACTED ENTITIES & RELATIONS (grouped by paper)")
    print("=" * 70)

    for paper in filtered:
        paper_entities = [e for e in all_entities if e.source_paper_id == paper.paper_id]
        paper_relations = [r for r in all_relations if r.source_paper_id == paper.paper_id]

        print(f"\n  Paper : {paper.title[:65]}")
        print(f"  ID    : {paper.paper_id[:50]}")
        print(f"  Score : {paper.relevance_score:.4f}")

        if paper_entities:
            print(f"\n  Entities ({len(paper_entities)}):")
            for e in paper_entities:
                print(f"    - [{e.type}]  {e.label}  (id: {e.id})")
        else:
            print("  Entities: none extracted")

        if paper_relations:
            print(f"\n  Relations ({len(paper_relations)}):")
            for r in paper_relations:
                print(f"    - {r.source_id}  --[{r.relation_type}]-->  {r.target_id}")
        else:
            print("  Relations: none extracted")

        print()

    print("=" * 70)
    print(f"  TOTAL: {len(all_entities)} unique entities, {len(all_relations)} relations")
    print("=" * 70)
