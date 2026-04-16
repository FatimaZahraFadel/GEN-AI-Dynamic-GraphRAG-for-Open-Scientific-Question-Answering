"""
Stage 0 — Query Planner: understand the query before retrieving.

This module classifies the query into a general reasoning type, extracts
key entities and intent signals, and produces a structured plan that drives
every downstream stage (retrieval, extraction, graph building, answer
generation).

The planner is fully domain-agnostic — it uses LLM classification with
heuristic fallback and never relies on hardcoded domain keyword lists.

Reasoning types
---------------
- definition       — "What is X?"
- comparison       — "X vs Y", "differences between"
- method_process   — "How to", techniques, procedures
- causal_mechanism — "Why", causes, effects, mechanisms
- optimization     — recommendations, best practices, improvements
"""

import os
import re
import json
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq
import requests

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_GENERAL_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_TEMPERATURE,
    USE_OLLAMA_PRIMARY,
)

from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reasoning type vocabulary — domain-agnostic
# ---------------------------------------------------------------------------

REASONING_TYPES = [
    "definition",
    "comparison",
    "method_process",
    "causal_mechanism",
    "optimization",
]

# Entity type priorities per reasoning type — guides extraction focus
_EXTRACTION_PRIORITIES: Dict[str, List[str]] = {
    "definition": [
        "Concept / Entity",
        "Context / Location",
        "Effect / Outcome",
    ],
    "comparison": [
        "Effect / Outcome",
        "Method / Intervention",
        "Concept / Entity",
    ],
    "method_process": [
        "Method / Intervention",
        "Effect / Outcome",
        "Context / Location",
    ],
    "causal_mechanism": [
        "Cause / Factor",
        "Effect / Outcome",
        "Problem / Condition",
    ],
    "optimization": [
        "Method / Intervention",
        "Effect / Outcome",
        "Cause / Factor",
    ],
}

# Relation type priorities per reasoning type
_RELATION_PRIORITIES: Dict[str, List[str]] = {
    "definition": ["part_of", "studied_in", "correlates_with"],
    "comparison": ["affects", "correlates_with", "leads_to"],
    "method_process": ["mitigates", "detected_by", "depends_on"],
    "causal_mechanism": ["caused_by", "leads_to", "affects"],
    "optimization": ["mitigates", "leads_to", "affects"],
}

# Answer structure guidance per reasoning type
_ANSWER_STRUCTURES: Dict[str, str] = {
    "definition": "concise_explanation",
    "comparison": "table_or_contrast",
    "method_process": "ordered_list",
    "causal_mechanism": "causal_chain",
    "optimization": "ranked_strategies",
}

# Minimum required entity/relation counts per reasoning type (adaptive)
_MIN_GRAPH_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "definition": {"min_nodes": 3, "min_edges": 2, "min_entity_types": 1},
    "comparison": {"min_nodes": 4, "min_edges": 3, "min_entity_types": 2},
    "method_process": {"min_nodes": 3, "min_edges": 3, "min_entity_types": 2},
    "causal_mechanism": {"min_nodes": 5, "min_edges": 4, "min_entity_types": 2},
    "optimization": {"min_nodes": 4, "min_edges": 3, "min_entity_types": 2},
}

# ---------------------------------------------------------------------------
# Heuristic classification signals
# ---------------------------------------------------------------------------

_HEURISTIC_SIGNALS: Dict[str, List[str]] = {
    "definition": [
        "what is", "what are", "define", "definition of", "describe",
        "explain what", "meaning of", "overview of",
    ],
    "comparison": [
        "compare", "comparison", "difference between", "differences between",
        "versus", " vs ", " vs.", "contrast", "better than",
        "advantages and disadvantages", "pros and cons",
    ],
    "method_process": [
        "how to", "how does", "how do", "how can", "process of",
        "steps to", "procedure", "workflow", "implement", "technique",
        "methods for", "approaches to",
    ],
    "causal_mechanism": [
        "why does", "why do", "why is", "why are", "cause of",
        "causes of", "what causes", "mechanism", "due to", "reason for",
        "what leads to", "what triggers", "effect of", "impact of",
    ],
    "optimization": [
        "optimize", "optimization", "improve", "best practices",
        "strategies for", "how to reduce", "how to increase",
        "how to enhance", "recommendations", "mitigate", "remedy",
        "solutions for", "fix", "ways to",
    ],
}


class QueryPlan:
    """Structured output of query understanding — drives all downstream stages."""

    def __init__(
        self,
        reasoning_type: str,
        entities: List[str],
        dimensions: List[str],
        intent_signals: List[str],
        required_entity_types: List[str],
        required_relation_types: List[str],
        answer_structure: str,
        graph_requirements: Dict[str, int],
        complexity: str = "moderate",
        raw_query: str = "",
    ):
        self.reasoning_type = reasoning_type
        self.entities = entities
        self.dimensions = dimensions
        self.intent_signals = intent_signals
        self.required_entity_types = required_entity_types
        self.required_relation_types = required_relation_types
        self.answer_structure = answer_structure
        self.graph_requirements = graph_requirements
        self.complexity = complexity
        self.raw_query = raw_query

    def to_dict(self) -> Dict:
        external_type = {
            "method_process": "method/process",
            "causal_mechanism": "causal/mechanism",
            "optimization": "optimization/recommendation",
        }.get(self.reasoning_type, self.reasoning_type)
        required_nodes = int(self.graph_requirements.get("min_nodes", 5))
        return {
            "type": external_type,
            "entities": self.entities,
            "dimensions": self.dimensions,
            "intent_signals": self.intent_signals,
            "required_entity_types": self.required_entity_types,
            "required_relation_types": self.required_relation_types,
            "answer_structure": self.answer_structure,
            "graph_requirements": self.graph_requirements,
            "required_nodes": required_nodes,
            "complexity": self.complexity,
        }

    def __repr__(self) -> str:
        return (
            f"QueryPlan(type={self.reasoning_type!r}, "
            f"entities={self.entities}, "
            f"dimensions={self.dimensions}, "
            f"answer_structure={self.answer_structure!r})"
        )


class QueryPlanner:
    """
    LLM-driven query understanding with heuristic fallback.

    Produces a QueryPlan that informs every downstream pipeline stage:
    - Retrieval: expanded queries use plan entities + dimensions
    - Extraction: entity/relation type priorities from plan
    - Graph validation: min requirements from plan
    - Answer generation: structure from plan
    """

    def __init__(self, model: str = QUERY_PLANNER_MODEL) -> None:
        self.model = model
        self._client: Optional[Groq] = None
        self._use_ollama_primary = USE_OLLAMA_PRIMARY
        self._ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self._ollama_model = OLLAMA_GENERAL_MODEL
        self._ollama_timeout = OLLAMA_TIMEOUT_SECONDS
        self.llm_calls: int = 0

    def plan(self, query: str) -> QueryPlan:
        """
        Main entry point: understand the query and produce a structured plan.

        Tries LLM-based planning first; falls back to heuristic if LLM fails.

        Parameters
        ----------
        query : str
            Natural-language scientific question.

        Returns
        -------
        QueryPlan
            Structured plan driving all downstream stages.
        """
        # Try LLM-based planning
        plan = self._llm_plan(query)
        if plan is not None:
            logger.info("QueryPlanner: LLM plan produced — %s", plan)
            return plan

        # Fallback to heuristic
        logger.warning("QueryPlanner: LLM planning failed, using heuristic fallback.")
        return self._heuristic_plan(query)

    # ------------------------------------------------------------------
    # LLM-based planning
    # ------------------------------------------------------------------

    def _llm_plan(self, query: str) -> Optional[QueryPlan]:
        """Use LLM to classify and decompose the query."""
        prompt = self._build_planning_prompt(query)
        try:
            raw = self._call_llm(prompt=prompt, max_tokens=400)
            return self._parse_plan_response(raw, query)
        except Exception as e:
            logger.warning("QueryPlanner LLM call failed: %s", e)
            return None

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call Ollama first when enabled, then fall back to Groq."""
        if self._use_ollama_primary:
            try:
                self.llm_calls += 1
                response = requests.post(
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
                response.raise_for_status()
                payload = response.json() or {}
                raw = (payload.get("response") or "").strip()
                if raw:
                    return raw
            except Exception as e:
                logger.warning("QueryPlanner Ollama call failed, falling back to Groq: %s", e)

        client = self._get_client()
        self.llm_calls += 1
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=QUERY_PLANNER_TEMPERATURE,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _build_planning_prompt(self, query: str) -> str:
        return f"""You are a query analysis system. Classify the question and extract a structured plan.

Reasoning types (pick exactly one):
- definition: asking what something is, descriptions, overviews
- comparison: comparing two or more things, differences, trade-offs
- method_process: asking how something works, techniques, procedures, steps
- causal_mechanism: asking why something happens, causes, effects, mechanisms
- optimization: asking how to improve, best practices, recommendations, solutions

Analyze this question and return ONLY valid JSON (no markdown, no explanation):
{{
  "type": "<reasoning_type>",
  "entities": ["<key entity 1>", "<key entity 2>", ...],
  "dimensions": ["<comparison dimension or analysis angle 1>", ...],
  "intent_signals": ["<what the user really wants to know>"],
  "complexity": "simple|moderate|complex"
}}

Rules:
- entities: extract the main scientific concepts, objects, or phenomena mentioned
- dimensions: what aspects should be analyzed (e.g., "effectiveness", "mechanism", "environmental impact")
- intent_signals: 1-2 phrases capturing what the user truly needs answered
- complexity: simple (1 entity, direct answer), moderate (2-3 entities), complex (4+ entities or multi-step reasoning)

Question: {query}"""

    def _parse_plan_response(self, raw: str, query: str) -> Optional[QueryPlan]:
        """Parse LLM JSON response into a QueryPlan."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        reasoning_type = data.get("type", "").strip().lower()
        if reasoning_type not in REASONING_TYPES:
            # Try fuzzy matching
            reasoning_type = self._fuzzy_match_type(reasoning_type)

        entities = [str(e).strip() for e in data.get("entities", []) if str(e).strip()]
        dimensions = [str(d).strip() for d in data.get("dimensions", []) if str(d).strip()]
        intent_signals = [str(s).strip() for s in data.get("intent_signals", []) if str(s).strip()]
        complexity = data.get("complexity", "moderate")

        if not entities:
            entities = self._extract_entities_heuristic(query)

        return self._build_plan(
            reasoning_type=reasoning_type,
            entities=entities,
            dimensions=dimensions,
            intent_signals=intent_signals,
            complexity=complexity,
            query=query,
        )

    def _fuzzy_match_type(self, candidate: str) -> str:
        """Match a potentially mis-spelled reasoning type to the closest valid one."""
        candidate = candidate.lower().replace(" ", "_").replace("/", "_")
        for rt in REASONING_TYPES:
            if rt in candidate or candidate in rt:
                return rt
        # keyword fallback
        if any(k in candidate for k in ["cause", "why", "mechanism", "effect"]):
            return "causal_mechanism"
        if any(k in candidate for k in ["method", "process", "how", "technique"]):
            return "method_process"
        if any(k in candidate for k in ["compar", "versus", "differ"]):
            return "comparison"
        if any(k in candidate for k in ["optim", "improve", "recommend", "best"]):
            return "optimization"
        return "definition"

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_plan(self, query: str) -> QueryPlan:
        """Classify and plan using lightweight heuristics when LLM is unavailable."""
        reasoning_type = self._classify_heuristic(query)
        entities = self._extract_entities_heuristic(query)
        dimensions = self._extract_dimensions_heuristic(query, reasoning_type)
        intent_signals = self._extract_intent_heuristic(query, reasoning_type)
        complexity = self._estimate_complexity(query, entities)

        return self._build_plan(
            reasoning_type=reasoning_type,
            entities=entities,
            dimensions=dimensions,
            intent_signals=intent_signals,
            complexity=complexity,
            query=query,
        )

    def _classify_heuristic(self, query: str) -> str:
        """Classify reasoning type from surface patterns."""
        q = query.lower().strip()
        scores: Dict[str, float] = {rt: 0.0 for rt in REASONING_TYPES}

        # Disambiguation boosts for common ambiguous phrasings like
        # "what are methods for ..." where definition and method/process both match.
        if re.search(r"\b(methods?|techniques?|approaches?|process(es)?|steps?)\s+(for|to|of)\b", q):
            scores["method_process"] += 1.5
        if re.search(r"\b(compare|comparison|difference|versus|\bvs\b)\b", q):
            scores["comparison"] += 1.0

        for rt, signals in _HEURISTIC_SIGNALS.items():
            for signal in signals:
                if signal in q:
                    scores[rt] += 1.0

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0.0:
            # No signal matched — default by question word
            if q.startswith(("what is", "what are", "define")):
                return "definition"
            if q.startswith(("why", "what cause")):
                return "causal_mechanism"
            if q.startswith(("how",)):
                return "method_process"
            return "definition"

        return best

    def _extract_entities_heuristic(self, query: str) -> List[str]:
        """Extract key noun phrases from the query using lightweight heuristics."""
        q = query.strip()
        # Remove question words and common prefixes
        q_clean = re.sub(
            r"^(what|which|how|why|when|where|who|can|does|do|is|are)\s+",
            "", q, flags=re.IGNORECASE,
        )
        q_clean = re.sub(
            r"^(the|a|an)\s+", "", q_clean, flags=re.IGNORECASE,
        )

        # Split on common delimiters
        chunks = re.split(r"\b(?:and|or|versus|vs\.?|compared to|between|of|in|on|for|with|from)\b", q_clean, flags=re.IGNORECASE)
        entities = []
        for chunk in chunks:
            chunk = chunk.strip().strip("?.,;:!")
            # Keep multi-word phrases that look like entities (2-6 words)
            words = chunk.split()
            if 1 <= len(words) <= 6 and len(chunk) >= 3:
                entities.append(chunk)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:6]

    def _extract_dimensions_heuristic(self, query: str, reasoning_type: str) -> List[str]:
        """Infer analysis dimensions from query and reasoning type."""
        dimensions = []
        q = query.lower()

        if reasoning_type == "comparison":
            # Look for explicit comparison dimensions
            for signal in ["in terms of", "regarding", "with respect to", "for"]:
                if signal in q:
                    idx = q.index(signal) + len(signal)
                    tail = q[idx:].strip().strip("?.,")
                    if tail:
                        dimensions.extend(
                            d.strip() for d in re.split(r",|\band\b", tail) if d.strip()
                        )
            if not dimensions:
                dimensions = ["effectiveness", "trade-offs", "applicability"]

        elif reasoning_type == "causal_mechanism":
            dimensions = ["primary causes", "mechanisms", "downstream effects"]

        elif reasoning_type == "method_process":
            dimensions = ["techniques", "implementation", "effectiveness"]

        elif reasoning_type == "optimization":
            dimensions = ["strategies", "trade-offs", "evidence strength"]

        else:  # definition
            dimensions = ["core concept", "key properties", "context"]

        return dimensions[:5]

    def _extract_intent_heuristic(self, query: str, reasoning_type: str) -> List[str]:
        """Produce a 1-line intent summary."""
        intent_map = {
            "definition": "understand what the concept is and its key properties",
            "comparison": "compare alternatives and understand trade-offs",
            "method_process": "learn techniques and how they work",
            "causal_mechanism": "understand why something happens and the mechanism",
            "optimization": "find the best strategies and recommendations",
        }
        return [intent_map.get(reasoning_type, "understand the topic")]

    def _estimate_complexity(self, query: str, entities: List[str]) -> str:
        """Estimate query complexity from entity count and query length."""
        n_entities = len(entities)
        n_words = len(query.split())

        if n_entities <= 1 and n_words < 12:
            return "simple"
        if n_entities >= 4 or n_words > 25:
            return "complex"
        return "moderate"

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        reasoning_type: str,
        entities: List[str],
        dimensions: List[str],
        intent_signals: List[str],
        complexity: str,
        query: str,
    ) -> QueryPlan:
        """Assemble a complete QueryPlan from classified components."""
        required_entity_types = _EXTRACTION_PRIORITIES.get(
            reasoning_type, _EXTRACTION_PRIORITIES["definition"]
        )
        required_relation_types = _RELATION_PRIORITIES.get(
            reasoning_type, _RELATION_PRIORITIES["definition"]
        )
        answer_structure = _ANSWER_STRUCTURES.get(
            reasoning_type, "concise_explanation"
        )
        graph_requirements = dict(_MIN_GRAPH_REQUIREMENTS.get(
            reasoning_type, _MIN_GRAPH_REQUIREMENTS["definition"]
        ))

        # Scale graph requirements by complexity
        if complexity == "complex":
            graph_requirements["min_nodes"] = int(graph_requirements["min_nodes"] * 1.5)
            graph_requirements["min_edges"] = int(graph_requirements["min_edges"] * 1.5)
        elif complexity == "simple":
            graph_requirements["min_nodes"] = max(2, graph_requirements["min_nodes"] - 1)
            graph_requirements["min_edges"] = max(1, graph_requirements["min_edges"] - 1)

        return QueryPlan(
            reasoning_type=reasoning_type,
            entities=entities,
            dimensions=dimensions,
            intent_signals=intent_signals,
            required_entity_types=required_entity_types,
            required_relation_types=required_relation_types,
            answer_structure=answer_structure,
            graph_requirements=graph_requirements,
            complexity=complexity,
            raw_query=query,
        )

    # ------------------------------------------------------------------
    # Query expansion helpers (used by retriever)
    # ------------------------------------------------------------------

    def build_retrieval_queries(self, plan: QueryPlan) -> List[str]:
        """
        Generate expanded retrieval queries from the plan.

        Returns up to 3 query variants:
        1. Entity-focused: combines extracted entities
        2. Dimension-focused: entities + analysis dimensions
        3. Intent-focused: entities + intent signal
        """
        queries = []

        # Entity-focused query
        if plan.entities:
            queries.append(" ".join(plan.entities))

        # Dimension-focused
        if plan.entities and plan.dimensions:
            dim_terms = " ".join(plan.dimensions[:2])
            queries.append(f"{' '.join(plan.entities[:3])} {dim_terms}")

        # Intent-focused
        if plan.entities and plan.intent_signals:
            queries.append(f"{' '.join(plan.entities[:3])} {plan.intent_signals[0]}")

        return queries[:3]

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client(self) -> Groq:
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set.")
            self._client = Groq(api_key=api_key)
        return self._client


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")

    planner = QueryPlanner()

    test_queries = [
        "What is CRISPR and how does it work?",
        "Compare organic fertilizers and chemical fertilizers for wheat production",
        "How do convolutional neural networks detect objects in images?",
        "Why do tectonic plates move and cause earthquakes?",
        "What are the best strategies to reduce soil erosion in tropical regions?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"  Query: {q}")
        plan = planner.plan(q)
        print(f"  Plan:  {plan}")
        print(f"  Dict:  {json.dumps(plan.to_dict(), indent=2)}")
        print(f"  Retrieval queries: {planner.build_retrieval_queries(plan)}")
