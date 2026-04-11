"""
Evaluation module: benchmark Dynamic GraphRAG vs Classic RAG baseline.

Two answer-generation modes are compared on a 12-question benchmark
(3 questions per domain: Agriculture, Geoscience, Environment, Supply Chain).

Classic RAG  — top-3 paper abstracts concatenated as plain text context.
Dynamic GraphRAG — full pipeline (EntityExtractor -> GraphBuilder ->
                   GraphRetriever -> AnswerGenerator).

Metrics computed per answer:
  - rouge1_f1       : unigram F1 overlap with reference answer
  - rouge2_f1       : bigram  F1 overlap with reference answer
  - keyword_coverage: fraction of reference keywords found in prediction
  - answer_length   : word count (proxy for completeness)
"""

import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)

ABLATION_MODES = [
    "classic_rag",
    "graphrag_base",
    "graphrag_reasoning_only",
    "graphrag_full",
]

MODES = {
    "classic_rag": {},
    "graphrag_base": {
        "enable_reasoning": False,
        "enable_confidence": False,
        "enable_expansion": False,
    },
    "graphrag_reasoning_only": {
        "enable_reasoning": True,
        "enable_confidence": False,
        "enable_expansion": False,
    },
    "graphrag_full": {
        "enable_reasoning": True,
        "enable_confidence": True,
        "enable_expansion": True,
    },
}

# ---------------------------------------------------------------------------
# Benchmark dataset
# ---------------------------------------------------------------------------

BENCHMARK: List[Dict] = [
    # --- Agriculture ---
    {
        "domain": "Agriculture",
        "question": "What fungal diseases affect wheat crops in humid environments?",
        "reference": (
            "Wheat crops in humid environments are susceptible to several fungal diseases "
            "including rust (stem rust, leaf rust, stripe rust caused by Puccinia species), "
            "powdery mildew (Blumeria graminis), Fusarium head blight (scab), and Septoria "
            "leaf blotch. High humidity and warm temperatures promote spore germination and "
            "spread. These diseases can cause significant yield losses and reduce grain quality."
        ),
    },
    {
        "domain": "Agriculture",
        "question": "How does drought stress affect crop yield?",
        "reference": (
            "Drought stress reduces crop yield by limiting water availability for photosynthesis, "
            "nutrient uptake, and cell expansion. It triggers stomatal closure, reducing CO2 "
            "assimilation and increasing leaf temperature. At critical growth stages such as "
            "flowering and grain filling, drought causes significant yield penalties through "
            "reduced grain number, smaller grain size, and early senescence."
        ),
    },
    {
        "domain": "Agriculture",
        "question": "What are the effects of soil degradation on agricultural productivity?",
        "reference": (
            "Soil degradation reduces agricultural productivity by decreasing organic matter, "
            "nutrient availability, and water retention capacity. Erosion removes the fertile "
            "topsoil layer, compaction reduces root penetration and aeration, and salinization "
            "impairs water uptake by plants. Degraded soils support fewer beneficial microorganisms "
            "and produce lower crop yields with higher input requirements."
        ),
    },
    # --- Geoscience ---
    {
        "domain": "Geoscience",
        "question": "How do tectonic plate movements trigger earthquakes?",
        "reference": (
            "Tectonic plates move due to convection currents in the mantle. At plate boundaries, "
            "stress accumulates as plates converge, diverge, or slide past one another. When "
            "accumulated elastic strain exceeds the frictional strength of faults, rupture occurs "
            "and stored energy is released as seismic waves. Subduction zones and transform faults "
            "are particularly prone to large earthquakes."
        ),
    },
    {
        "domain": "Geoscience",
        "question": "What causes volcanic eruptions near subduction zones?",
        "reference": (
            "At subduction zones, an oceanic plate descends beneath a continental or oceanic plate. "
            "As the subducting slab sinks deeper, increasing pressure and temperature cause "
            "dehydration of water-bearing minerals. The released water lowers the melting point "
            "of the overlying mantle wedge, generating magma. This magma rises through the crust "
            "and erupts at the surface forming volcanic arcs."
        ),
    },
    {
        "domain": "Geoscience",
        "question": "How does erosion affect sediment transport in river systems?",
        "reference": (
            "Erosion detaches soil and rock particles from hillslopes and channel banks, "
            "supplying sediment to river systems. Rivers transport this material as bedload, "
            "suspended load, and dissolved load depending on particle size and flow velocity. "
            "Higher erosion rates increase sediment supply, affecting channel morphology, "
            "delta formation, and downstream deposition. Vegetation loss and land use change "
            "significantly accelerate erosion and sediment loads."
        ),
    },
    # --- Environment ---
    {
        "domain": "Environment",
        "question": "What are the main causes of deforestation in tropical regions?",
        "reference": (
            "The main causes of tropical deforestation include agricultural expansion "
            "(cattle ranching and soy production), commercial logging, infrastructure development "
            "(roads and dams), mining, and urban growth. Smallholder subsistence farming also "
            "contributes. Weak governance and lack of land tenure security enable illegal clearing. "
            "Economic incentives for land conversion often outweigh conservation value in "
            "short-term decision making."
        ),
    },
    {
        "domain": "Environment",
        "question": "How do carbon emissions contribute to climate change?",
        "reference": (
            "Carbon dioxide and other greenhouse gases released by burning fossil fuels, "
            "deforestation, and industrial processes trap outgoing infrared radiation in the "
            "atmosphere — the greenhouse effect. This raises global average temperatures, "
            "alters precipitation patterns, and increases the frequency of extreme weather events. "
            "Elevated CO2 also contributes to ocean acidification by dissolving into seawater."
        ),
    },
    {
        "domain": "Environment",
        "question": "What is the impact of biodiversity loss on ecosystems?",
        "reference": (
            "Biodiversity loss weakens ecosystem functioning by reducing species that perform "
            "key roles such as pollination, decomposition, nutrient cycling, and pest regulation. "
            "Fewer species means reduced redundancy and resilience to disturbance. Loss of "
            "predator–prey relationships can trigger trophic cascades. Ecosystem services "
            "including clean water, food production, and climate regulation are diminished "
            "as biodiversity declines."
        ),
    },
    # --- Supply Chain ---
    {
        "domain": "Supply Chain",
        "question": "How can inventory forecasting improve warehouse efficiency?",
        "reference": (
            "Accurate inventory forecasting reduces overstock and stockout situations by aligning "
            "stock levels with expected demand. This minimises storage costs, frees up warehouse "
            "space, and reduces waste from perishable goods. Improved forecasting also enables "
            "better labour planning, faster order fulfilment, and reduced emergency procurement. "
            "Demand-driven replenishment models improve turnover ratios and working capital."
        ),
    },
    {
        "domain": "Supply Chain",
        "question": "What strategies reduce supplier disruption risk?",
        "reference": (
            "Strategies to reduce supplier disruption risk include multi-sourcing (qualifying "
            "multiple suppliers for critical components), holding safety stock, nearshoring or "
            "regionalising supply networks, and developing supplier relationships through "
            "long-term contracts. Real-time supply chain visibility tools, risk scoring of "
            "suppliers, and business continuity planning also help organisations detect and "
            "respond quickly to disruptions."
        ),
    },
    {
        "domain": "Supply Chain",
        "question": "How does procurement optimization reduce logistics costs?",
        "reference": (
            "Procurement optimisation reduces logistics costs by consolidating orders to achieve "
            "volume discounts and reduce shipment frequency, selecting suppliers closer to "
            "manufacturing sites to lower transport distances, and negotiating better freight "
            "terms. Standardising packaging, improving demand forecasting to avoid expedited "
            "shipping, and using reverse auctions to increase supplier competition further drive "
            "down total landed costs."
        ),
    },
]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase and split text into word tokens, stripping punctuation."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n-grams from a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(prediction: str, reference: str, n: int) -> float:
    """
    Compute ROUGE-N F1 between a prediction and a reference string.

    Parameters
    ----------
    prediction : str
        Generated answer text.
    reference : str
        Reference answer text.
    n : int
        N-gram size (1 for ROUGE-1, 2 for ROUGE-2).

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    pred_tokens = _tokenise(prediction)
    ref_tokens = _tokenise(reference)

    pred_ng = _ngrams(pred_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)

    overlap = sum((pred_ng & ref_ng).values())
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, reference: str) -> float:
    """
    Compute the fraction of non-stopword reference tokens present in the prediction.

    Parameters
    ----------
    prediction : str
        Generated answer text.
    reference : str
        Reference answer text.

    Returns
    -------
    float
        Coverage ratio in [0, 1].
    """
    _STOP = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "it", "its",
        "that", "this", "as", "not", "no", "so", "how", "what", "which",
        "can", "will", "would", "could", "should", "may", "might",
    }
    ref_keywords = {t for t in _tokenise(reference) if t not in _STOP}
    pred_tokens = set(_tokenise(prediction))
    if not ref_keywords:
        return 0.0
    return len(ref_keywords & pred_tokens) / len(ref_keywords)


# ---------------------------------------------------------------------------
# Main Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Benchmarks Dynamic GraphRAG against a Classic RAG baseline on a fixed
    12-question dataset spanning Agriculture, Geoscience, Environment, and
    Supply Chain domains.

    Classic RAG
    -----------
    Retrieve papers -> Filter -> concatenate top-3 abstracts as plain text
    context -> call Groq LLM directly (no graph).

    Dynamic GraphRAG
    ----------------
    Full pipeline: DomainDetector -> PaperRetriever -> PaperFilter ->
    EntityExtractor -> GraphBuilder -> GraphRetriever -> AnswerGenerator.

    Metrics per answer
    ------------------
    - ``rouge1_f1``       : ROUGE-1 F1 vs reference answer
    - ``rouge2_f1``       : ROUGE-2 F1 vs reference answer
    - ``keyword_coverage``: fraction of reference keywords in prediction
    - ``answer_length``   : word count (completeness proxy)

    Attributes
    ----------
    model : str
        Groq model used for both Classic RAG and Dynamic GraphRAG generation.
    temperature : float
        LLM sampling temperature.
    _client : Groq or None
        Lazy-initialised Groq client.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        judge_repeats: int = 2,
    ) -> None:
        """
        Initialise the evaluator.

        Parameters
        ----------
        model : str
            Groq model for answer generation. Defaults to
            ``llama-3.1-8b-instant``.
        temperature : float
            LLM sampling temperature. Defaults to 0.2.
        """
        self.model = model
        self.temperature = temperature
        self._client: Optional[Groq] = None
        self.judge_repeats = max(1, judge_repeats)

    # ------------------------------------------------------------------
    # Classic RAG baseline
    # ------------------------------------------------------------------

    def run_classic_rag(self, question: str, domain: str) -> Dict[str, str]:
        """
        Generate an answer using Classic RAG (plain abstract concatenation).

        Retrieves papers with :class:`PaperRetriever`, filters with
        :class:`PaperFilter`, concatenates the top-3 abstracts, and calls
        the Groq LLM directly — no entity extraction or graph construction.

        Parameters
        ----------
        question : str
            User question.
        domain : str
            Detected domain (passed to retriever for query scoping).

        Returns
        -------
        dict
            ``{"answer": ..., "context": ...}`` where ``context`` is the
            retrieval context used to generate the answer.
        """
        from pipeline.paper_retriever import PaperRetriever
        from pipeline.paper_filter import PaperFilter

        papers = PaperRetriever(top_n=20).retrieve(question, domain)
        filtered = PaperFilter().filter(papers, question, top_k=5)

        # Build plain-text context from top-3 abstracts
        context_parts: List[str] = []
        for i, p in enumerate(filtered[:3], 1):
            snippet = (p.abstract or "")[:400].strip()
            context_parts.append(f"[{i}] {p.title}\n{snippet}")
        context = "\n\n".join(context_parts) if context_parts else "No context available."

        prompt = (
            f"You are a scientific question-answering assistant.\n"
            f"Answer the question using ONLY the provided paper abstracts.\n"
            f"Be concise and factual. Cite papers as [Paper: title] where relevant.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT (paper abstracts):\n{context}\n\n"
            f"ANSWER:"
        )
        answer = self._call_llm(prompt)
        return {
            "answer": answer,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Dynamic GraphRAG
    # ------------------------------------------------------------------

    def run_graphrag(self, question: str, domain: str) -> Dict[str, str]:
        """
        Generate an answer using the full Dynamic GraphRAG pipeline.

        Runs all seven stages: DomainDetector -> PaperRetriever ->
        PaperFilter -> EntityExtractor -> GraphBuilder ->
        GraphRetriever -> AnswerGenerator.

        Parameters
        ----------
        question : str
            User question.
        domain : str
            Pre-detected domain (skips DomainDetector to save an LLM call).

        Returns
        -------
        dict
            ``{"answer": ..., "context": ...}`` where ``context`` is the
            retrieval context used to generate the answer.
        """
        from main import run_pipeline
        from utils.session_state import SessionState

        details = run_pipeline(
            query=question,
            fast_mode=True,
            session_state=SessionState(),
            return_details=True,
        )
        return {
            "answer": details["answer_dict"]["answer"],
            "context": details.get("retrieval", {}).get("context_text", ""),
        }

    def run_graphrag_mode(self, question: str, mode: str) -> Dict:
        """Run GraphRAG with feature toggles for ablation experiments."""
        run_pipeline = self._import_run_pipeline()
        from utils.session_state import SessionState

        if mode == "graphrag_base":
            flags = {
                "enable_reasoning_controller": False,
                "enable_confidence_retrieval": False,
                "enable_iterative_expansion": False,
            }
        elif mode == "graphrag_reasoning_only":
            flags = {
                "enable_reasoning_controller": True,
                "enable_confidence_retrieval": False,
                "enable_iterative_expansion": False,
            }
        elif mode == "graphrag_full":
            flags = {
                "enable_reasoning_controller": True,
                "enable_confidence_retrieval": True,
                "enable_iterative_expansion": True,
            }
        else:
            raise ValueError(f"Unsupported GraphRAG mode: {mode}")

        return run_pipeline(
            query=question,
            fast_mode=True,
            session_state=SessionState(),
            return_details=True,
            **flags,
        )

    # ------------------------------------------------------------------
    # Metrics — LLM-as-judge
    # ------------------------------------------------------------------

    def score_answer(self, answer: str, question: str, context: str) -> Dict[str, int]:
        """
        Score an answer on three criteria using an LLM judge (1–5 scale).

        The judge evaluates:

        - **groundedness** — every claim is traceable to a source or graph
          triple; no invented facts.
        - **reasoning_depth** — the answer connects multiple concepts and
          explains relationships.
        - **hallucination_resistance** — the answer avoids stating facts not
          present in the provided context (5 = zero hallucination).

        Parameters
        ----------
        answer : str
            The generated answer to evaluate.
        question : str
            The original user question the answer addresses.
        context : str
            Retrieval context used to generate the answer.

        Returns
        -------
        dict
            Keys: ``groundedness``, ``reasoning_depth``,
            ``hallucination_resistance`` — each an integer 1–5.
        """
        judge_prompt = (
            f"You are an evaluation judge. Score the following answer "
            f"to the given question on four criteria, each from 1 to 5:\n"
            f"- groundedness: every claim is traceable to a source or "
            f"graph triple, no invented facts\n"
            f"- reasoning_depth: the answer connects multiple concepts "
            f"and explains relationships\n"
            f"- hallucination_resistance: the answer avoids stating facts "
            f"not present in the provided context (5 = no hallucination)\n\n"
            f"- reasoning_trace_quality: clarity of stepwise reasoning, "
            f"logical consistency, and use of multi-hop evidence\n\n"
            f"Return ONLY a JSON object:\n"
            f"{{\n"
            f'  "groundedness": <int>,\n'
            f'  "reasoning_depth": <int>,\n'
            f'  "hallucination_resistance": <int>,\n'
            f'  "reasoning_trace_quality": <int>\n'
            f"}}\n"
            f"No explanation, no markdown, only the JSON object.\n\n"
            f"Question: {question}\n\n"
            f"Retrieval Context: {context}\n\n"
            f"Answer: {answer}"
        )

        try:
            raw = self._call_llm(judge_prompt)
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            scores = json.loads(cleaned)
            return {
                "groundedness":           int(scores.get("groundedness", 1)),
                "reasoning_depth":        int(scores.get("reasoning_depth", 1)),
                "hallucination_resistance": int(scores.get("hallucination_resistance", 1)),
                "reasoning_trace_quality": int(scores.get("reasoning_trace_quality", 1)),
                # aliases used by research-ablation interface
                "reasoning": int(scores.get("reasoning", scores.get("reasoning_depth", 1))),
                "hallucination": int(scores.get("hallucination", scores.get("hallucination_resistance", 1))),
                "valid": 1,
            }
        except Exception as e:
            logger.error(f"LLM judge scoring failed: {e}")
            return {
                "groundedness": 1,
                "reasoning_depth": 1,
                "hallucination_resistance": 1,
                "reasoning_trace_quality": 1,
                "reasoning": 1,
                "hallucination": 1,
                "valid": 0,
            }

    def score_answer_stable(self, answer: str, question: str, context: str) -> Dict[str, int]:
        """Run judge multiple times and average scores for stability."""
        score_runs = [self.score_answer(answer, question, context) for _ in range(self.judge_repeats)]
        keys = ["groundedness", "reasoning", "hallucination", "reasoning_trace_quality"]
        averaged = {
            k: int(round(sum(float(s.get(k, 1)) for s in score_runs) / max(len(score_runs), 1)))
            for k in keys
        }
        averaged["reasoning_depth"] = averaged["reasoning"]
        averaged["hallucination_resistance"] = averaged["hallucination"]
        averaged["valid"] = int(any(int(s.get("valid", 0)) == 1 for s in score_runs))
        return averaged

    def _normalize_mode_output(self, out: Dict) -> Dict:
        """Ensure each mode output contains required fields with safe defaults."""
        return {
            "answer": out.get("answer", ""),
            "context": out.get("context", out.get("retrieval", {}).get("context_text", "")),
            "scores": out.get("scores", {}),
            "metrics": {
                "runtime": float(out.get("metrics", {}).get("runtime", 0.0)),
                "llm_calls": int(out.get("metrics", {}).get("llm_calls", 0)),
                "embedding_calls": int(out.get("metrics", {}).get("embedding_calls", 0)),
                "retrieved_papers": int(out.get("metrics", {}).get("retrieved_papers", 0)),
                "graph_expansion_iters": int(out.get("metrics", {}).get("graph_expansion_iters", 0)),
            },
            "graph_metrics": {
                "num_nodes": int(out.get("graph_metrics", {}).get("num_nodes", 0)),
                "num_edges": int(out.get("graph_metrics", {}).get("num_edges", 0)),
                "density": float(out.get("graph_metrics", {}).get("density", 0.0)),
                "avg_degree": float(out.get("graph_metrics", {}).get("avg_degree", 0.0)),
            },
            "confidence": float(out.get("confidence", 0.0)),
        }

    def compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation via numpy.corrcoef."""
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return 0.0
        try:
            value = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(value):
                return 0.0
            return value
        except Exception:
            return 0.0

    def run_ablation(self, questions: List[str]) -> Dict:
        """Run requested ablation modes and return research-structured outputs."""
        run_pipeline = self._import_run_pipeline()
        from pipeline.embedding_service import EmbeddingService
        from utils.session_state import SessionState

        benchmark_map = {b["question"]: b for b in BENCHMARK}
        rows: List[Dict] = []

        for question in questions:
            meta = benchmark_map.get(question, {"question": question, "domain": "Unknown", "reference": ""})
            domain = meta.get("domain", "Unknown")

            for mode, flags in MODES.items():
                # Reset query counters.
                EmbeddingService.reset_metrics()

                try:
                    if mode == "classic_rag":
                        t0 = time.perf_counter()
                        classic_out = self.run_classic_rag(question, domain)
                        answer = classic_out["answer"]
                        context = classic_out["context"]
                        runtime = time.perf_counter() - t0
                        out = {
                            "answer": answer,
                            "context": context,
                            "metrics": {
                                "runtime": round(runtime, 4),
                                "llm_calls": 1,
                                "embedding_calls": 0,
                                "retrieved_papers": 0,
                                "graph_expansion_iters": 0,
                            },
                            "graph_metrics": {
                                "num_nodes": 0,
                                "num_edges": 0,
                                "density": 0.0,
                                "avg_degree": 0.0,
                            },
                            "confidence": 0.0,
                        }
                    else:
                        out = run_pipeline(
                            query=question,
                            session_state=SessionState(),
                            return_metrics=True,
                            enable_reasoning=flags["enable_reasoning"],
                            enable_confidence=flags["enable_confidence"],
                            enable_expansion=flags["enable_expansion"],
                        )
                except Exception as e:
                    logger.warning("Mode '%s' failed for question '%s': %s", mode, question, e)
                    out = {
                        "answer": f"[ERROR: {e}]",
                        "metrics": {
                            "runtime": 0.0,
                            "llm_calls": 0,
                            "embedding_calls": 0,
                            "retrieved_papers": 0,
                            "graph_expansion_iters": 0,
                        },
                        "graph_metrics": {
                            "num_nodes": 0,
                            "num_edges": 0,
                            "density": 0.0,
                            "avg_degree": 0.0,
                        },
                        "confidence": 0.0,
                    }

                out = self._normalize_mode_output(out)
                scores = self.score_answer_stable(out["answer"], question, out.get("context", ""))
                rows.append(
                    {
                        "mode": mode,
                        "question": question,
                        "domain": domain,
                        "reference": meta.get("reference", ""),
                        "answer": out["answer"],
                        "scores": scores,
                        "metrics": out["metrics"],
                        "graph_metrics": out["graph_metrics"],
                        "confidence": out["confidence"],
                        "valid": bool(scores.get("valid", 0)),
                    }
                )

        def _avg(mode: str, key: str) -> float:
            vals = [float(r["scores"][key]) for r in rows if r["mode"] == mode and r.get("valid", True)]
            return round(sum(vals) / max(len(vals), 1), 3)

        ablation = {
            mode: {
                "groundedness": _avg(mode, "groundedness"),
                "reasoning": _avg(mode, "reasoning"),
                "hallucination": _avg(mode, "hallucination"),
            }
            for mode in MODES.keys()
        }

        efficiency = {}
        for mode in MODES.keys():
            mode_rows = [r for r in rows if r["mode"] == mode and r.get("valid", True)]
            efficiency[mode] = {
                "runtime": round(sum(float(r["metrics"].get("runtime", 0.0)) for r in mode_rows) / max(len(mode_rows), 1), 3),
                "llm_calls": round(sum(float(r["metrics"].get("llm_calls", 0.0)) for r in mode_rows) / max(len(mode_rows), 1), 3),
                "embedding_calls": round(sum(float(r["metrics"].get("embedding_calls", 0.0)) for r in mode_rows) / max(len(mode_rows), 1), 3),
            }

        full_rows = [r for r in rows if r["mode"] == "graphrag_full"]
        conf = [r["confidence"] for r in full_rows]
        grounded = [float(r["scores"].get("groundedness", 0.0)) for r in full_rows]
        reasoning = [float(r["scores"].get("reasoning", 0.0)) for r in full_rows]
        density = [float(r["graph_metrics"].get("density", 0.0)) for r in full_rows]
        avg_degree = [float(r["graph_metrics"].get("avg_degree", 0.0)) for r in full_rows]

        correlations = {
            "confidence_vs_groundedness": round(self.compute_correlation(conf, grounded), 4),
            "confidence_vs_reasoning": round(self.compute_correlation(conf, reasoning), 4),
            "density_vs_reasoning": round(self.compute_correlation(density, reasoning), 4),
            "density_vs_groundedness": round(self.compute_correlation(density, grounded), 4),
            "avg_degree_vs_reasoning": round(self.compute_correlation(avg_degree, reasoning), 4),
            "avg_degree_vs_groundedness": round(self.compute_correlation(avg_degree, grounded), 4),
        }

        best_mode = max(ablation.keys(), key=lambda m: (ablation[m]["groundedness"], ablation[m]["reasoning"]))
        insights = {
            "best_mode": best_mode,
            "tradeoffs": {
                m: {
                    "quality": ablation[m],
                    "efficiency": efficiency[m],
                }
                for m in MODES.keys()
            },
        }

        ranked_full = sorted(
            full_rows,
            key=lambda r: (
                r["scores"].get("groundedness", 0),
                r["scores"].get("reasoning", 0),
                r["scores"].get("reasoning_trace_quality", 0),
            ),
            reverse=True,
        )
        examples = {
            "highest_scoring_example": ranked_full[0] if ranked_full else {},
            "lowest_scoring_example": ranked_full[-1] if ranked_full else {},
        }

        return {
            "ablation": ablation,
            "efficiency": efficiency,
            "correlations": correlations,
            "insights": insights,
            "examples": examples,
            "rows": rows,
        }

    def generate_report(self, results: Dict) -> Dict:
        """Return structured summary tables and findings from ablation results."""
        ablation_table = [
            {
                "Mode": mode,
                "Groundedness": vals["groundedness"],
                "Reasoning": vals["reasoning"],
                "Hallucination": vals["hallucination"],
            }
            for mode, vals in results.get("ablation", {}).items()
        ]

        efficiency_table = [
            {
                "Mode": mode,
                "Runtime": vals.get("runtime", 0.0),
                "LLM Calls": vals.get("llm_calls", 0.0),
                "Embedding Calls": vals.get("embedding_calls", 0.0),
            }
            for mode, vals in results.get("efficiency", {}).items()
        ]

        correlations = results.get("correlations", {})
        insights = results.get("insights", {})
        examples = results.get("examples", {})

        return {
            # map-style outputs requested by final presentation script
            "ablation": results.get("ablation", {}),
            "efficiency": results.get("efficiency", {}),
            "correlations": correlations,
            "insights": insights,
            "examples": examples,
            # tabular mirrors for downstream rendering
            "ablation_table": ablation_table,
            "efficiency_table": efficiency_table,
        }

    def run_research_evaluation(self) -> Dict:
        """One-call runner: load default questions, run ablation, build report."""
        questions = [b["question"] for b in BENCHMARK]
        results = self.run_ablation(questions)
        report = self.generate_report(results)

        # Sanity checks
        warnings_list: List[str] = []
        ab = report.get("ablation", {})
        if "graphrag_full" in ab:
            full = ab["graphrag_full"]
            for mode in ["classic_rag", "graphrag_base", "graphrag_reasoning_only"]:
                if mode in ab and full["groundedness"] < ab[mode]["groundedness"]:
                    warnings_list.append(f"Warning: graphrag_full groundedness < {mode}")
        corr = report.get("correlations", {})
        if corr.get("confidence_vs_groundedness", 0.0) < 0:
            warnings_list.append("Warning: confidence_vs_groundedness is negative")

        report["sanity_warnings"] = warnings_list
        return report

    # ------------------------------------------------------------------
    # Research-grade ablation evaluation
    # ------------------------------------------------------------------

    def run_ablation_study(
        self,
        questions: Optional[List[Dict]] = None,
        modes: Optional[List[str]] = None,
        delay: float = 2.0,
    ) -> List[Dict]:
        """Run ablation across classic and GraphRAG variants with efficiency logs."""
        if questions is None:
            questions = BENCHMARK
        if modes is None:
            modes = ABLATION_MODES

        rows: List[Dict] = []

        for i, item in enumerate(questions, 1):
            question = item["question"]
            domain = item["domain"]

            print(f"\n[{i:02d}/{len(questions)}] {domain} :: {question}")
            for mode in modes:
                print(f"  Running mode={mode}...", end=" ", flush=True)
                try:
                    t0 = time.perf_counter()
                    if mode == "classic_rag":
                        classic_out = self.run_classic_rag(question, domain)
                        answer = classic_out["answer"]
                        context = classic_out["context"]
                        runtime_seconds = time.perf_counter() - t0
                        metrics = {
                            "runtime_seconds": round(runtime_seconds, 4),
                            "llm_calls": 1,
                            "embedding_calls": 0,
                            "retrieved_papers": 0,
                            "graph_expansion_iterations": 0,
                        }
                        graph_confidence = 0.0
                        graph_metrics = {
                            "num_nodes": 0,
                            "num_edges": 0,
                            "density": 0.0,
                            "avg_degree": 0.0,
                        }
                    else:
                        details = self.run_graphrag_mode(question, mode)
                        answer = details["answer_dict"]["answer"]
                        context = details.get("retrieval", {}).get("context_text", "")
                        metrics = details.get("runtime_metrics", {})
                        graph_confidence = float(details.get("graph_confidence", 0.0))
                        graph_metrics = details.get("graph_metrics", {})

                    scores = self.score_answer(answer, question, context)
                    rows.append(
                        {
                            "mode": mode,
                            "domain": domain,
                            "question": question,
                            "reference": item.get("reference", ""),
                            "answer": answer,
                            "scores": scores,
                            "efficiency": metrics,
                            "graph_confidence": graph_confidence,
                            "graph_metrics": graph_metrics,
                        }
                    )
                    print("done.")
                except Exception as e:
                    logger.exception("Mode '%s' failed for question '%s': %s", mode, question, e)
                    rows.append(
                        {
                            "mode": mode,
                            "domain": domain,
                            "question": question,
                            "reference": item.get("reference", ""),
                            "answer": f"[ERROR: {e}]",
                            "scores": {
                                "groundedness": 1,
                                "reasoning_depth": 1,
                                "hallucination_resistance": 1,
                                "reasoning_trace_quality": 1,
                            },
                            "efficiency": {
                                "runtime_seconds": 0.0,
                                "llm_calls": 0,
                                "embedding_calls": 0,
                                "retrieved_papers": 0,
                                "graph_expansion_iterations": 0,
                            },
                            "graph_confidence": 0.0,
                            "graph_metrics": {
                                "num_nodes": 0,
                                "num_edges": 0,
                                "density": 0.0,
                                "avg_degree": 0.0,
                            },
                        }
                    )
                    print(f"ERROR: {e}")
                time.sleep(delay)

        return rows

    def _pearson(self, xs: List[float], ys: List[float]) -> float:
        """Compute Pearson correlation safely."""
        if len(xs) < 2 or len(xs) != len(ys):
            return 0.0
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
        den = den_x * den_y
        if den == 0.0:
            return 0.0
        return round(num / den, 4)

    def compute_research_stats(self, rows: List[Dict]) -> Dict:
        """Aggregate quality, efficiency, and correlation stats for report output."""
        score_metrics = [
            "groundedness",
            "reasoning_depth",
            "hallucination_resistance",
            "reasoning_trace_quality",
        ]
        eff_metrics = [
            "runtime_seconds",
            "llm_calls",
            "embedding_calls",
            "retrieved_papers",
            "graph_expansion_iterations",
        ]

        by_mode: Dict[str, Dict[str, float]] = {}
        for mode in sorted({r["mode"] for r in rows}):
            mode_rows = [r for r in rows if r["mode"] == mode]
            mode_stats: Dict[str, float] = {}
            for m in score_metrics:
                mode_stats[m] = round(sum(r["scores"][m] for r in mode_rows) / max(len(mode_rows), 1), 3)
            for m in eff_metrics:
                mode_stats[m] = round(
                    sum(float(r.get("efficiency", {}).get(m, 0.0)) for r in mode_rows) / max(len(mode_rows), 1),
                    3,
                )
            by_mode[mode] = mode_stats

        full_rows = [r for r in rows if r["mode"] == "graphrag_full"]
        conf = [float(r.get("graph_confidence", 0.0)) for r in full_rows]
        grounded = [float(r["scores"].get("groundedness", 0.0)) for r in full_rows]
        reasoning = [float(r["scores"].get("reasoning_depth", 0.0)) for r in full_rows]

        density = [float(r.get("graph_metrics", {}).get("density", 0.0)) for r in full_rows]
        avg_degree = [float(r.get("graph_metrics", {}).get("avg_degree", 0.0)) for r in full_rows]

        correlations = {
            "confidence_vs_groundedness": self._pearson(conf, grounded),
            "confidence_vs_reasoning": self._pearson(conf, reasoning),
            "density_vs_reasoning": self._pearson(density, reasoning),
            "density_vs_groundedness": self._pearson(density, grounded),
            "avg_degree_vs_reasoning": self._pearson(avg_degree, reasoning),
            "avg_degree_vs_groundedness": self._pearson(avg_degree, grounded),
        }

        return {
            "by_mode": by_mode,
            "correlations": correlations,
        }

    def print_research_report(self, rows: List[Dict]) -> None:
        """Print research-grade summary tables, findings, and case studies."""
        stats = self.compute_research_stats(rows)
        by_mode = stats["by_mode"]
        corr = stats["correlations"]

        score_metrics = [
            "groundedness",
            "reasoning_depth",
            "hallucination_resistance",
            "reasoning_trace_quality",
        ]
        eff_metrics = [
            "runtime_seconds",
            "llm_calls",
            "embedding_calls",
            "retrieved_papers",
            "graph_expansion_iterations",
        ]

        sep = "=" * 92
        print(f"\n{sep}")
        print("RESEARCH EVALUATION — Dynamic GraphRAG Ablation")
        print(sep)

        print("\nQuality Table (avg 1–5):")
        header = f"{'Mode':<24}" + "".join(f"{m[:18]:>18}" for m in score_metrics)
        print(header)
        print("-" * len(header))
        for mode in ABLATION_MODES:
            if mode not in by_mode:
                continue
            row = f"{mode:<24}" + "".join(f"{by_mode[mode][m]:>18.3f}" for m in score_metrics)
            print(row)

        print("\nEfficiency Table (lower is better for runtime/calls):")
        header = f"{'Mode':<24}" + "".join(f"{m[:18]:>18}" for m in eff_metrics)
        print(header)
        print("-" * len(header))
        for mode in ABLATION_MODES:
            if mode not in by_mode:
                continue
            row = f"{mode:<24}" + "".join(f"{by_mode[mode][m]:>18.3f}" for m in eff_metrics)
            print(row)

        print("\nConfidence / Graph correlations:")
        for k, v in corr.items():
            print(f"  {k:<36}: {v:+.4f}")

        # Key findings: best component gain vs base.
        if "graphrag_base" in by_mode and "graphrag_full" in by_mode and "graphrag_reasoning_only" in by_mode:
            base = by_mode["graphrag_base"]
            ro = by_mode["graphrag_reasoning_only"]
            full = by_mode["graphrag_full"]
            reasoning_gain = ro["reasoning_depth"] - base["reasoning_depth"]
            full_grounding_gain = full["groundedness"] - base["groundedness"]
            runtime_tradeoff = full["runtime_seconds"] - base["runtime_seconds"]

            print("\nKey findings:")
            print(f"  - Reasoning controller gain (reasoning depth): {reasoning_gain:+.3f}")
            print(f"  - Full system gain (groundedness vs base): {full_grounding_gain:+.3f}")
            print(f"  - Runtime trade-off (full - base, seconds): {runtime_tradeoff:+.3f}")

        # Case studies
        question_modes: Dict[str, Dict[str, Dict]] = {}
        for r in rows:
            question_modes.setdefault(r["question"], {})[r["mode"]] = r

        best_q = None
        best_gain = -999.0
        worst_q = None
        worst_gain = 999.0
        for q, mode_map in question_modes.items():
            if "classic_rag" not in mode_map or "graphrag_full" not in mode_map:
                continue
            gain = (
                mode_map["graphrag_full"]["scores"]["groundedness"]
                - mode_map["classic_rag"]["scores"]["groundedness"]
            )
            if gain > best_gain:
                best_gain = gain
                best_q = q
            if gain < worst_gain:
                worst_gain = gain
                worst_q = q

        print("\nCase studies:")
        if best_q is not None:
            print(f"  - Success case: '{best_q}' (groundedness gain: {best_gain:+.2f})")
        if worst_q is not None:
            print(f"  - Failure case: '{worst_q}' (groundedness gain: {worst_gain:+.2f})")

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        questions: Optional[List[Dict]] = None,
        delay: float = 3.0,
    ) -> List[Dict]:
        """
        Run both Classic RAG and Dynamic GraphRAG on all benchmark questions
        and compute scores for each.

        Parameters
        ----------
        questions : list[dict] or None
            Questions to evaluate. Defaults to the built-in 12-question
            ``BENCHMARK`` dataset.
        delay : float
            Seconds to sleep between API calls to avoid rate-limiting.
            Defaults to 3.0.

        Returns
        -------
        list[dict]
            One result dict per question containing:
            ``domain``, ``question``, ``reference``,
            ``classic_answer``, ``graphrag_answer``,
            ``classic_scores``, ``graphrag_scores``.
        """
        if questions is None:
            questions = BENCHMARK

        results: List[Dict] = []

        for i, item in enumerate(questions, 1):
            q = item["question"]
            domain = item["domain"]
            ref = item["reference"]

            print(f"\n[{i:02d}/{len(questions)}] Domain: {domain}")
            print(f"  Q: {q}")

            # Classic RAG
            print("  Running Classic RAG...", end=" ", flush=True)
            try:
                classic_out = self.run_classic_rag(q, domain)
                classic_answer = classic_out["answer"]
                classic_context = classic_out["context"]
                print("done.")
            except Exception as e:
                classic_answer = f"[ERROR: {e}]"
                classic_context = ""
                print(f"ERROR: {e}")
            time.sleep(delay)

            # Dynamic GraphRAG
            print("  Running Dynamic GraphRAG...", end=" ", flush=True)
            try:
                graphrag_out = self.run_graphrag(q, domain)
                graphrag_answer = graphrag_out["answer"]
                graphrag_context = graphrag_out["context"]
                print("done.")
            except Exception as e:
                graphrag_answer = f"[ERROR: {e}]"
                graphrag_context = ""
                print(f"ERROR: {e}")
            time.sleep(delay)

            print("  Scoring with LLM judge...", end=" ", flush=True)
            classic_scores = self.score_answer(classic_answer, q, classic_context)
            graphrag_scores = self.score_answer(graphrag_answer, q, graphrag_context)
            print("done.")

            results.append({
                "domain":          domain,
                "question":        q,
                "reference":       ref,
                "classic_answer":  classic_answer,
                "graphrag_answer": graphrag_answer,
                "classic_scores":  classic_scores,
                "graphrag_scores": graphrag_scores,
            })

            print(
                f"  Classic  → G:{classic_scores['groundedness']}  "
                f"R:{classic_scores['reasoning_depth']}  "
                f"H:{classic_scores['hallucination_resistance']}"
            )
            print(
                f"  GraphRAG → G:{graphrag_scores['groundedness']}  "
                f"R:{graphrag_scores['reasoning_depth']}  "
                f"H:{graphrag_scores['hallucination_resistance']}"
            )

        return results

    def aggregate(self, results: List[Dict]) -> Dict:
        """
        Aggregate per-question LLM-judge scores into means per mode and domain.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_benchmark`.

        Returns
        -------
        dict
            Keys ``"overall"`` and each domain name, each containing
            ``{"classic": {...}, "graphrag": {...}}`` averaged score dicts.
        """
        metrics = ["groundedness", "reasoning_depth", "hallucination_resistance"]

        def _mean_scores(rows: List[Dict], key: str) -> Dict[str, float]:
            agg: Dict[str, float] = {m: 0.0 for m in metrics}
            for r in rows:
                for m in metrics:
                    agg[m] += r[key].get(m, 0)
            n = max(len(rows), 1)
            return {m: round(agg[m] / n, 2) for m in metrics}

        agg: Dict = {
            "overall": {
                "classic":  _mean_scores(results, "classic_scores"),
                "graphrag": _mean_scores(results, "graphrag_scores"),
            }
        }

        domains = sorted({r["domain"] for r in results})
        for domain in domains:
            domain_rows = [r for r in results if r["domain"] == domain]
            agg[domain] = {
                "classic":  _mean_scores(domain_rows, "classic_scores"),
                "graphrag": _mean_scores(domain_rows, "graphrag_scores"),
            }

        return agg

    def print_report(self, results: List[Dict]) -> None:
        """
        Print a formatted benchmark report with LLM-judge scores.

        Shows per-question answers and scores for both modes, followed by
        an aggregated comparison table broken down by domain with a WINNER
        row for each metric.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_benchmark`.
        """
        sep = "=" * 72
        metrics = ["groundedness", "reasoning_depth", "hallucination_resistance"]

        print(f"\n{sep}")
        print("  BENCHMARK REPORT — Dynamic GraphRAG vs Classic RAG")
        print(sep)

        for i, r in enumerate(results, 1):
            print(f"\n[{i:02d}] [{r['domain']}] {r['question']}")
            print()
            print("  -- Classic RAG Answer --")
            for line in r["classic_answer"].split("\n"):
                print(f"  {line}")
            cs = r["classic_scores"]
            print(
                f"  Scores → Groundedness: {cs['groundedness']}  "
                f"Reasoning: {cs['reasoning_depth']}  "
                f"Anti-hallucination: {cs['hallucination_resistance']}"
            )
            print()
            print("  -- Dynamic GraphRAG Answer --")
            for line in r["graphrag_answer"].split("\n"):
                print(f"  {line}")
            gs = r["graphrag_scores"]
            print(
                f"  Scores → Groundedness: {gs['groundedness']}  "
                f"Reasoning: {gs['reasoning_depth']}  "
                f"Anti-hallucination: {gs['hallucination_resistance']}"
            )
            print()
            print("-" * 72)

        # Aggregated comparison table
        agg = self.aggregate(results)
        col_w = 12

        print(f"\n{sep}")
        print("  AGGREGATED SCORES  (LLM-as-Judge, scale 1-5)")
        print(sep)

        header = (
            f"  {'Scope':<18} {'Mode':<10}"
            + "".join(f"  {m[:col_w]:>{col_w}}" for m in metrics)
        )
        print(header)
        print("  " + "-" * (28 + len(metrics) * (col_w + 2)))

        for scope in ["overall"] + sorted(k for k in agg if k != "overall"):
            c = agg[scope]["classic"]
            g = agg[scope]["graphrag"]
            c_row = f"  {scope:<18} {'classic':<10}" + "".join(f"  {c[m]:>{col_w}.2f}" for m in metrics)
            g_row = f"  {'':<18} {'graphrag':<10}" + "".join(f"  {g[m]:>{col_w}.2f}" for m in metrics)
            winner_cells = "".join(
                f"  {'GraphRAG' if g[m] > c[m] else 'Classic ' if c[m] > g[m] else 'Tie     ':>{col_w}}"
                for m in metrics
            )
            w_row = f"  {'':<18} {'WINNER':<10}{winner_cells}"
            print(c_row)
            print(g_row)
            print(w_row)
            print("  " + "-" * (28 + len(metrics) * (col_w + 2)))

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Call the Groq chat-completion API and return the answer text.

        Parameters
        ----------
        prompt : str
            Fully assembled prompt.

        Returns
        -------
        str
            Stripped answer text from the LLM.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _import_run_pipeline(self):
        """Import ``run_pipeline`` robustly whether evaluator is run as module or script."""
        try:
            from main import run_pipeline
            return run_pipeline
        except ModuleNotFoundError:
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from main import run_pipeline
            return run_pipeline

    def _get_client(self) -> Groq:
        """
        Return the cached Groq client, creating it on first call.

        Returns
        -------
        Groq
            Authenticated client.

        Raises
        ------
        EnvironmentError
            If ``GROQ_API_KEY`` is not set.
        """
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set in .env")
            self._client = Groq(api_key=api_key)
        return self._client


# ---------------------------------------------------------------------------
# Entry point — run full benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    print("\nDynamic GraphRAG — Final Research Evaluation")
    print(f"Questions : {len(BENCHMARK)} ({len(BENCHMARK)//4} per domain)")
    print("Modes     : classic_rag, graphrag_base, graphrag_reasoning_only, graphrag_full")
    print("Metrics   : groundedness, reasoning, hallucination, reasoning_trace_quality")

    evaluator = Evaluator()
    report = evaluator.run_research_evaluation()

    print("\n=== ABLATION RESULTS ===")
    for mode, scores in report["ablation"].items():
        print(mode, scores)

    print("\n=== EFFICIENCY ===")
    for mode, metrics in report["efficiency"].items():
        print(mode, metrics)

    print("\n=== CORRELATIONS ===")
    for k, v in report["correlations"].items():
        print(k, v)

    print("\n=== INSIGHTS ===")
    print(report["insights"])

    if report.get("sanity_warnings"):
        print("\n=== SANITY WARNINGS ===")
        for w in report["sanity_warnings"]:
            print(w)

    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"\nSaved: {out_path}")
