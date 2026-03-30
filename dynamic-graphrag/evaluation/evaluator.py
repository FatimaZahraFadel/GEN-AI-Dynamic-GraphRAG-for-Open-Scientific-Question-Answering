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

import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

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

    # ------------------------------------------------------------------
    # Classic RAG baseline
    # ------------------------------------------------------------------

    def run_classic_rag(self, question: str, domain: str) -> str:
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
        str
            Generated answer text.
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
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Dynamic GraphRAG
    # ------------------------------------------------------------------

    def run_graphrag(self, question: str, domain: str) -> str:
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
        str
            Generated answer text.
        """
        from pipeline.paper_retriever import PaperRetriever
        from pipeline.paper_filter import PaperFilter
        from pipeline.entity_extractor import EntityExtractor
        from pipeline.graph_builder import GraphBuilder
        from pipeline.graph_retriever import GraphRetriever
        from pipeline.answer_generator import AnswerGenerator

        papers = PaperRetriever(top_n=20).retrieve(question, domain)
        filtered = PaperFilter().filter(papers, question, top_k=5)
        entities, relations = EntityExtractor(model=self.model).extract(filtered)
        graph = GraphBuilder().build(entities, relations)
        retrieval = GraphRetriever().retrieve(question, graph)
        result = AnswerGenerator(model=self.model, temperature=self.temperature).generate(
            question=question,
            context_text=retrieval["context_text"],
            papers=filtered,
        )
        return result["answer"]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute all metrics for a single prediction/reference pair.

        Parameters
        ----------
        prediction : str
            Generated answer.
        reference : str
            Ground-truth reference answer.

        Returns
        -------
        dict
            Keys: ``rouge1_f1``, ``rouge2_f1``, ``keyword_coverage``,
            ``answer_length``.
        """
        return {
            "rouge1_f1":        round(rouge_n_f1(prediction, reference, 1), 4),
            "rouge2_f1":        round(rouge_n_f1(prediction, reference, 2), 4),
            "keyword_coverage": round(keyword_coverage(prediction, reference), 4),
            "answer_length":    len(_tokenise(prediction)),
        }

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
                classic_answer = self.run_classic_rag(q, domain)
                print("done.")
            except Exception as e:
                classic_answer = f"[ERROR: {e}]"
                print(f"ERROR: {e}")
            time.sleep(delay)

            # Dynamic GraphRAG
            print("  Running Dynamic GraphRAG...", end=" ", flush=True)
            try:
                graphrag_answer = self.run_graphrag(q, domain)
                print("done.")
            except Exception as e:
                graphrag_answer = f"[ERROR: {e}]"
                print(f"ERROR: {e}")
            time.sleep(delay)

            classic_scores = self.score(classic_answer, ref)
            graphrag_scores = self.score(graphrag_answer, ref)

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
                f"  Classic  → ROUGE-1: {classic_scores['rouge1_f1']:.3f}  "
                f"KW-cov: {classic_scores['keyword_coverage']:.3f}"
            )
            print(
                f"  GraphRAG → ROUGE-1: {graphrag_scores['rouge1_f1']:.3f}  "
                f"KW-cov: {graphrag_scores['keyword_coverage']:.3f}"
            )

        return results

    def aggregate(self, results: List[Dict]) -> Dict:
        """
        Aggregate per-question scores into mean values per mode and domain.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_benchmark`.

        Returns
        -------
        dict
            Keys ``"overall"`` and each domain name, each containing
            ``{"classic": {...}, "graphrag": {...}}`` averaged metric dicts.
        """
        metrics = ["rouge1_f1", "rouge2_f1", "keyword_coverage", "answer_length"]

        def _mean_scores(rows: List[Dict], key: str) -> Dict[str, float]:
            agg: Dict[str, float] = {m: 0.0 for m in metrics}
            for r in rows:
                for m in metrics:
                    agg[m] += r[key][m]
            n = max(len(rows), 1)
            return {m: round(agg[m] / n, 4) for m in metrics}

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
        Print a formatted benchmark report to stdout.

        Shows per-question answers, scores for both modes, and an aggregated
        comparison table broken down by domain.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`run_benchmark`.
        """
        sep = "=" * 72

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
                f"  Scores → ROUGE-1: {cs['rouge1_f1']:.3f}  "
                f"ROUGE-2: {cs['rouge2_f1']:.3f}  "
                f"KW-cov: {cs['keyword_coverage']:.3f}  "
                f"Len: {cs['answer_length']} words"
            )
            print()
            print("  -- Dynamic GraphRAG Answer --")
            for line in r["graphrag_answer"].split("\n"):
                print(f"  {line}")
            gs = r["graphrag_scores"]
            print(
                f"  Scores → ROUGE-1: {gs['rouge1_f1']:.3f}  "
                f"ROUGE-2: {gs['rouge2_f1']:.3f}  "
                f"KW-cov: {gs['keyword_coverage']:.3f}  "
                f"Len: {gs['answer_length']} words"
            )
            print()
            print("-" * 72)

        # Aggregated comparison table
        agg = self.aggregate(results)
        print(f"\n{sep}")
        print("  AGGREGATED SCORES")
        print(sep)
        header = f"  {'Scope':<18} {'Mode':<10} {'ROUGE-1':>8} {'ROUGE-2':>8} {'KW-cov':>8} {'Length':>7}"
        print(header)
        print("  " + "-" * 68)

        def _row(scope: str, mode: str, scores: Dict) -> str:
            return (
                f"  {scope:<18} {mode:<10} "
                f"{scores['rouge1_f1']:>8.3f} "
                f"{scores['rouge2_f1']:>8.3f} "
                f"{scores['keyword_coverage']:>8.3f} "
                f"{scores['answer_length']:>7}"
            )

        for scope in ["overall"] + sorted(k for k in agg if k != "overall"):
            print(_row(scope, "classic",  agg[scope]["classic"]))
            print(_row(scope, "graphrag", agg[scope]["graphrag"]))
            print("  " + "-" * 68)

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
    import io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    print("\nDynamic GraphRAG — Benchmark Evaluation")
    print(f"Questions : {len(BENCHMARK)} ({len(BENCHMARK)//4} per domain)")
    print(f"Domains   : Agriculture, Geoscience, Environment, Supply Chain")
    print(f"Metrics   : ROUGE-1, ROUGE-2, Keyword Coverage, Answer Length")
    print()

    evaluator = Evaluator()
    results = evaluator.run_benchmark(delay=3.0)
    evaluator.print_report(results)
