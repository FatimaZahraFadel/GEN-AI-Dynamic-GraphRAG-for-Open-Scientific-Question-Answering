"""
Stage 3 — Paper Filter: clean, date-filter, and rank retrieved papers.

Pipeline applied inside filter():
    1. filter_by_abstract       — drop papers without a usable abstract
    2. filter_by_date           — drop papers older than min_year
    3. rank_by_relevance        — embed query + papers, sort by cosine similarity
    4. adaptive_filter          — keep top percentile; remove low-similarity outliers
    5. truncate                 — return top_k results

Upgrades
--------
- Step 3: Embedding-based relevance scoring (already present; now exposed via
  ``relevance_score`` on each Paper, used downstream).
- Step 4: Adaptive percentile filtering — keeps the top ``keep_percentile`` of
  papers by similarity score rather than a fixed keyword threshold.
"""

import re
import math
from typing import List, Set

import numpy as np

from config.settings import EMBEDDING_MODEL, TOP_N_PAPERS
from models.paper import Paper
from pipeline.embedding_service import EmbeddingService
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_ABSTRACT_LENGTH = 50
_ADAPTIVE_KEEP_PERCENTILE = 60   # Step 4: keep top 60 % by similarity score


class PaperFilter:
    """
    Filters and ranks a raw list of Paper objects so that only the most
    relevant, recent, and content-rich papers reach the graph-building stage.

    The embedding model is loaded lazily on the first call to
    :meth:`rank_by_relevance` and cached for the lifetime of the instance.

    Attributes
    ----------
    embedding_model_name : str
        Name of the sentence-transformers model used for semantic ranking.
    _embedding_service : EmbeddingService
        Shared embedding service instance.
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """
        Initialise the filter.

        Parameters
        ----------
        embedding_model_name : str
            Sentence-transformers model name. Defaults to
            ``config.settings.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
        """
        self.embedding_model_name = embedding_model_name
        self._embedding_service = (
            embedding_service or EmbeddingService.get_instance(embedding_model_name)
        )
        self.last_intent_relevant_count: int = 0

    _INTENT_CUES = {
        "solution": {
            "method", "approach", "improve", "reduce", "mitigate", "optimize",
            "strategy", "intervention", "fix", "remedy", "technique", "framework",
        },
        "cause": {
            "cause", "causes", "caused", "driver", "determinant", "results",
            "leads", "because", "due", "mechanism", "factor", "trigger",
        },
        "process": {
            "process", "workflow", "pipeline", "steps", "procedure", "mechanism",
            "how", "sequence", "stages", "implementation", "execution",
        },
        "fact": {
            "is", "are", "includes", "consists", "types", "list", "overview",
        },
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_focus_terms(self, question: str) -> Set[str]:
        """Extract informative query terms for lexical coverage scoring."""
        tokens = set(re.findall(r"\b[a-z]{4,}\b", (question or "").lower()))
        stop = {
            "what", "which", "when", "where", "does", "that", "this",
            "from", "with", "into", "about", "affect", "affects",
            "impact", "impacts", "environments", "environment",
        }
        return {t for t in tokens if t not in stop}

    def _keyword_coverage_score(self, question: str, text: str) -> float:
        """Score how well paper text covers question terms (domain-agnostic)."""
        q_terms = self._extract_focus_terms(question)
        if not q_terms:
            return 0.0

        text_l = (text or "").lower()
        matches = sum(1 for t in q_terms if t in text_l)
        return matches / max(len(q_terms), 1)

    def _metadata_prior_score(self, paper: Paper) -> float:
        """Compute a lightweight credibility prior from citations and recency."""
        citation_score = min(math.log1p(max(paper.citation_count, 0)) / 6.0, 1.0)
        recency_score = 0.0
        if paper.year is not None:
            recency_score = max(0.0, min((paper.year - 1990) / 40.0, 1.0))
        return 0.7 * citation_score + 0.3 * recency_score

    def filter(
        self,
        papers: List[Paper],
        question: str,
        intent: str = "fact",
        top_k: int = 30,
    ) -> List[Paper]:
        """
        Main entry point — apply the full filtering and ranking pipeline.

        Steps (in order):
            1. :meth:`filter_by_abstract`  — remove papers with no/short abstract
            2. :meth:`filter_by_date`      — remove papers older than min_year
            3. :meth:`rank_by_relevance`   — rank by semantic similarity to query
            4. :meth:`adaptive_filter`     — remove low-similarity outliers
            5. Truncate to ``top_k``

        Parameters
        ----------
        papers : list[Paper]
            Raw papers from the retrieval stage.
        question : str
            Original user question, used as the ranking reference.
        top_k : int
            Maximum number of papers to return. Defaults to
            ``config.settings.TOP_N_PAPERS``.

        Returns
        -------
        list[Paper]
            Filtered and ranked papers, at most ``top_k`` entries.
        """
        logger.info(f"Starting filter pipeline on {len(papers)} papers.")

        papers = self.filter_by_abstract(papers)
        papers = self.filter_by_date(papers)
        papers = self.rank_by_relevance(papers, question, intent=intent)
        papers = self.adaptive_filter(papers)
        papers = papers[:top_k]

        self.last_intent_relevant_count = sum(
            1 for p in papers if float(getattr(p, "_intent_score", 0.0)) >= 0.25
        )
        logger.info(
            "Intent-aware filtering: intent='%s', intent_relevant_papers=%d/%d",
            intent,
            self.last_intent_relevant_count,
            len(papers),
        )

        logger.info(f"Filter pipeline complete. Kept {len(papers)} papers.")
        return papers

    def assess_evidence_consistency(self, question: str, papers: List[Paper]) -> dict:
        """
        Compute a domain-agnostic evidence consistency score for filtered papers.

        Returns a dictionary with:
        - consistency_score: float in [0, 1]
        - is_consistent: bool
        - avg_pairwise_overlap: lexical agreement across papers
        - citation_signal: credibility signal from citation distribution
        """
        if len(papers) < 3:
            return {
                "consistency_score": 0.0,
                "is_consistent": False,
                "avg_pairwise_overlap": 0.0,
                "citation_signal": 0.0,
                "reason": "insufficient_papers",
            }

        focus_terms = self._extract_focus_terms(question)
        if not focus_terms:
            focus_terms = set(re.findall(r"\b[a-z]{4,}\b", (question or "").lower()))

        text_sets: List[Set[str]] = []
        for p in papers:
            txt = f"{p.title}. {p.abstract or ''}".lower()
            toks = set(re.findall(r"\b[a-z][a-z\-]{3,}\b", txt))
            if focus_terms:
                toks = {t for t in toks if (t in focus_terms) or any(ft in t or t in ft for ft in focus_terms)}
            text_sets.append(toks)

        overlaps: List[float] = []
        for i in range(len(text_sets)):
            for j in range(i + 1, len(text_sets)):
                a, b = text_sets[i], text_sets[j]
                if not a and not b:
                    continue
                inter = len(a & b)
                union = max(len(a | b), 1)
                overlaps.append(inter / union)
        avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0

        citation_vals = np.array([max(int(p.citation_count or 0), 0) for p in papers], dtype=float)
        citation_signal = 0.0
        if len(citation_vals) > 0:
            citation_signal = min(float(np.log1p(np.median(citation_vals)) / 6.0), 1.0)

        score = (0.75 * avg_overlap) + (0.25 * citation_signal)
        is_consistent = score >= 0.22

        return {
            "consistency_score": round(float(score), 4),
            "is_consistent": bool(is_consistent),
            "avg_pairwise_overlap": round(avg_overlap, 4),
            "citation_signal": round(citation_signal, 4),
            "papers_considered": len(papers),
        }

    def adaptive_filter(
        self,
        papers: List[Paper],
        keep_percentile: float = _ADAPTIVE_KEEP_PERCENTILE,
    ) -> List[Paper]:
        """
        Step 4 — Adaptive percentile filtering (no hardcoded keywords).

        Computes the distribution of ``relevance_score`` values across all
        papers and removes any paper whose score falls below the
        ``keep_percentile``-th percentile threshold.

        This is purely embedding-driven — no domain keywords are used.

        Parameters
        ----------
        papers : list[Paper]
            Papers ranked by :meth:`rank_by_relevance` (must have
            ``relevance_score`` populated).
        keep_percentile : float
            Papers below this percentile by similarity score are dropped.
            Defaults to 60 (keep top 60 %).

        Returns
        -------
        list[Paper]
            Papers at or above the adaptive threshold, order preserved.
        """
        if len(papers) < 4:
            return papers

        scores = np.array([p.relevance_score for p in papers], dtype=float)
        threshold = float(np.percentile(scores, 100.0 - keep_percentile))
        kept = [p for p in papers if p.relevance_score >= threshold]

        removed = len(papers) - len(kept)
        logger.info(
            "adaptive_filter: threshold=%.4f (p%d) — removed %d outlier paper(s), "
            "%d remaining. Score range kept: [%.4f, %.4f]",
            threshold,
            int(100 - keep_percentile),
            removed,
            len(kept),
            float(scores.min()) if len(scores) else 0.0,
            float(scores.max()) if len(scores) else 0.0,
        )
        return kept

    def filter_by_abstract(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove papers whose abstract is missing or too short to be useful.

        A paper is kept only when its abstract is a non-empty string with at
        least ``_MIN_ABSTRACT_LENGTH`` (50) characters after stripping
        whitespace.

        Parameters
        ----------
        papers : list[Paper]
            Input paper list.

        Returns
        -------
        list[Paper]
            Papers that have a sufficiently long abstract.
        """
        before = len(papers)
        kept = [
            p for p in papers
            if p.abstract and len(p.abstract.strip()) >= _MIN_ABSTRACT_LENGTH
        ]
        removed = before - len(kept)
        if removed:
            logger.info(
                f"filter_by_abstract: removed {removed} paper(s) "
                f"(no/short abstract). {len(kept)} remaining."
            )
        else:
            logger.info(
                f"filter_by_abstract: all {len(kept)} papers passed."
            )
        return kept

    def filter_by_date(
        self, papers: List[Paper], min_year: int = 2000
    ) -> List[Paper]:
        """
        Remove papers published before *min_year*.

        Papers with no year information are kept to avoid discarding
        potentially relevant work with missing metadata.

        Parameters
        ----------
        papers : list[Paper]
            Input paper list.
        min_year : int
            Earliest acceptable publication year. Defaults to 2000.

        Returns
        -------
        list[Paper]
            Papers published in or after ``min_year`` (plus those with no year).
        """
        before = len(papers)
        kept = [
            p for p in papers
            if p.year is None or p.year >= min_year
        ]
        removed = before - len(kept)
        if removed:
            logger.info(
                f"filter_by_date (min_year={min_year}): removed {removed} "
                f"paper(s). {len(kept)} remaining."
            )
        else:
            logger.info(
                f"filter_by_date (min_year={min_year}): all {len(kept)} papers passed."
            )
        return kept

    def _extract_key_phrases(self, text: str) -> Set[str]:
        """Extract lightweight unigram/bigram phrases from text."""
        tokens = re.findall(r"\b[a-z][a-z\-]{3,}\b", (text or "").lower())
        phrases: Set[str] = set(tokens)
        for i in range(len(tokens) - 1):
            phrases.add(f"{tokens[i]} {tokens[i + 1]}")
        return phrases

    def _intent_signal_score(self, text: str, intent: str) -> float:
        """Score generic intent relevance using linguistic cue patterns."""
        cues = self._INTENT_CUES.get(intent, set())
        if not cues:
            return 0.0

        phrases = self._extract_key_phrases(text)
        matches = sum(1 for cue in cues if cue in phrases or cue in text.lower())
        return min(matches / max(len(cues) * 0.3, 1.0), 1.0)

    def rank_by_relevance(self, papers: List[Paper], question: str, intent: str = "fact") -> List[Paper]:
        """
        Rank papers by semantic similarity between the question and each paper.

        Each paper is represented as the concatenation of its title and
        abstract.  Both the question and all paper texts are encoded with the
        ``all-MiniLM-L6-v2`` sentence-transformer model.  Cosine similarity
        between the question embedding and each paper embedding is stored in
        ``paper.relevance_score``, and the list is sorted descending by score.

        The model is lazy-loaded and cached on first call.

        Parameters
        ----------
        papers : list[Paper]
            Papers to rank (must have passed abstract filtering).
        question : str
            User query used as the ranking anchor.

        Returns
        -------
        list[Paper]
            Same papers with ``relevance_score`` populated, sorted descending.
        """
        if not papers:
            logger.warning("rank_by_relevance: received empty paper list.")
            return papers

        # Build text corpus: one string per paper
        paper_texts = [
            f"{p.title}. {p.abstract}".strip() for p in papers
        ]

        logger.info(
            f"rank_by_relevance: embedding {len(papers)} papers + query…"
        )

        # Encode everything in a single batch for efficiency
        all_texts = [question] + paper_texts
        embeddings = self._embedding_service.encode_batch(all_texts)

        query_emb = embeddings[0]           # shape (dim,)
        paper_embs = embeddings[1:]         # shape (n_papers, dim)

        # Cosine similarity: dot(q, p) / (|q| * |p|)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        paper_norms = paper_embs / (
            np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-10
        )
        scores = paper_norms @ query_norm   # shape (n_papers,)

        # Hybrid score: semantic relevance + lexical coverage + metadata prior.
        for paper, score in zip(papers, scores):
            paper_text = f"{paper.title}. {paper.abstract}".strip()
            lexical = self._keyword_coverage_score(question, paper_text)
            semantic = float((score + 1.0) / 2.0)  # map cosine [-1,1] to [0,1]
            metadata = self._metadata_prior_score(paper)
            intent_signal = self._intent_signal_score(paper_text, intent)
            setattr(paper, "_semantic_score", semantic)
            setattr(paper, "_lexical_score", lexical)
            setattr(paper, "_intent_score", intent_signal)
            paper.relevance_score = float(
                0.58 * semantic + 0.18 * lexical + 0.14 * metadata + 0.10 * intent_signal
            )

        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        # Off-topic pruning (domain-agnostic): remove papers that are both
        # semantically weak and lexically weak for the current question.
        min_semantic = 0.25
        min_lexical = 0.10 if len(self._extract_focus_terms(question)) >= 2 else 0.0
        pruned: List[Paper] = []
        for paper in papers:
            sem = float(getattr(paper, "_semantic_score", 0.0))
            lex = float(getattr(paper, "_lexical_score", 0.0))
            if sem >= min_semantic or lex >= min_lexical:
                pruned.append(paper)

        if len(pruned) >= 5:
            papers = pruned

        # Credibility pruning (domain-agnostic): when enough papers exist and
        # citation data is present, drop the bottom citation quartile unless
        # a paper has exceptionally high query relevance.
        if len(papers) >= 8:
            citation_vals = np.array([max(int(p.citation_count or 0), 0) for p in papers], dtype=float)
            if float(np.count_nonzero(citation_vals)) >= max(4.0, len(papers) * 0.5):
                citation_floor = float(np.percentile(citation_vals, 25))
                credible = [
                    p for p in papers
                    if (float(p.citation_count or 0) >= citation_floor) or (p.relevance_score >= 0.70)
                ]
                if len(credible) >= 5:
                    papers = credible

        # Step 11 diagnostic logging — score distribution
        score_arr = np.array([p.relevance_score for p in papers])
        logger.info(
            "rank_by_relevance: n=%d | mean=%.4f | p25=%.4f | p50=%.4f | p75=%.4f | max=%.4f",
            len(papers),
            float(score_arr.mean()),
            float(np.percentile(score_arr, 25)),
            float(np.percentile(score_arr, 50)),
            float(np.percentile(score_arr, 75)),
            float(score_arr.max()),
        )
        logger.info("rank_by_relevance: top-3 results:")
        for i, p in enumerate(papers[:3], 1):
            logger.info(f"  #{i}  score={p.relevance_score:.4f}  |  {p.title[:80]}")

        return papers

# ---------------------------------------------------------------------------
# End-to-end smoke-test: Retriever -> Filter
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    # Add project root to path so relative imports work when run directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.paper_retriever import PaperRetriever

    question = "How do fungal pathogens develop resistance to antifungal treatments in wheat?"
    domain = "Agriculture"
    TOP_K = 5

    print()
    print("=" * 70)
    print(f"  Question : {question}")
    print(f"  Domain   : {domain}")
    print("=" * 70)

    # Stage 2: retrieve
    retriever = PaperRetriever(top_n=20)
    raw_papers = retriever.retrieve(question, domain)
    print(f"\n  [Retriever] fetched {len(raw_papers)} raw papers.\n")

    # Stage 3: filter + rank
    paper_filter = PaperFilter()
    final_papers = paper_filter.filter(raw_papers, question, top_k=TOP_K)

    print()
    print(f"  [Filter] Final ranked papers (top {TOP_K}):")
    print("  " + "-" * 66)
    for i, paper in enumerate(final_papers, 1):
        print(f"\n  [{i}] {paper.title}")
        print(f"       Score     : {paper.relevance_score:.4f}")
        print(f"       Year      : {paper.year}")
        print(f"       Citations : {paper.citation_count}")
        print(f"       Source    : {paper.source}")
        if paper.abstract:
            snippet = paper.abstract[:130].replace("\n", " ")
            print(f"       Abstract  : {snippet}…")
    print()
