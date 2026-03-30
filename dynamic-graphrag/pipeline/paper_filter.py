"""
Stage 3 — Paper Filter: clean, date-filter, and rank retrieved papers.

Pipeline applied inside filter():
    1. filter_by_abstract  — drop papers without a usable abstract
    2. filter_by_date      — drop papers older than min_year
    3. rank_by_relevance   — embed query + papers, sort by cosine similarity
    4. truncate            — return top_k results
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, TOP_N_PAPERS
from models.paper import Paper
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_ABSTRACT_LENGTH = 50


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
    _model : SentenceTransformer or None
        Cached model instance; ``None`` until first use.
    """

    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL) -> None:
        """
        Initialise the filter.

        Parameters
        ----------
        embedding_model_name : str
            Sentence-transformers model name. Defaults to
            ``config.settings.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
        """
        self.embedding_model_name = embedding_model_name
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        papers: List[Paper],
        question: str,
        top_k: int = 30,
    ) -> List[Paper]:
        """
        Main entry point — apply the full filtering and ranking pipeline.

        Steps (in order):
            1. :meth:`filter_by_abstract` — remove papers with no/short abstract
            2. :meth:`filter_by_date`     — remove papers older than min_year
            3. :meth:`rank_by_relevance`  — rank by semantic similarity to query
            4. Truncate to ``top_k``

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
        papers = self.rank_by_relevance(papers, question)
        papers = papers[:top_k]

        logger.info(f"Filter pipeline complete. Kept {len(papers)} papers.")
        return papers

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

    def rank_by_relevance(self, papers: List[Paper], question: str) -> List[Paper]:
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

        model = self._get_model()

        # Build text corpus: one string per paper
        paper_texts = [
            f"{p.title}. {p.abstract}".strip() for p in papers
        ]

        logger.info(
            f"rank_by_relevance: embedding {len(papers)} papers + query…"
        )

        # Encode everything in a single batch for efficiency
        all_texts = [question] + paper_texts
        embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

        query_emb = embeddings[0]           # shape (dim,)
        paper_embs = embeddings[1:]         # shape (n_papers, dim)

        # Cosine similarity: dot(q, p) / (|q| * |p|)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        paper_norms = paper_embs / (
            np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-10
        )
        scores = paper_norms @ query_norm   # shape (n_papers,)

        # Attach scores and sort
        for paper, score in zip(papers, scores):
            paper.relevance_score = float(score)

        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        # Log top-3 for visibility
        logger.info("rank_by_relevance: top-3 results:")
        for i, p in enumerate(papers[:3], 1):
            logger.info(f"  #{i}  score={p.relevance_score:.4f}  |  {p.title[:80]}")

        return papers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """
        Return the cached SentenceTransformer model, loading it on first call.

        Returns
        -------
        SentenceTransformer
            Loaded and ready sentence-transformer model.
        """
        if self._model is None:
            logger.info(
                f"Loading SentenceTransformer model '{self.embedding_model_name}'…"
            )
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model


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
