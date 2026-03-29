"""
Stage 2 — Paper Retriever: fetch candidate papers from external academic APIs.

Primary source  : Semantic Scholar Graph API
Fallback source : OpenAlex API (used when Semantic Scholar returns no results)
"""

import time
from typing import List

import requests

from config.settings import TOP_N_PAPERS
from models.paper import Paper
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SS_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = "paperId,title,abstract,year,citationCount,authors"

_OA_BASE_URL = "https://api.openalex.org/works"
_OA_SELECT = "id,title,abstract_inverted_index,publication_year,cited_by_count,authorships"

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "that", "this", "these",
    "those", "what", "how", "why", "when", "where", "which", "who", "whom",
    "it", "its", "as", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "few", "more", "most", "other", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "then", "once", "there", "here", "about",
}

_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds


class PaperRetriever:
    """
    Retrieves scientific papers from Semantic Scholar and OpenAlex.

    The :meth:`retrieve` method is the main entry point.  It builds a clean
    search query from the user's question and domain, queries Semantic Scholar
    first, and falls back to OpenAlex if the primary source returns nothing.

    Attributes
    ----------
    top_n : int
        Maximum number of papers to request from each API.
    """

    def __init__(self, top_n: int = TOP_N_PAPERS) -> None:
        """
        Initialise the retriever.

        Parameters
        ----------
        top_n : int
            Maximum papers to retrieve per source. Defaults to
            ``config.settings.TOP_N_PAPERS``.
        """
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, domain: str) -> List[Paper]:
        """
        Main entry point: retrieve papers relevant to the question and domain.

        Builds a search query with :meth:`build_query`, then tries Semantic
        Scholar.  If no results are returned it falls back to OpenAlex.

        Parameters
        ----------
        question : str
            Natural-language scientific question from the user.
        domain : str
            Detected scientific domain (e.g., ``"Agriculture"``).

        Returns
        -------
        list[Paper]
            Retrieved papers, up to ``self.top_n`` entries.
        """
        query = self.build_query(question, domain)
        logger.info(f"Built search query: '{query}'")

        papers = self.fetch_semantic_scholar(query)

        if papers:
            logger.info(f"Using Semantic Scholar ({len(papers)} papers retrieved).")
        else:
            logger.warning("Semantic Scholar returned no results. Falling back to OpenAlex.")
            papers = self.fetch_openalex(query)
            if papers:
                logger.info(f"Using OpenAlex ({len(papers)} papers retrieved).")
            else:
                logger.warning("Both sources returned no results for query: '%s'", query)

        return papers

    def build_query(self, question: str, domain: str) -> str:
        """
        Build a clean search query from the question and domain.

        Tokenises the question, strips punctuation, removes stop words, and
        appends the domain name so that results are topically anchored.

        Parameters
        ----------
        question : str
            Raw user question.
        domain : str
            Scientific domain to append as a keyword anchor.

        Returns
        -------
        str
            Space-separated query string ready for API submission.
        """
        # Tokenise and strip punctuation
        tokens = []
        for token in question.lower().split():
            clean = "".join(ch for ch in token if ch.isalnum() or ch == "-")
            if clean and clean not in _STOP_WORDS:
                tokens.append(clean)

        # Append domain words (also cleaned), avoiding duplicates
        for word in domain.lower().split():
            if word not in tokens:
                tokens.append(word)

        return " ".join(tokens)

    def fetch_semantic_scholar(self, query: str) -> List[Paper]:
        """
        Query the Semantic Scholar Graph API for papers matching *query*.

        Retries up to ``_MAX_RETRIES`` times with a ``_RETRY_DELAY`` second
        pause between attempts.  HTTP and connection errors are logged and
        result in an empty list rather than an exception.

        Parameters
        ----------
        query : str
            Search string as built by :meth:`build_query`.

        Returns
        -------
        list[Paper]
            Papers returned by Semantic Scholar, or ``[]`` on failure.
        """
        params = {
            "query": query,
            "fields": _SS_FIELDS,
            "limit": self.top_n,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(f"Semantic Scholar request (attempt {attempt}): {params}")
                response = requests.get(_SS_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                papers = self._parse_semantic_scholar(data)
                logger.info(f"Semantic Scholar: {len(papers)} papers retrieved.")
                return papers

            except requests.exceptions.HTTPError as e:
                logger.warning(f"Semantic Scholar HTTP error (attempt {attempt}): {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Semantic Scholar connection error (attempt {attempt}): {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"Semantic Scholar request timed out (attempt {attempt}).")
            except Exception as e:
                logger.error(f"Unexpected error from Semantic Scholar (attempt {attempt}): {e}")

            if attempt < _MAX_RETRIES:
                logger.debug(f"Retrying in {_RETRY_DELAY}s…")
                time.sleep(_RETRY_DELAY)

        logger.error("Semantic Scholar: all retries exhausted.")
        return []

    def fetch_openalex(self, query: str) -> List[Paper]:
        """
        Query the OpenAlex API for papers matching *query*.

        OpenAlex returns abstracts in an inverted-index format; they are
        reconstructed into plain text by :meth:`rebuild_abstract`.

        Retries up to ``_MAX_RETRIES`` times with a ``_RETRY_DELAY`` second
        pause between attempts.

        Parameters
        ----------
        query : str
            Search string as built by :meth:`build_query`.

        Returns
        -------
        list[Paper]
            Papers returned by OpenAlex, or ``[]`` on failure.
        """
        params = {
            "search": query,
            "per_page": self.top_n,
            "select": _OA_SELECT,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(f"OpenAlex request (attempt {attempt}): {params}")
                response = requests.get(_OA_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                papers = self._parse_openalex(data)
                logger.info(f"OpenAlex: {len(papers)} papers retrieved.")
                return papers

            except requests.exceptions.HTTPError as e:
                logger.warning(f"OpenAlex HTTP error (attempt {attempt}): {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"OpenAlex connection error (attempt {attempt}): {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"OpenAlex request timed out (attempt {attempt}).")
            except Exception as e:
                logger.error(f"Unexpected error from OpenAlex (attempt {attempt}): {e}")

            if attempt < _MAX_RETRIES:
                logger.debug(f"Retrying in {_RETRY_DELAY}s…")
                time.sleep(_RETRY_DELAY)

        logger.error("OpenAlex: all retries exhausted.")
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def rebuild_abstract(self, inverted_index: dict) -> str:
        """
        Reconstruct a plain-text abstract from OpenAlex's inverted-index format.

        OpenAlex stores abstracts as ``{word: [position, ...], ...}``.  This
        method inverts the mapping back to a position-ordered word list and
        joins them into a sentence.

        Parameters
        ----------
        inverted_index : dict
            Mapping of ``word -> list[int]`` as returned by OpenAlex.

        Returns
        -------
        str
            Reconstructed abstract string, or ``""`` if the index is empty
            or ``None``.
        """
        if not inverted_index:
            return ""

        # Build position -> word mapping
        position_word: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word

        if not position_word:
            return ""

        ordered_words = [position_word[i] for i in sorted(position_word)]
        return " ".join(ordered_words)

    def _parse_semantic_scholar(self, data: dict) -> List[Paper]:
        """
        Parse a Semantic Scholar API response into a list of Paper objects.

        Parameters
        ----------
        data : dict
            JSON-decoded response from the Semantic Scholar API.

        Returns
        -------
        list[Paper]
            Parsed papers; entries with missing titles are skipped.
        """
        papers: List[Paper] = []
        for item in data.get("data", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]

            papers.append(Paper(
                paper_id=item.get("paperId") or "",
                title=title,
                abstract=(item.get("abstract") or "").strip(),
                authors=authors,
                year=item.get("year"),
                citation_count=item.get("citationCount") or 0,
                source="semantic_scholar",
            ))
        return papers

    def _parse_openalex(self, data: dict) -> List[Paper]:
        """
        Parse an OpenAlex API response into a list of Paper objects.

        Parameters
        ----------
        data : dict
            JSON-decoded response from the OpenAlex API.

        Returns
        -------
        list[Paper]
            Parsed papers; entries with missing titles are skipped.
        """
        papers: List[Paper] = []
        for item in data.get("results", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            abstract = self.rebuild_abstract(
                item.get("abstract_inverted_index") or {}
            )

            # Extract author display names from authorships list
            authors = []
            for authorship in (item.get("authorships") or []):
                author = authorship.get("author") or {}
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            papers.append(Paper(
                paper_id=item.get("id") or "",
                title=title,
                abstract=abstract,
                authors=authors,
                year=item.get("publication_year"),
                citation_count=item.get("cited_by_count") or 0,
                source="openalex",
            ))
        return papers


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys, warnings, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    retriever = PaperRetriever(top_n=5)

    test_cases = [
        ("What fungal diseases affect wheat crops in humid environments?", "Agriculture"),
        ("How do tectonic plate movements trigger earthquakes?", "Geoscience"),
    ]

    for question, domain in test_cases:
        print()
        print("=" * 70)
        print(f"  Question : {question}")
        print(f"  Domain   : {domain}")
        print("=" * 70)

        papers = retriever.retrieve(question, domain)

        if not papers:
            print("  No papers found.")
        else:
            for i, paper in enumerate(papers, 1):
                print(f"\n  [{i}] {paper.title}")
                print(f"       Year    : {paper.year}")
                print(f"       Citations: {paper.citation_count}")
                print(f"       Authors : {', '.join(paper.authors[:3])}"
                      f"{'…' if len(paper.authors) > 3 else ''}")
                print(f"       Source  : {paper.source}")
                if paper.abstract:
                    snippet = paper.abstract[:120].replace("\n", " ")
                    print(f"       Abstract: {snippet}…")
        print()
