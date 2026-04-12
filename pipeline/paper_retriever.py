"""
Stage 2 — Paper Retriever: fetch candidate papers from external academic APIs.

Primary source  : Europe PMC API (free, no auth required, good agricultural coverage)
Secondary source: arXiv API (free preprints, good for cutting-edge research)
Fallback source : OpenAlex API (used when others return no results)

Upgrades
--------
- Step 1: LLM-based query expansion (domain-agnostic, via Groq)
- Step 2: Dual retrieval (original + expanded query) with deduplication
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Optional
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from groq import Groq
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config.settings import TOP_N_PAPERS

load_dotenv()
from models.paper import Paper
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EPMC_FORMAT = "json"

_ARXIV_BASE_URL = "http://export.arxiv.org/api/query"

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
    Retrieves scientific papers from Europe PMC, arXiv, and OpenAlex.

    The :meth:`retrieve` method is the main entry point.  It optionally
    expands the query with an LLM (Step 1), retrieves papers for both the
    original and expanded queries (Step 2 — dual retrieval), and deduplicates.

    Attributes
    ----------
    top_n : int
        Maximum number of papers to request from each API per query.
    use_query_expansion : bool
        If True, use Groq LLM to expand the query before retrieval.
    """

    def __init__(
        self,
        top_n: int = TOP_N_PAPERS,
        use_query_expansion: bool = True,
    ) -> None:
        """
        Initialise the retriever.

        Parameters
        ----------
        top_n : int
            Maximum papers to retrieve per source. Defaults to
            ``config.settings.TOP_N_PAPERS``.
        use_query_expansion : bool
            Whether to use LLM-based query expansion. Defaults to True.
        """
        self.top_n = top_n
        self.use_query_expansion = use_query_expansion
        self.retrieved_papers_count: int = 0
        self._session = requests.Session()
        self._groq_client: Optional[Groq] = None
        self._query_expansion_cache: dict[str, str] = {}
        self._query_expansion_lock = Lock()

        retry = Retry(
            total=_MAX_RETRIES,
            connect=_MAX_RETRIES,
            read=_MAX_RETRIES,
            status=_MAX_RETRIES,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _anchor_terms(self, question: str) -> List[str]:
        """Extract important non-stopword anchors from the original question."""
        tokens = re.findall(r"\b[a-z0-9\-]{4,}\b", (question or "").lower())
        anchors: List[str] = []
        for tok in tokens:
            if tok in _STOP_WORDS:
                continue
            if tok not in anchors:
                anchors.append(tok)
        return anchors[:10]

    def _is_expansion_anchored(self, question: str, expanded: str) -> bool:
        """Return True when expanded query preserves enough original anchors."""
        anchors = self._anchor_terms(question)
        if not anchors:
            return True
        exp_l = (expanded or "").lower()
        overlap = sum(1 for a in anchors if a in exp_l)
        return overlap / max(len(anchors), 1) >= 0.5

    def expand_query(self, question: str) -> str:
        """
        Step 1 — LLM-based query expansion (domain-agnostic).

        Sends the question to the Groq LLM with a general-purpose prompt that
        asks for key concepts, related terms, synonyms, and implicit subtopics.
        Returns only the expanded query string.  Falls back to the original
        question if the LLM call fails.

        Parameters
        ----------
        question : str
            Original natural-language user question.

        Returns
        -------
        str
            Dense, expanded query string for retrieval.
        """
        cache_key = question.strip().lower()
        with self._query_expansion_lock:
            cached = self._query_expansion_cache.get(cache_key)
        if cached is not None:
            logger.info("Query expansion cache hit for question '%s'", question[:60])
            return cached

        try:
            client = self._get_groq_client()
            prompt = (
                "Expand the following question into a dense search query for "
                "academic paper retrieval. Include:\n"
                "- key concepts and entities\n"
                "- related technical terms\n"
                "- possible synonyms and alternative phrasings\n"
                "- implicit subtopics that papers addressing this question would cover\n\n"
                "Constraints:\n"
                "- Preserve original domain intent and core entities.\n"
                "- Do NOT shift topic to adjacent fields unless explicitly requested.\n"
                "- Keep wording aligned with the original problem statement.\n\n"
                "Return ONLY a short expanded query (2-3 lines max). "
                "No explanation, no bullet points, no preamble.\n\n"
                f"Question: {question}"
            )
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=120,
            )
            expanded = response.choices[0].message.content.strip()
            if not self._is_expansion_anchored(question, expanded):
                logger.warning("Query expansion drift detected; falling back to original query.")
                expanded = question
            logger.info("Query expansion: '%s' → '%s'", question[:60], expanded[:100])
            with self._query_expansion_lock:
                self._query_expansion_cache[cache_key] = expanded
            return expanded
        except Exception as e:
            logger.warning("Query expansion failed (%s); using original query.", e)
            with self._query_expansion_lock:
                self._query_expansion_cache[cache_key] = question
            return question

    def _get_groq_client(self) -> Groq:
        """Return cached Groq client, creating it on first call."""
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set.")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def get_source_priority(self, domain: str, question: str) -> List[str]:
        """
        Decide retrieval source order based on domain/question keywords.

        Parameters
        ----------
        domain : str
            Detected domain label from stage 1.
        question : str
            Original user question.

        Returns
        -------
        list[str]
            Ordered source keys in priority order.
        """
        d = (domain or "").lower()
        q = (question or "").lower()
        text = f"{d} {q}"

        # Domain-aware routing heuristics.
        if any(k in text for k in [
            "agric", "crop", "plant", "fung", "disease", "pathogen",
            "biolog", "biomed", "medicine", "health", "microb",
            "environment", "ecolog",
        ]):
            return ["europe_pmc", "arxiv", "openalex"]

        if any(k in text for k in [
            "machine learning", "deep learning", "transformer", "llm", "ai",
            "computer vision", "nlp", "algorithm", "optimization", "theorem",
            "physics", "math", "quantum",
        ]):
            return ["arxiv", "europe_pmc", "openalex"]

        if any(k in text for k in [
            "geoscience", "earthquake", "tectonic", "climate", "soil",
            "lithium", "mining", "mineral", "geology", "extraction", "geothermal",
        ]):
            return ["europe_pmc", "openalex", "arxiv"]

        # Default robust route.
        return ["europe_pmc", "arxiv", "openalex"]

    def retrieve(self, question: str, domain: str, expanded_question: Optional[str] = None) -> List[Paper]:
        """
        Main entry point: retrieve papers relevant to the question and domain.

        Steps
        -----
        1. Optionally expand the query with the LLM (Step 1).
        2. Build API search strings for both original and expanded queries.
        3. Fetch papers for each query from the priority-ordered sources
           (Step 2 — dual retrieval).
        4. Merge and deduplicate by paper ID / title.

        Parameters
        ----------
        question : str
            Natural-language scientific question from the user.
        domain : str
            Detected scientific domain (e.g., ``"Agriculture"``).

        Returns
        -------
        list[Paper]
            Retrieved papers, deduplicated, up to ``2 * self.top_n`` entries.
        """
        # Step 1 — LLM query expansion
        if expanded_question is None:
            expanded_question = self.expand_query(question) if self.use_query_expansion else question

        query_original = self.build_query(question, domain)
        query_expanded = self.build_query(expanded_question, domain)
        logger.info("Original search query : '%s'", query_original)
        logger.info("Expanded search query : '%s'", query_expanded[:120])

        source_priority = self.get_source_priority(domain, question)
        logger.info(
            "Source routing for domain '%s': %s",
            domain,
            " -> ".join(source_priority),
        )

        fetchers = {
            "europe_pmc": self.fetch_europe_pmc,
            "arxiv": self.fetch_arxiv,
            "openalex": self.fetch_openalex,
        }
        source_labels = {
            "europe_pmc": "Europe PMC",
            "arxiv": "arXiv",
            "openalex": "OpenAlex",
        }

        # Step 2 — Parallel dual retrieval: fire all (source, query) pairs concurrently.
        # Build task list: original query for each source, then expanded if different.
        tasks: List[tuple] = []
        for source_key in source_priority:
            tasks.append((source_key, query_original, "original"))
        if query_expanded.strip() != query_original.strip():
            for source_key in source_priority:
                tasks.append((source_key, query_expanded, "expanded"))

        def _fetch_task(source_key: str, query: str, label: str) -> tuple:
            fetcher = fetchers.get(source_key)
            if fetcher is None:
                return (source_key, label, [])
            try:
                papers = fetcher(query)
                return (source_key, label, papers or [])
            except Exception as exc:
                logger.warning("Fetch error [%s / %s]: %s", label, source_key, exc)
                return (source_key, label, [])

        all_papers: List[Paper] = []
        seen_ids: set = set()

        max_workers = min(len(tasks), 6)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_fetch_task, sk, q, lbl): (sk, lbl)
                for sk, q, lbl in tasks
            }
            for future in as_completed(futures):
                source_key, label, papers = future.result()
                if papers:
                    added = 0
                    for p in papers:
                        dedup_key = p.paper_id or p.title.lower().strip()
                        if dedup_key and dedup_key not in seen_ids:
                            seen_ids.add(dedup_key)
                            all_papers.append(p)
                            added += 1
                    logger.info(
                        "[%s / %s] %d new papers added (%d total).",
                        label,
                        source_labels.get(source_key, source_key),
                        added,
                        len(all_papers),
                    )
                else:
                    logger.warning(
                        "[%s] %s returned no results.",
                        label,
                        source_labels.get(source_key, source_key),
                    )

        self.retrieved_papers_count += len(all_papers)
        logger.info(
            "Dual retrieval complete: %d deduplicated papers (original + expanded).",
            len(all_papers),
        )

        if not all_papers:
            logger.warning("All sources returned no results for query: '%s'", query_original)

        return all_papers

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

    def fetch_europe_pmc(self, query: str) -> List[Paper]:
        """
        Query the Europe PMC API for papers matching *query*.

        Europe PMC provides excellent coverage of agricultural, biomedical,
        and life sciences research with free, unauthenticated access.

        Retries up to ``_MAX_RETRIES`` times with a ``_RETRY_DELAY`` second
        pause between attempts.

        Parameters
        ----------
        query : str
            Search string as built by :meth:`build_query`.

        Returns
        -------
        list[Paper]
            Papers returned by Europe PMC, or ``[]`` on failure.
        """
        params = {
            "query": query,
            "pageSize": self.top_n,
            "format": _EPMC_FORMAT,
            "sortBy": "CITED",  # Sort by citation count
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(f"Europe PMC request (attempt {attempt}): {params}")
                response = self._session.get(_EPMC_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                papers = self._parse_europe_pmc(data)
                logger.info(f"Europe PMC: {len(papers)} papers retrieved.")
                return papers

            except requests.exceptions.HTTPError as e:
                logger.warning(f"Europe PMC HTTP error (attempt {attempt}): {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Europe PMC connection error (attempt {attempt}): {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"Europe PMC request timed out (attempt {attempt}).")
            except Exception as e:
                logger.error(f"Unexpected error from Europe PMC (attempt {attempt}): {e}")

            if attempt < _MAX_RETRIES:
                logger.debug(f"Retrying in {_RETRY_DELAY}s…")
                time.sleep(_RETRY_DELAY)

        logger.error("Europe PMC: all retries exhausted.")
        return []

    def fetch_arxiv(self, query: str) -> List[Paper]:
        """
        Query the arXiv API for papers matching *query*.

        arXiv returns results in XML format and provides access to preprints
        across many disciplines including computer science, physics, and
        quantitative biology.

        Retries up to ``_MAX_RETRIES`` times with a ``_RETRY_DELAY`` second
        pause between attempts.

        Parameters
        ----------
        query : str
            Search string as built by :meth:`build_query`.

        Returns
        -------
        list[Paper]
            Papers returned by arXiv, or ``[]`` on failure.
        """
        # Build search query with boolean operators for better results
        arxiv_query = "+AND+".join(query.split())
        params = {
            "search_query": f"all:{arxiv_query}",
            "max_results": self.top_n,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.debug(f"arXiv request (attempt {attempt}): {params}")
                response = self._session.get(_ARXIV_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                papers = self._parse_arxiv(response.text)
                logger.info(f"arXiv: {len(papers)} papers retrieved.")
                return papers

            except requests.exceptions.HTTPError as e:
                logger.warning(f"arXiv HTTP error (attempt {attempt}): {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"arXiv connection error (attempt {attempt}): {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"arXiv request timed out (attempt {attempt}).")
            except Exception as e:
                logger.error(f"Unexpected error from arXiv (attempt {attempt}): {e}")

            if attempt < _MAX_RETRIES:
                logger.debug(f"Retrying in {_RETRY_DELAY}s…")
                time.sleep(_RETRY_DELAY)

        logger.error("arXiv: all retries exhausted.")
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
                response = self._session.get(_OA_BASE_URL, params=params, timeout=10)
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

    def _parse_europe_pmc(self, data: dict) -> List[Paper]:
        """
        Parse a Europe PMC API response into a list of Paper objects.

        Parameters
        ----------
        data : dict
            JSON-decoded response from the Europe PMC API.

        Returns
        -------
        list[Paper]
            Parsed papers; entries with missing titles are skipped.
        """
        papers: List[Paper] = []

        # Europe PMC may return resultList as {"result": [...]}.
        result_list = data.get("resultList", {}) if isinstance(data, dict) else {}
        if isinstance(result_list, dict):
            raw_items = result_list.get("result", [])
        elif isinstance(result_list, list):
            raw_items = result_list
        else:
            raw_items = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            if not title:
                continue

            abstract = (item.get("abstractText") or "").strip()
            
            # Skip papers without abstracts (less useful for entity extraction)
            if not abstract:
                continue

            author_list = item.get("authorList", {})
            author_items = []
            if isinstance(author_list, dict):
                author_items = author_list.get("author", []) or []
            elif isinstance(author_list, list):
                author_items = author_list

            authors = []
            for a in author_items:
                if isinstance(a, dict):
                    name = (a.get("fullName") or "").strip()
                    if name:
                        authors.append(name)

            papers.append(Paper(
                paper_id=item.get("pmid") or item.get("id") or "",
                title=title,
                abstract=abstract,
                authors=authors,
                year=item.get("pubYear"),
                citation_count=item.get("citedByCount") or 0,
                source="europe_pmc",
            ))
        return papers

    def _parse_arxiv(self, xml_response: str) -> List[Paper]:
        """
        Parse an arXiv API response (XML) into a list of Paper objects.

        Parameters
        ----------
        xml_response : str
            XML string returned by the arXiv API.

        Returns
        -------
        list[Paper]
            Parsed papers; entries with missing titles or abstracts are skipped.
        """
        papers: List[Paper] = []
        
        try:
            root = ET.fromstring(xml_response)
            # arXiv uses Atom namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                title = (title_elem.text or "").strip() if title_elem is not None else ""
                if not title:
                    continue

                summary_elem = entry.find("atom:summary", ns)
                abstract = (summary_elem.text or "").strip() if summary_elem is not None else ""
                
                # Skip papers without abstracts
                if not abstract:
                    continue

                # Extract authors
                authors = []
                for author_elem in entry.findall("atom:author", ns):
                    name_elem = author_elem.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text or "")

                # Extract paper ID (arxiv ID)
                id_elem = entry.find("atom:id", ns)
                paper_id = ""
                if id_elem is not None and id_elem.text:
                    # Extract ID from URL like http://arxiv.org/abs/2023.xxxxx
                    paper_id = id_elem.text.split("/abs/")[-1] if "/abs/" in id_elem.text else id_elem.text

                # Extract publication date (year)
                published_elem = entry.find("atom:published", ns)
                year = None
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, IndexError):
                        year = None

                papers.append(Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    citation_count=0,  # arXiv doesn't provide citation counts in feed
                    source="arxiv",
                ))
        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv XML response: {e}")
            return []

        return papers

    def _parse_openalex(self, data: dict) -> List[Paper]:
        """
        Parse an OpenAlex API response into a list of Paper objects.
        
        Filters out very low-quality papers (abstracts missing or <50 words,
        zero citations and published >5 years ago) to prioritize substantive,
        cited research.

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
        from datetime import datetime
        current_year = datetime.now().year
        
        for item in data.get("results", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            abstract = self.rebuild_abstract(
                item.get("abstract_inverted_index") or {}
            )
            
            # Filter: skip papers with no abstract (less useful for entity extraction)
            if not abstract or len(abstract.split()) < 50:
                continue
            
            # Filter: skip very old papers with no citations
            year = item.get("publication_year") or 0
            citations = item.get("cited_by_count") or 0
            if year < current_year - 5 and citations == 0:
                continue

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
                year=year,
                citation_count=citations,
                source="openalex",
            ))
        
        # Sort by citations descending (best papers first)
        papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
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
