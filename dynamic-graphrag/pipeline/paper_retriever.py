"""
Stage 2 — Paper Retriever: fetch candidate papers from external academic APIs.
"""

from typing import List, Optional

from models.paper import Paper


class PaperRetriever:
    """
    Queries academic literature APIs (Semantic Scholar, OpenAlex) to retrieve
    a pool of candidate papers relevant to the user query and detected domain.

    Attributes:
        semantic_scholar_api_key: API key for Semantic Scholar.
        openalex_email: Polite-pool email for the OpenAlex API.
        top_n: Maximum number of papers to return per source.
    """

    def __init__(
        self,
        semantic_scholar_api_key: Optional[str] = None,
        openalex_email: Optional[str] = None,
        top_n: int = 20,
    ) -> None:
        """
        Initialize the retriever with API credentials.

        Args:
            semantic_scholar_api_key: Key for authenticated Semantic Scholar requests.
            openalex_email: Email address for OpenAlex polite-pool access.
            top_n: Number of papers to retrieve per API. Defaults to 20.
        """
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.openalex_email = openalex_email
        self.top_n = top_n

    def retrieve(self, query: str, domain: str) -> List[Paper]:
        """
        Retrieve papers from all available sources and merge the results.

        Args:
            query: Natural-language search query.
            domain: Scientific domain label to scope the search.

        Returns:
            List[Paper]: Combined, deduplicated list of candidate papers.
        """
        pass

    def retrieve_from_semantic_scholar(self, query: str) -> List[Paper]:
        """
        Query the Semantic Scholar Graph API for papers matching the query.

        Args:
            query: Search string to submit to the API.

        Returns:
            List[Paper]: Papers returned by Semantic Scholar.
        """
        pass

    def retrieve_from_openalex(self, query: str) -> List[Paper]:
        """
        Query the OpenAlex API for papers matching the query.

        Args:
            query: Search string to submit to the API.

        Returns:
            List[Paper]: Papers returned by OpenAlex.
        """
        pass

    def _parse_semantic_scholar_response(self, response: dict) -> List[Paper]:
        """
        Parse a raw Semantic Scholar API response into Paper objects.

        Args:
            response: JSON-decoded API response dictionary.

        Returns:
            List[Paper]: Parsed Paper instances.
        """
        pass

    def _parse_openalex_response(self, response: dict) -> List[Paper]:
        """
        Parse a raw OpenAlex API response into Paper objects.

        Args:
            response: JSON-decoded API response dictionary.

        Returns:
            List[Paper]: Parsed Paper instances.
        """
        pass
