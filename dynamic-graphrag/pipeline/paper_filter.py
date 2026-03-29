"""
Stage 3 — Paper Filter: rank and prune the retrieved paper pool.
"""

from typing import List

from models.paper import Paper
from config.settings import TOP_N_PAPERS, EMBEDDING_MODEL


class PaperFilter:
    """
    Scores and selects the most relevant papers from the retrieval pool
    using semantic similarity and metadata-based heuristics.

    Embeddings are computed once and reused across calls within a session.

    Attributes:
        embedding_model: Name of the sentence-transformers model to use.
        top_n: Number of papers to keep after filtering.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, top_n: int = TOP_N_PAPERS) -> None:
        """
        Initialize the filter and load the embedding model.

        Args:
            embedding_model: Sentence-transformers model name.
            top_n: How many top-scoring papers to retain.
        """
        self.embedding_model = embedding_model
        self.top_n = top_n

    def filter(self, papers: List[Paper], query: str) -> List[Paper]:
        """
        Score all papers against the query and return the top_n most relevant.

        Args:
            papers: Candidate papers from the retrieval stage.
            query: Original user query used as the reference for scoring.

        Returns:
            List[Paper]: Top-ranked papers sorted by descending relevance score.
        """
        pass

    def embed_query(self, query: str) -> List[float]:
        """
        Compute a dense embedding for the query string.

        Args:
            query: Text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        pass

    def embed_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Compute and attach embeddings to each paper in the list.
        The embedding is derived from the paper's title and abstract.

        Args:
            papers: Papers that need embeddings attached.

        Returns:
            List[Paper]: Same papers with the embedding field populated.
        """
        pass

    def score_papers(self, papers: List[Paper], query_embedding: List[float]) -> List[Paper]:
        """
        Assign a relevance_score to each paper based on cosine similarity
        to the query embedding, optionally boosted by citation count.

        Args:
            papers: Papers with pre-computed embeddings.
            query_embedding: Dense vector of the user query.

        Returns:
            List[Paper]: Papers with relevance_score populated.
        """
        pass
