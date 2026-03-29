"""
Data model representing a scientific paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Paper:
    """
    Represents a retrieved scientific paper with its metadata and content.

    Attributes:
        paper_id: Unique identifier from the source API (e.g., Semantic Scholar ID).
        title: Title of the paper.
        abstract: Abstract text of the paper.
        authors: List of author names.
        year: Publication year.
        venue: Journal or conference name.
        citations: Number of citations the paper has received.
        url: URL to the paper's page or PDF.
        embedding: Dense vector embedding of the paper (abstract or title+abstract).
        relevance_score: Score assigned during filtering (higher = more relevant).
        source: API or source from which the paper was retrieved (e.g., "semantic_scholar").
    """

    paper_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citations: int = 0
    url: Optional[str] = None
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0
    source: str = ""

    def to_dict(self) -> dict:
        """
        Serialize the Paper instance to a plain dictionary.

        Returns:
            dict: Dictionary representation of the paper (embedding excluded).
        """
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        """
        Deserialize a Paper instance from a dictionary.

        Args:
            data: Dictionary containing paper fields.

        Returns:
            Paper: Reconstructed Paper instance.
        """
        pass
