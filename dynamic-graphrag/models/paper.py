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
        citation_count: Number of citations the paper has received.
        source: API or source from which the paper was retrieved
                (e.g., "semantic_scholar" or "openalex").
        embedding: Dense vector embedding of the paper (title + abstract).
        relevance_score: Score assigned during filtering (higher = more relevant).
    """

    paper_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    citation_count: int = 0
    source: str = ""
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        """
        Serialize the Paper instance to a plain dictionary (embedding excluded).

        Returns:
            dict: Dictionary representation of the paper.
        """
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "citation_count": self.citation_count,
            "source": self.source,
            "relevance_score": self.relevance_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        """
        Deserialize a Paper instance from a dictionary.

        Args:
            data: Dictionary containing paper fields.

        Returns:
            Paper: Reconstructed Paper instance.
        """
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=data.get("authors", []),
            year=data.get("year"),
            citation_count=data.get("citation_count", 0),
            source=data.get("source", ""),
            relevance_score=data.get("relevance_score", 0.0),
        )
