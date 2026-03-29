"""
Data model representing a node in the knowledge graph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphNode:
    """
    Represents a node in the dynamic knowledge graph.

    A node can correspond to a paper, an entity (concept, method, dataset),
    or an author extracted from the scientific literature.

    Attributes:
        node_id: Unique identifier for the node within the graph.
        node_type: Category of the node, e.g. "paper", "entity", "author".
        label: Human-readable name or title of the node.
        properties: Arbitrary key-value metadata associated with the node.
        embedding: Dense vector representation used for similarity retrieval.
        neighbors: List of node IDs this node is directly connected to.
    """

    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    neighbors: List[str] = field(default_factory=list)

    def add_neighbor(self, node_id: str) -> None:
        """
        Register a neighboring node by its ID.

        Args:
            node_id: The ID of the adjacent node to add.
        """
        pass

    def to_dict(self) -> dict:
        """
        Serialize the GraphNode to a plain dictionary.

        Returns:
            dict: Dictionary representation of the node (embedding excluded).
        """
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "GraphNode":
        """
        Deserialize a GraphNode from a dictionary.

        Args:
            data: Dictionary containing node fields.

        Returns:
            GraphNode: Reconstructed GraphNode instance.
        """
        pass
