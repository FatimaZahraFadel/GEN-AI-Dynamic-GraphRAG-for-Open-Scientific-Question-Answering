"""
Data models for nodes and edges in the dynamic knowledge graph.

Classes
-------
GraphNode  — generic graph node (paper, entity, author)
Entity     — a named scientific entity extracted from a paper abstract
Relation   — a directed relationship between two entities
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """
    Represents a node in the dynamic knowledge graph.

    A node can correspond to a paper, an entity (concept, method, dataset),
    or an author extracted from the scientific literature.

    Attributes
    ----------
    node_id : str
        Unique identifier for the node within the graph.
    node_type : str
        Category of the node, e.g. ``"paper"``, ``"entity"``, ``"author"``.
    label : str
        Human-readable name or title of the node.
    properties : dict
        Arbitrary key-value metadata associated with the node.
    embedding : list[float] or None
        Dense vector representation used for similarity retrieval.
    neighbors : list[str]
        List of node IDs this node is directly connected to.
    """

    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    neighbors: List[str] = field(default_factory=list)

    def add_neighbor(self, node_id: str) -> None:
        """
        Register a neighbouring node by its ID (no-op if already present).

        Parameters
        ----------
        node_id : str
            The ID of the adjacent node to add.
        """
        if node_id not in self.neighbors:
            self.neighbors.append(node_id)

    def to_dict(self) -> dict:
        """
        Serialise the GraphNode to a plain dictionary (embedding excluded).

        Returns
        -------
        dict
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "properties": self.properties,
            "neighbors": self.neighbors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphNode":
        """
        Deserialise a GraphNode from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing node fields.

        Returns
        -------
        GraphNode
            Reconstructed GraphNode instance.
        """
        return cls(
            node_id=data.get("node_id", ""),
            node_type=data.get("node_type", ""),
            label=data.get("label", ""),
            properties=data.get("properties", {}),
            neighbors=data.get("neighbors", []),
        )


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    A named scientific entity extracted from a paper's abstract.

    Attributes
    ----------
    id : str
        Slugified identifier derived from the label
        (lowercase, spaces replaced by underscores), e.g. ``"rust_disease"``.
    label : str
        Original human-readable name, e.g. ``"Rust Disease"``.
    type : str
        Entity category.  One of:
        ``"Concept / Entity"``, ``"Problem / Condition"``,
        ``"Method / Intervention"``, ``"Context / Location"``,
        ``"Cause / Factor"``, ``"Effect / Outcome"``.
    source_paper_id : str
        ``paper_id`` of the Paper from which this entity was extracted.
    """

    id: str
    label: str
    type: str
    source_paper_id: str

    def to_dict(self) -> dict:
        """
        Serialise the Entity to a plain dictionary.

        Returns
        -------
        dict
            Dictionary representation of the entity.
        """
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "source_paper_id": self.source_paper_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """
        Deserialise an Entity from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing entity fields.

        Returns
        -------
        Entity
            Reconstructed Entity instance.
        """
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            type=data.get("type", ""),
            source_paper_id=data.get("source_paper_id", ""),
        )


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """
    A directed relationship between two entities in the knowledge graph.

    Attributes
    ----------
    source_id : str
        ``id`` of the source Entity.
    target_id : str
        ``id`` of the target Entity.
    relation_type : str
        Type of the relationship.  One of:
        ``"affects"``, ``"treats"``, ``"occurs_in"``, ``"caused_by"``,
        ``"leads_to"``, ``"detected_by"``, ``"studied_in"``.
    source_paper_id : str
        ``paper_id`` of the Paper from which this relation was extracted.
    """

    source_id: str
    target_id: str
    relation_type: str
    source_paper_id: str

    def to_dict(self) -> dict:
        """
        Serialise the Relation to a plain dictionary.

        Returns
        -------
        dict
            Dictionary representation of the relation.
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "source_paper_id": self.source_paper_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        """
        Deserialise a Relation from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing relation fields.

        Returns
        -------
        Relation
            Reconstructed Relation instance.
        """
        return cls(
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("relation_type", ""),
            source_paper_id=data.get("source_paper_id", ""),
        )
