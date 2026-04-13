"""
Confidence & Coverage Scoring: assess answer confidence based on graph structure
and supporting evidence.

Produces confidence and coverage scores based on:
- Graph density and connectivity
- Number of supporting papers
- Reasoning path count and quality
- Entity type diversity
"""

from typing import Dict, List, Optional
import numpy as np
import networkx as nx

from models.paper import Paper
from utils.logger import get_logger

logger = get_logger(__name__)


# Global scorer instance
_scorer = None


def score_confidence_and_coverage(subgraph, papers, reasoning_paths, seed_scores, query_plan=None):
    """
    Convenience function for scoring confidence and coverage.
    """
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer.score(subgraph, papers, reasoning_paths, seed_scores, query_plan)


class ConfidenceScorer:
    """
    Scores the confidence and coverage of an answer based on retrieved graph
    and supporting evidence.

    Confidence: how confident we are in the answer (0-1)
    Coverage: how well the graph covers the query intent (0-1)
    """

    def __init__(self):
        pass

    def score(
        self,
        subgraph: nx.DiGraph,
        papers: List[Paper],
        reasoning_paths: List[Dict],
        seed_scores: Dict[str, float],
        query_plan: Optional[Dict] = None,
    ) -> Dict:
        """
        Main entry point: compute confidence and coverage scores.

        Parameters
        ----------
        subgraph : nx.DiGraph
            Retrieved knowledge graph subgraph.
        papers : List[Paper]
            Supporting papers used as evidence.
        reasoning_paths : List[Dict]
            Multi-hop reasoning paths extracted from graph.
        seed_scores : Dict[str, float]
            Similarity scores for seed entities.
        query_plan : Dict, optional
            Query plan with type, entities, dimensions.

        Returns
        -------
        Dict
            Score dictionary with keys:
            - confidence: float [0, 1]
            - coverage: float [0, 1]
            - supporting_papers: int
            - reasoning_paths_count: int
            - graph_density: float
            - warnings: List[str]
        """
        warnings = []

        # Graph structure scores
        graph_density = self._compute_density(subgraph)
        avg_degree = self._compute_avg_degree(subgraph)
        node_count = subgraph.number_of_nodes()
        edge_count = subgraph.number_of_edges()

        # Evidence strength
        paper_signal = self._score_papers(papers)
        path_signal = self._score_reasoning_paths(reasoning_paths)
        seed_signal = self._score_seeds(seed_scores)

        # Graph coverage of plan requirements
        coverage = self._compute_coverage(subgraph, query_plan)

        # Composite confidence score
        confidence = (
            0.30 * graph_density
            + 0.25 * float(np.tanh(avg_degree / max(node_count, 1)))
            + 0.20 * paper_signal
            + 0.15 * path_signal
            + 0.10 * seed_signal
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Generate warnings for low-confidence scenarios
        if node_count < 3:
            warnings.append("sparse_graph: <3 nodes")
        if edge_count < 2:
            warnings.append("disconnected: <2 edges")
        if graph_density < 0.08:
            warnings.append("low_density: <0.08")
        if len(papers) < 2:
            warnings.append("insufficient_evidence: <2 papers")
        if len(reasoning_paths) < 1:
            warnings.append("no_multi_hop_paths")
        if confidence < 0.40:
            warnings.append(f"low_confidence: {confidence:.2f}")

        return {
            "confidence": round(confidence, 4),
            "coverage": round(coverage, 4),
            "supporting_papers": len(papers),
            "reasoning_paths_count": len(reasoning_paths),
            "graph_nodes": node_count,
            "graph_edges": edge_count,
            "graph_density": round(graph_density, 4),
            "avg_degree": round(avg_degree, 4),
            "warnings": warnings,
            "is_confident": confidence >= 0.50,
        }

    def _compute_density(self, graph: nx.DiGraph) -> float:
        """Compute graph density [0, 1]."""
        if graph.number_of_nodes() < 2:
            return 0.0
        density = nx.density(graph)
        return float(np.clip(density, 0.0, 1.0))

    def _compute_avg_degree(self, graph: nx.DiGraph) -> float:
        """Compute average degree (in + out)."""
        if graph.number_of_nodes() == 0:
            return 0.0
        degrees = [graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes()]
        return float(np.mean(degrees)) if degrees else 0.0

    def _score_papers(self, papers: List[Paper]) -> float:
        """Score evidence strength from supporting papers."""
        if len(papers) == 0:
            return 0.0
        if len(papers) < 2:
            return 0.3
        if len(papers) < 4:
            return 0.6
        # Multiple papers with varied citations
        citations = np.array([max(int(p.citation_count or 0), 0) for p in papers], dtype=float)
        citation_diversity = float(np.std(citations)) / (float(np.mean(citations)) + 1.0)
        return float(np.clip(0.7 + 0.3 * citation_diversity / 10.0, 0.0, 1.0))

    def _score_reasoning_paths(self, paths: List[Dict]) -> float:
        """Score multi-hop reasoning evidence."""
        if len(paths) == 0:
            return 0.0
        if len(paths) < 2:
            return 0.4
        # Reward longer paths (more reasoning hops)
        lengths = [p.get("length", 2) for p in paths]
        avg_length = float(np.mean(lengths))
        length_score = float(np.tanh((avg_length - 2.0) / 2.0))
        # Reward paths with multi-paper support
        papers_freq = [p.get("paper_frequency", 0) for p in paths]
        avg_freq = float(np.mean(papers_freq)) if papers_freq else 0.0
        freq_score = float(np.tanh(avg_freq / 2.0))
        return 0.6 * length_score + 0.4 * freq_score

    def _score_seeds(self, seed_scores: Dict[str, float]) -> float:
        """Score query alignment of seed entities."""
        if len(seed_scores) == 0:
            return 0.0
        scores = list(seed_scores.values())
        mean_score = float(np.mean(scores))
        # Map [-1, 1] to [0, 1]
        return float((mean_score + 1.0) / 2.0)

    def _compute_coverage(self, graph: nx.DiGraph, query_plan: Optional[Dict]) -> float:
        """
        Compute coverage of query plan requirements.

        If no plan provided, use structural heuristics.
        """
        if query_plan is None:
            # Fallback: check basic connectivity
            if graph.number_of_nodes() < 2:
                return 0.0
            num_components = nx.number_weakly_connected_components(graph)
            largest_cc = max(
                (len(c) for c in nx.weakly_connected_components(graph)), default=0
            )
            connectivity = largest_cc / max(graph.number_of_nodes(), 1)
            return float(np.clip(1.0 - (num_components / max(graph.number_of_nodes(), 1)), 0.0, 1.0))

        # Plan-driven coverage check
        requirements = query_plan.get("graph_requirements", {})
        min_nodes = requirements.get("min_nodes", 3)
        min_edges = requirements.get("min_edges", 2)
        min_types = requirements.get("min_entity_types", 1)

        node_check = 1.0 if graph.number_of_nodes() >= min_nodes else graph.number_of_nodes() / max(min_nodes, 1)
        edge_check = 1.0 if graph.number_of_edges() >= min_edges else graph.number_of_edges() / max(min_edges, 1)

        entity_types = set()
        for _, attrs in graph.nodes(data=True):
            entity_types.add(attrs.get("type", "unknown"))
        type_check = 1.0 if len(entity_types) >= min_types else len(entity_types) / max(min_types, 1)

        coverage = 0.4 * node_check + 0.4 * edge_check + 0.2 * type_check
        return float(np.clip(coverage, 0.0, 1.0))
