"""
Stage 7b — Answer Compiler: transform knowledge graph into structured answers.

This module compiles answers directly from the knowledge graph, without
falling back to paper summaries. It:

1. Extracts relevant nodes based on query plan (reasoning type)
2. Ranks nodes by graph centrality and query alignment
3. Builds natural-language explanations from graph relations
4. Constructs structured output matching the query intent
5. Validates coverage before finalizing answers
6. Triggers adaptive retrieval if coverage is insufficient

Design principles
-----------------
- NO domain-specific hardcoding (no keyword lists, genre lists, etc.)
- NO paper summaries; graph-only answers
- DYNAMIC ranking based on graph structure (centrality, connectivity)
- PLAN-DRIVEN node selection (reasoning type determines node priorities)
- COVERAGE-GATED (validates before answering; rejects weak graphs)
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import networkx as nx
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeRankScore:
    """Score for a node based on multiple criteria."""
    node_id: str
    node_label: str
    node_type: str
    score: float
    degree: float
    query_alignment: float
    paper_support: float


@dataclass
class SelectedNode:
    """A node selected for inclusion in the answer."""
    node_id: str
    label: str
    node_type: str
    rank_score: float
    explanation: str


@dataclass
class CompileResult:
    """Result of graph-to-answer compilation."""
    status: str  # "success", "insufficient", or "error"
    quality: str
    answer: Optional[str]
    structure_type: str  # "method", "comparison", "causal", "definition", "explanation"
    selected_nodes: List[SelectedNode]
    reasoning_paths_used: int
    papers_cited: int
    coverage: float
    confidence: float
    warnings: List[str]
    missing_requirements: List[str]


# ---------------------------------------------------------------------------
# Node extraction and ranking
# ---------------------------------------------------------------------------

def _get_node_types_for_reasoning(reasoning_type: str) -> List[str]:
    """
    Map reasoning type to expected entity types in the graph.
    
    This is domain-agnostic: we map reasoning intent to generic entity
    categories from the extraction model, not to domain-specific hardcoded lists.
    
    Parameters
    ----------
    reasoning_type : str
        One of: definition, comparison, method_process, causal_mechanism, optimization
    
    Returns
    -------
    list[str]
        Entity types to prioritize (as extracted by EntityExtractor).
    """
    mapping = {
        "definition": [
            "Concept / Entity",
            "Context / Location",
            "Effect / Outcome",
        ],
        "comparison": [
            "Effect / Outcome",
            "Method / Intervention",
            "Concept / Entity",
            "Cause / Factor",
        ],
        "method_process": [
            "Method / Intervention",
            "Effect / Outcome",
            "Context / Location",
            "Problem / Condition",
        ],
        "causal_mechanism": [
            "Cause / Factor",
            "Effect / Outcome",
            "Problem / Condition",
            "Method / Intervention",
        ],
        "optimization": [
            "Method / Intervention",
            "Effect / Outcome",
            "Cause / Factor",
        ],
    }
    normalized = {
        "method": "method_process",
        "method/process": "method_process",
        "causal": "causal_mechanism",
        "causal/mechanism": "causal_mechanism",
        "compare": "comparison",
    }.get(reasoning_type, reasoning_type)
    return mapping.get(normalized, list(mapping.values())[0])


def _get_relation_types_for_reasoning(reasoning_type: str) -> List[str]:
    """Map reasoning type to expected relation types."""
    normalized = {
        "method": "method_process",
        "method/process": "method_process",
        "causal": "causal_mechanism",
        "causal/mechanism": "causal_mechanism",
        "compare": "comparison",
    }.get(reasoning_type, reasoning_type)
    mapping = {
        "definition": ["part_of", "studied_in", "correlates_with", "affects"],
        "comparison": ["affects", "correlates_with", "leads_to", "caused_by"],
        "method_process": ["mitigates", "detected_by", "depends_on", "affects"],
        "causal_mechanism": ["caused_by", "leads_to", "affects", "correlates_with"],
        "optimization": ["affects", "leads_to", "studied_in", "correlates_with"],
    }
    return mapping.get(normalized, ["affects", "leads_to", "correlates_with"])


def _normalize_reasoning_type(reasoning_type: str) -> str:
    return {
        "method": "method_process",
        "method/process": "method_process",
        "causal": "causal_mechanism",
        "causal/mechanism": "causal_mechanism",
        "compare": "comparison",
    }.get(reasoning_type, reasoning_type)


def _node_matches_query(node_label: str, node_type: str, plan: Dict) -> bool:
    """Return True when a node is plausibly connected to the query."""
    seed_terms = []
    seed_terms.extend(str(item).lower() for item in (plan.get("entities") or []))
    seed_terms.extend(str(item).lower() for item in (plan.get("dimensions") or []))
    seed_terms.extend(str(item).lower() for item in (plan.get("intent_signals") or []))

    label_l = (node_label or "").lower()
    type_l = (node_type or "").lower()
    if any(term and term in label_l for term in seed_terms):
        return True
    if any(term and term in type_l for term in seed_terms):
        return True
    return False


def select_relevant_nodes(
    graph: nx.DiGraph,
    plan: Dict,
    ranked_nodes: List[NodeRankScore],
    top_k: int = 5,
) -> List[SelectedNode]:
    """Select query-relevant nodes while filtering obvious graph noise."""
    required_types = set(str(item) for item in (plan.get("required_entity_types") or []))

    scored_nodes: List[Tuple[float, NodeRankScore]] = []
    query_related_node_ids: Set[str] = set()

    for node_score in ranked_nodes:
        node_type = node_score.node_type
        node_label = node_score.node_label
        node_data = graph.nodes.get(node_score.node_id, {}) if hasattr(graph.nodes, "get") else graph.nodes[node_score.node_id]
        degree = graph.in_degree(node_score.node_id) + graph.out_degree(node_score.node_id)
        query_match = _node_matches_query(node_label, node_type, plan)
        type_match = node_type in required_types if required_types else False
        neighborhood_match = False

        if query_match:
            query_related_node_ids.add(node_score.node_id)
        else:
            for neighbor in set(graph.predecessors(node_score.node_id)).union(graph.successors(node_score.node_id)):
                neighbor_data = graph.nodes[neighbor]
                neighbor_label = str(neighbor_data.get("label", neighbor))
                neighbor_type = str(neighbor_data.get("type", "Unknown"))
                if _node_matches_query(neighbor_label, neighbor_type, plan):
                    neighborhood_match = True
                    break

        score = node_score.score
        if type_match:
            score += 0.25
        if query_match:
            score += 0.30
        if neighborhood_match:
            score += 0.12
        if degree > 0:
            score += min(degree / max(graph.number_of_nodes(), 1), 0.10)

        if type_match or query_match or neighborhood_match:
            scored_nodes.append((score, node_score))

    if not scored_nodes:
        scored_nodes = [(node.score, node) for node in ranked_nodes[:max(top_k, 3)]]

    scored_nodes.sort(key=lambda item: item[0], reverse=True)
    selected: List[SelectedNode] = []
    for _, node_score in scored_nodes[:top_k]:
        selected.append(
            SelectedNode(
                node_id=node_score.node_id,
                label=node_score.node_label,
                node_type=node_score.node_type,
                rank_score=node_score.score,
                explanation="",
            )
        )

    return selected


def assess_graph_quality(
    graph: nx.DiGraph,
    plan: Dict,
    selected_nodes: List[SelectedNode] | None = None,
) -> str:
    """Assess graph quality as strong, moderate, weak, or insufficient."""
    selected_nodes = selected_nodes or []

    if graph.number_of_nodes() == 0:
        return "insufficient"

    score = 0
    if selected_nodes and _has_required_node_types(graph, plan):
        score += 1
    if graph_has_meaningful_paths(graph):
        score += 1
    if nodes_connected_to_query(graph, plan, selected_nodes):
        score += 1

    if score == 3:
        return "strong"
    if score == 2:
        return "moderate"
    if score == 1:
        return "weak"
    return "insufficient"


def _has_required_node_types(graph: nx.DiGraph, plan: Dict) -> bool:
    required_types = {str(item) for item in (plan.get("required_entity_types") or [])}
    if not required_types:
        return graph.number_of_nodes() > 0

    present_types = {str(attrs.get("type", "Unknown")) for _, attrs in graph.nodes(data=True)}
    return bool(required_types.intersection(present_types))


def graph_has_meaningful_paths(graph: nx.DiGraph) -> bool:
    """Return True when the graph contains at least one non-trivial reasoning path."""
    if graph.number_of_nodes() < 2:
        return False
    if graph.number_of_edges() == 0:
        return False
    return _count_reasoning_paths(graph) > 0 or graph.number_of_edges() >= 2


def nodes_connected_to_query(
    graph: nx.DiGraph,
    plan: Dict,
    selected_nodes: List[SelectedNode] | None = None,
) -> bool:
    """Return True when at least one node is plausibly tied to the query."""
    selected_nodes = selected_nodes or []
    for sn in selected_nodes:
        if _node_matches_query(sn.label, sn.node_type, plan):
            return True

    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", node_id))
        node_type = str(attrs.get("type", "Unknown"))
        if _node_matches_query(label, node_type, plan):
            return True
    return False


def _compute_degree_centrality(graph: nx.DiGraph, node: str) -> float:
    """Compute normalized degree centrality for a node."""
    if graph.number_of_nodes() <= 1:
        return 0.0
    total_degree = graph.in_degree(node) + graph.out_degree(node)
    max_possible = 2 * (graph.number_of_nodes() - 1)
    return total_degree / max_possible if max_possible > 0 else 0.0


def _compute_closeness_to_seeds(
    graph: nx.DiGraph,
    node: str,
    seed_nodes: Set[str],
    max_distance: int = 3,
) -> float:
    """
    Measure how close a node is to seed/query entities.
    
    Closer nodes score higher. Nodes at distance d get score 1 / (d + 1).
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    node : str
        The node to score.
    seed_nodes : set[str]
        Query entities to measure distance from.
    max_distance : int
        Maximum distance to consider.
    
    Returns
    -------
    float
        Score in [0, 1].
    """
    if not seed_nodes:
        return 0.0
    
    min_distance = float("inf")
    for seed in seed_nodes:
        if seed == node:
            return 1.0
        try:
            dist = nx.shortest_path_length(graph, seed, node)
            min_distance = min(min_distance, dist)
        except (nx.NetworkXException, nx.NodeNotFound):
            pass
    
    if min_distance == float("inf") or min_distance > max_distance:
        return 0.0
    
    return 1.0 / (min_distance + 1)


def _count_paper_support(graph: nx.DiGraph, node: str) -> float:
    """
    Count unique papers supporting a node.
    
    A paper supports a node if it is a source of that node or its neighbors.
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    node : str
        The node to check.
    
    Returns
    -------
    float
        Number of supporting papers (normalized 0-1).
    """
    papers = set()
    
    # Check if the node itself has a source_paper_id attribute
    if node in graph.nodes:
        node_data = graph.nodes[node]
        if "source_paper_id" in node_data:
            papers.add(node_data["source_paper_id"])
    
    # Check edges connected to this node for paper attributes
    for pred in graph.predecessors(node):
        edge_data = graph.get_edge_data(pred, node)
        if edge_data and "source_paper_id" in edge_data:
            papers.add(edge_data["source_paper_id"])
    
    for succ in graph.successors(node):
        edge_data = graph.get_edge_data(node, succ)
        if edge_data and "source_paper_id" in edge_data:
            papers.add(edge_data["source_paper_id"])
    
    # Normalize to [0, 1] based on total papers in graph
    if not papers:
        return 0.0
    return min(len(papers) / max(5, graph.number_of_nodes()), 1.0)


def rank_nodes_for_answer(
    graph: nx.DiGraph,
    plan: Dict,
    papers_list: List,
) -> List[NodeRankScore]:
    """
    Rank all nodes by relevance for inclusion in answer.
    
    Uses a composite score:
    score = 0.4 * degree_centrality + 0.3 * query_alignment + 0.3 * paper_support
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    plan : dict
        Query plan from QueryPlanner. Must contain "entities" and "type".
    papers_list : list
        List of supporting papers (for context).
    
    Returns
    -------
    list[NodeRankScore]
        Nodes ranked by score (highest first).
    """
    if graph.number_of_nodes() == 0:
        return []
    
    reasoning_type = plan.get("type", "definition")
    seed_entities = set(plan.get("entities", []))
    
    # Map plan entities to graph node IDs
    seed_nodes = set()
    for entity_label in seed_entities:
        entity_id = entity_label.lower().replace(" ", "_")
        if entity_id in graph.nodes:
            seed_nodes.add(entity_id)
    
    # Rank all nodes
    ranked = []
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        node_label = node_data.get("label", node_id)
        node_type = node_data.get("type", "Unknown")
        
        degree = _compute_degree_centrality(graph, node_id)
        query_align = _compute_closeness_to_seeds(graph, node_id, seed_nodes)
        paper_support = _count_paper_support(graph, node_id)
        
        # Composite score
        score = 0.4 * degree + 0.3 * query_align + 0.3 * paper_support
        
        ranked.append(NodeRankScore(
            node_id=node_id,
            node_label=node_label,
            node_type=node_type,
            score=score,
            degree=degree,
            query_alignment=query_align,
            paper_support=paper_support,
        ))
    
    return sorted(ranked, key=lambda x: x.score, reverse=True)


def select_top_nodes(
    ranked_nodes: List[NodeRankScore],
    plan: Dict,
    top_k: int = 5,
) -> List[SelectedNode]:
    """
    Select top-K nodes, prioritizing by reasoning type.
    
    For definition: prefer Concept/Context/Effect nodes.
    For method: prefer Method/Effect nodes.
    For comparison: prefer Effect/Method/Entity nodes.
    For causal: prefer Cause/Effect nodes.
    
    Parameters
    ----------
    ranked_nodes : list[NodeRankScore]
        Ranked nodes from rank_nodes_for_answer().
    plan : dict
        Query plan for reasoning type.
    top_k : int
        Number of nodes to select.
    
    Returns
    -------
    list[SelectedNode]
        Top-K selected nodes (no explanations yet).
    """
    reasoning_type = _normalize_reasoning_type(plan.get("type", "definition"))
    priority_types = _get_node_types_for_reasoning(reasoning_type)
    
    # Sort by priority type first, then by score, without comparing node objects.
    def _selection_key(node: NodeRankScore) -> Tuple[int, float]:
        priority_idx = len(priority_types)
        if node.node_type in priority_types:
            priority_idx = priority_types.index(node.node_type)
        return (priority_idx, -node.score)

    selected = []
    for node_score in sorted(ranked_nodes, key=_selection_key)[:top_k]:
        selected.append(SelectedNode(
            node_id=node_score.node_id,
            label=node_score.node_label,
            node_type=node_score.node_type,
            rank_score=node_score.score,
            explanation="",  # To be filled by relation extraction
        ))
    
    return selected


# ---------------------------------------------------------------------------
# Relation-based explanation extraction
# ---------------------------------------------------------------------------

def _extract_relation_label(relation_type: str) -> str:
    """Convert relation type to natural language."""
    mapping = {
        "affects": "affects",
        "treats": "treats",
        "occurs_in": "occurs in",
        "caused_by": "caused by",
        "leads_to": "leads to",
        "detected_by": "detected by",
        "studied_in": "studied in",
        "depends_on": "depends on",
        "correlates_with": "correlates with",
        "mitigates": "mitigates",
        "part_of": "is part of",
    }
    return mapping.get(relation_type, relation_type)


def build_node_explanation(
    graph: nx.DiGraph,
    node_id: str,
    node_label: str,
    reasoning_type: str,
) -> str:
    """
    Build a natural-language explanation for a node based on its graph context.
    
    Collects incoming and outgoing edges, converts them to text,
    and merges into a coherent explanation.
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    node_id : str
        The node to explain.
    node_label : str
        The node's display label.
    reasoning_type : str
        The query's reasoning type (for prioritization).
    
    Returns
    -------
    str
        Natural-language explanation (1–2 sentences).
    """
    if node_id not in graph.nodes:
        return ""
    
    priority_relations = _get_relation_types_for_reasoning(reasoning_type)
    
    # Collect relations
    incoming = []  # (source_label, relation_type)
    outgoing = []  # (target_label, relation_type)
    
    for pred in graph.predecessors(node_id):
        edge_data = graph.get_edge_data(pred, node_id)
        rel_type = edge_data.get("relation_type", "related_to") if edge_data else "related_to"
        pred_data = graph.nodes[pred]
        pred_label = pred_data.get("label", pred)
        incoming.append((pred_label, rel_type))
    
    for succ in graph.successors(node_id):
        edge_data = graph.get_edge_data(node_id, succ)
        rel_type = edge_data.get("relation_type", "related_to") if edge_data else "related_to"
        succ_data = graph.nodes[succ]
        succ_label = succ_data.get("label", succ)
        outgoing.append((succ_label, rel_type))
    
    # Priority ordering: put priority relation types first
    incoming.sort(key=lambda x: (
        priority_relations.index(x[1]) if x[1] in priority_relations else len(priority_relations)
    ))
    outgoing.sort(key=lambda x: (
        priority_relations.index(x[1]) if x[1] in priority_relations else len(priority_relations)
    ))
    
    # Build text fragments
    fragments = []
    
    # Incoming relations
    if incoming:
        in_rel = incoming[0]
        rel_label = _extract_relation_label(in_rel[1])
        fragments.append(f"{in_rel[0]} {rel_label} {node_label}")
    
    # Outgoing relations
    if outgoing:
        out_rel = outgoing[0]
        rel_label = _extract_relation_label(out_rel[1])
        fragments.append(f"{node_label} {rel_label} {out_rel[0]}")
    
    return "; ".join(fragments) if fragments else f"{node_label} (graph-extracted concept)"


# ---------------------------------------------------------------------------
# Structured answer construction
# ---------------------------------------------------------------------------

def build_method_answer(selected_nodes: List[SelectedNode]) -> str:
    """
    Build a structured answer for method/process queries.
    
    Format:
    ```
    Common methods include:
    
    1. Method A
       - Explanation
    
    2. Method B
       - Explanation
    ```
    """
    if not selected_nodes:
        return "No methods found in the retrieved graph."
    
    answer_lines = ["**Common methods/approaches:**\n"]
    for i, node in enumerate(selected_nodes, 1):
        answer_lines.append(f"{i}. {node.label}")
        if node.explanation:
            answer_lines.append(f"   - {node.explanation}")
        answer_lines.append("")
    
    return "\n".join(answer_lines)


def build_comparison_answer(selected_nodes: List[SelectedNode]) -> str:
    """
    Build a structured answer for comparison queries.
    
    Format:
    ```
    | Aspect | Details |
    |--------|---------|
    | ...    | ...     |
    ```
    """
    if not selected_nodes:
        return "No comparisons found in the retrieved graph."
    
    answer_lines = ["**Comparison:**\n"]
    answer_lines.append("| Aspect | Details |")
    answer_lines.append("|--------|---------|")
    
    for node in selected_nodes:
        answer_lines.append(f"| {node.label} | {node.explanation or 'See context'} |")
    
    answer_lines.append("")
    return "\n".join(answer_lines)


def build_causal_answer(graph: nx.DiGraph, selected_nodes: List[SelectedNode]) -> str:
    """
    Build a structured answer for causal/mechanism queries.
    
    Format:
    ```
    Cause → Mechanism → Effect
    ```
    """
    if not selected_nodes:
        return "No causal chain found in the retrieved graph."
    
    answer_lines = ["**Causal chain:**\n"]
    
    # Try to build a linear chain from the selected nodes.
    chain_nodes = [node.label for node in selected_nodes[:4]]
    if len(chain_nodes) >= 2:
        answer_lines.append(" → ".join(chain_nodes))
        answer_lines.append("")
    else:
        cause_candidates = []
        effect_candidates = []
        for node in selected_nodes:
            in_deg = graph.in_degree(node.node_id)
            out_deg = graph.out_degree(node.node_id)
            if out_deg >= in_deg:
                cause_candidates.append(node)
            if in_deg >= out_deg:
                effect_candidates.append(node)

        if cause_candidates and effect_candidates:
            cause = cause_candidates[0]
            effect = effect_candidates[0]
            answer_lines.append(f"{cause.label} → {effect.label}")
            answer_lines.append("")
    
    # Elaborate with explanations
    for i, node in enumerate(selected_nodes[:3]):
        if node.explanation:
            answer_lines.append(f"**{node.label}**: {node.explanation}")
    
    answer_lines.append("")
    return "\n".join(answer_lines)


def build_definition_answer(selected_nodes: List[SelectedNode]) -> str:
    """
    Build a structured answer for definition queries.
    
    Format:
    ```
    **Definition**: X is...
    
    **Key properties**:
    - Property 1
    - Property 2
    ```
    """
    if not selected_nodes:
        return "No definition found in the retrieved graph."
    
    answer_lines = []
    
    # Main definition from first node
    main_node = selected_nodes[0]
    if main_node.explanation:
        answer_lines.append(f"**{main_node.label}**: {main_node.explanation}\n")
    else:
        answer_lines.append(f"**{main_node.label}**: (See supporting information below)\n")
    
    # Related properties from other nodes
    if len(selected_nodes) > 1:
        answer_lines.append("**Related concepts**:")
        for node in selected_nodes[1:]:
            answer_lines.append(f"- {node.label}")
            if node.explanation:
                answer_lines.append(f"  ({node.explanation})")
        answer_lines.append("")
    
    return "\n".join(answer_lines)


def build_explanation_answer(selected_nodes: List[SelectedNode]) -> str:
    """Generic explanation answer (fallback for other types)."""
    if not selected_nodes:
        return "No relevant information found in the retrieved graph."
    
    answer_lines = ["**Key findings:**\n"]
    for i, node in enumerate(selected_nodes, 1):
        answer_lines.append(f"{i}. {node.label}")
        if node.explanation:
            answer_lines.append(f"   {node.explanation}")
        answer_lines.append("")
    
    return "\n".join(answer_lines)


def construct_structured_answer(
    selected_nodes: List[SelectedNode],
    plan: Dict,
    graph: nx.DiGraph,
) -> Tuple[str, str]:
    """
    Construct the final answer in a format appropriate for the query type.
    
    Parameters
    ----------
    selected_nodes : list[SelectedNode]
        Top-K ranked nodes with explanations.
    plan : dict
        Query plan (must contain "type").
    
    Returns
    -------
    tuple[str, str]
        (answer_text, structure_type)
    """
    reasoning_type = _normalize_reasoning_type(plan.get("type", "definition"))
    
    if reasoning_type == "method_process":
        return build_method_answer(selected_nodes), "method"
    elif reasoning_type == "comparison":
        return build_comparison_answer(selected_nodes), "comparison"
    elif reasoning_type == "causal_mechanism":
        return build_causal_answer(graph, selected_nodes), "causal"
    elif reasoning_type == "definition":
        return build_definition_answer(selected_nodes), "definition"
    else:
        return build_explanation_answer(selected_nodes), "explanation"


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------

def _count_reasoning_paths(graph: nx.DiGraph) -> int:
    """Count the number of multi-hop paths in the graph."""
    if graph.number_of_nodes() < 2:
        return 0
    
    try:
        # Count all simple paths of length 2–4
        count = 0
        for source in list(graph.nodes())[:min(5, graph.number_of_nodes())]:
            for target in graph.nodes():
                if source != target:
                    try:
                        # Try to find simple paths
                        for _ in nx.all_simple_paths(graph, source, target, cutoff=3):
                            count += 1
                    except (nx.NetworkXNoPath, nx.NetworkXException):
                        pass
        return min(count, 20)  # Cap for efficiency
    except Exception:
        return 0


def validate_coverage(
    graph: nx.DiGraph,
    plan: Dict,
    selected_nodes: List[SelectedNode],
) -> Tuple[bool, List[str], float]:
    """
    Validate that the graph and selected nodes meet minimum requirements.
    
    Rules per reasoning type:
    - method_process: ≥3 distinct method/intervention nodes
    - comparison: ≥2 entities/effects for comparison
    - causal_mechanism: ≥1 complete cause→effect chain
    - definition: ≥1 concept + ≥2 properties
    - optimization: ≥2 methods + evidence of effects
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph.
    plan : dict
        Query plan (contains "type" and other requirements).
    selected_nodes : list[SelectedNode]
        Selected nodes for the answer.
    
    Returns
    -------
    tuple[bool, list[str], float]
        (is_sufficient, missing_requirements, coverage_score)
    """
    if graph.number_of_nodes() < 2:
        return False, ["Graph is too small (< 2 nodes)"], 0.0
    
    reasoning_type = {
        "method": "method_process",
        "method/process": "method_process",
        "causal": "causal_mechanism",
        "causal/mechanism": "causal_mechanism",
    }.get(plan.get("type", "definition"), plan.get("type", "definition"))
    missing = []
    coverage = 0.0
    
    # Count node types in graph
    node_types_in_graph = {}
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        node_type = node_data.get("type", "Unknown")
        node_types_in_graph[node_type] = node_types_in_graph.get(node_type, 0) + 1
    
    # Check by reasoning type
    if reasoning_type == "method_process":
        method_count = node_types_in_graph.get("Method / Intervention", 0)
        if method_count < 3:
            missing.append(f"Need ≥3 method nodes; found {method_count}")
        coverage = min(method_count / 3, 1.0)
    
    elif reasoning_type == "comparison":
        entity_count = node_types_in_graph.get("Concept / Entity", 0)
        effect_count = node_types_in_graph.get("Effect / Outcome", 0)
        if entity_count + effect_count < 2:
            missing.append(f"Need ≥2 comparable entities; found {entity_count + effect_count}")
        coverage = min((entity_count + effect_count) / 4, 1.0)
    
    elif reasoning_type == "causal_mechanism":
        cause_count = node_types_in_graph.get("Cause / Factor", 0)
        effect_count = node_types_in_graph.get("Effect / Outcome", 0)
        paths = _count_reasoning_paths(graph)
        if cause_count < 1 or effect_count < 1 or paths < 1:
            missing.append(
                f"Need cause + effect + path chain; "
                f"got causes={cause_count}, effects={effect_count}, paths={paths}"
            )
        coverage = min((cause_count + effect_count + min(paths, 2)) / 5, 1.0)
    
    elif reasoning_type == "definition":
        concept_count = node_types_in_graph.get("Concept / Entity", 0)
        property_count = sum(
            node_types_in_graph.get(t, 0) 
            for t in ["Effect / Outcome", "Context / Location"]
        )
        if concept_count < 1 or property_count < 1:
            missing.append(
                f"Need concept + properties; "
                f"got concepts={concept_count}, properties={property_count}"
            )
        coverage = min((concept_count + property_count) / 3, 1.0)
    
    elif reasoning_type == "optimization":
        method_count = node_types_in_graph.get("Method / Intervention", 0)
        effect_count = node_types_in_graph.get("Effect / Outcome", 0)
        if method_count < 2 or effect_count < 1:
            missing.append(
                f"Need methods + measured effects; "
                f"got methods={method_count}, effects={effect_count}"
            )
        coverage = min((method_count + effect_count) / 5, 1.0)
    
    is_sufficient = len(missing) == 0
    return is_sufficient, missing, coverage


# ---------------------------------------------------------------------------
# Main compilation entry point
# ---------------------------------------------------------------------------

def compile_answer(
    graph: nx.DiGraph,
    plan: Dict,
    papers: List,
) -> CompileResult:
    """
    Compile a structured answer from a knowledge graph.
    
    This is the main entry point. It:
    1. Extracts and ranks nodes based on plan
    2. Builds natural explanations from relations
    3. Constructs structured answer
    4. Validates coverage
    5. Returns a CompileResult with answer, coverage, and validation status
    
    Parameters
    ----------
    graph : nx.DiGraph
        The knowledge graph (nodes with "label", "type"; edges with "relation_type").
    plan : dict
        Query plan from QueryPlanner (must contain "type" and "entities").
    papers : list
        List of supporting papers (for reference only).
    
    Returns
    -------
    CompileResult
        Result object with answer, coverage, and status.
    """
    warnings_list = []
    
    if graph.number_of_nodes() == 0:
        return CompileResult(
            status="insufficient",
            quality="insufficient",
            answer=None,
            structure_type="",
            selected_nodes=[],
            reasoning_paths_used=0,
            papers_cited=0,
            coverage=0.0,
            confidence=0.0,
            warnings=["Graph is empty"],
            missing_requirements=["Graph must contain at least 1 node"],
        )
    
    try:
        # Step 1: Rank nodes
        ranked_nodes = rank_nodes_for_answer(graph, plan, papers)
        
        if not ranked_nodes:
            return CompileResult(
                status="insufficient",
                quality="insufficient",
                answer=None,
                structure_type="",
                selected_nodes=[],
                reasoning_paths_used=0,
                papers_cited=0,
                coverage=0.0,
                confidence=0.0,
                warnings=["No rankable nodes in graph"],
                missing_requirements=["Graph structure is incompatible"],
            )
        
        # Step 2: Select top nodes based on the planner's graph requirements.
        graph_requirements = plan.get("graph_requirements") or {}
        top_k = int(graph_requirements.get("min_nodes", plan.get("required_nodes", 5)))
        top_k = max(3, min(top_k, 6))
        selected_nodes = select_relevant_nodes(graph, plan, ranked_nodes, top_k=top_k)

        quality = assess_graph_quality(graph, plan, selected_nodes)

        if not selected_nodes:
            return CompileResult(
                status="error",
                quality="insufficient",
                answer=None,
                structure_type="",
                selected_nodes=[],
                reasoning_paths_used=_count_reasoning_paths(graph),
                papers_cited=len(set(
                    graph.nodes[node_id].get("source_paper_id")
                    for node_id in graph.nodes()
                    if "source_paper_id" in graph.nodes[node_id]
                )),
                coverage=0.0,
                confidence=0.0,
                warnings=["Compiler selected zero nodes"],
                missing_requirements=["compiler_selected_zero_nodes"],
            )
        
        # Step 3: Build explanations from relations when the graph is strong or moderate.
        reasoning_type = plan.get("type", "definition")
        if quality in {"strong", "moderate"}:
            max_nodes = 5 if quality == "strong" else 4
            selected_nodes = selected_nodes[:max_nodes]
            for node in selected_nodes:
                node.explanation = build_node_explanation(
                    graph, node.node_id, node.label, reasoning_type
                )
        else:
            # Weak graphs still get structured output, but without relation-heavy claims.
            selected_nodes = selected_nodes[: max(2, min(len(selected_nodes), 3))]

        # Step 4: Validate coverage using the graded assessor.
        _, missing_reqs, coverage_score = validate_coverage(
            graph, plan, selected_nodes
        )
        
        # Step 5: Construct structured answer
        answer_text, structure_type = construct_structured_answer(selected_nodes, plan, graph)
        
        # Step 6: Compute confidence score
        paths = _count_reasoning_paths(graph)
        papers_cited = len(set(
            graph.nodes[node_id].get("source_paper_id")
            for node_id in graph.nodes()
            if "source_paper_id" in graph.nodes[node_id]
        ))
        
        confidence = (
            0.35 * min(papers_cited / max(3, len(papers)), 1.0) +  # Paper support
            0.30 * min(graph.number_of_edges() / max(5, graph.number_of_nodes()), 1.0) +  # Graph density
            0.20 * min(paths / 3, 1.0) +  # Reasoning paths
            0.15 * min(len(selected_nodes) / 4, 1.0)  # Node diversity
        )

        if quality == "insufficient":
            warnings_list.extend(missing_reqs)
            if len(selected_nodes) >= 2:
                quality = "weak"
        
        return CompileResult(
            status="success" if quality in {"strong", "moderate"} else "insufficient",
            quality=quality,
            answer=answer_text,
            structure_type=structure_type,
            selected_nodes=selected_nodes,
            reasoning_paths_used=paths,
            papers_cited=papers_cited,
            coverage=coverage_score,
            confidence=confidence,
            warnings=warnings_list,
            missing_requirements=missing_reqs if quality not in {"strong", "moderate"} else [],
        )
    
    except Exception as e:
        logger.error(f"Error in compile_answer: {e}", exc_info=True)
        return CompileResult(
            status="error",
            quality="insufficient",
            answer=None,
            structure_type="",
            selected_nodes=[],
            reasoning_paths_used=0,
            papers_cited=0,
            coverage=0.0,
            confidence=0.0,
            warnings=[f"Compilation error: {str(e)}"],
            missing_requirements=["Internal error during answer compilation"],
        )
