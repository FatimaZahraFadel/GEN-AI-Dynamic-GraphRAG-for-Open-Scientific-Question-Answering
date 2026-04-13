"""
Adaptive Retry Mechanism: expand queries when graph coverage is insufficient.

When coverage validation fails, this module:
1. Analyzes missing requirements
2. Expands the query with missing node type keywords
3. Re-runs retrieval, extraction, and graph building
4. Rebuilds and recompiles the answer
5. Limits retries to max 2 attempts

This ensures answers improve progressively without infinite loops.
"""

import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_missing_entity_types(missing_requirements: List[str]) -> List[str]:
    """
    Parse missing requirements to extract entity types we need.
    
    Examples:
    - "Need ≥3 method nodes; found 1" → ["Method / Intervention"]
    - "Need cause + effect + path chain..." → ["Cause / Factor", "Effect / Outcome"]
    
    Parameters
    ----------
    missing_requirements : list[str]
        Missing requirements from coverage validation.
    
    Returns
    -------
    list[str]
        Entity types that should be prioritized in the expanded query.
    """
    missing_text = " ".join(missing_requirements).lower()
    
    entity_keywords = {
        "method": "Method / Intervention",
        "techniques": "Method / Intervention",
        "procedures": "Method / Intervention",
        "process": "Method / Intervention",
        "cause": "Cause / Factor",
        "effect": "Effect / Outcome",
        "outcome": "Effect / Outcome",
        "entity": "Concept / Entity",
        "concept": "Concept / Entity",
        "comparison": "Concept / Entity",
        "context": "Context / Location",
        "mechanism": "Cause / Factor",
        "problem": "Problem / Condition",
        "condition": "Problem / Condition",
    }
    
    found_types = set()
    for keyword, entity_type in entity_keywords.items():
        if keyword in missing_text:
            found_types.add(entity_type)
    
    return list(found_types) if found_types else ["Concept / Entity", "Method / Intervention"]


def expand_query_with_context(
    original_query: str,
    missing_requirements: List[str],
    plan: Dict,
) -> str:
    """
    Expand the query with additional context focused on missing node types.
    
    strategy:
    - Extract missing entity types
    - Add descriptors like "additional methods", "more causes", "alternatives"
    
    Example:
    Original: "How is lithium extracted?"
    Missing: need ≥3 methods
    Expanded: "How is lithium extracted? What are the alternative methods, techniques, and procedures?"
    
    Parameters
    ----------
    original_query : str
        Original user query.
    missing_requirements : list[str]
        Missing requirements from coverage validation.
    plan : dict
        Query plan.
    
    Returns
    -------
    str
        Expanded query text.
    """
    missing_types = _extract_missing_entity_types(missing_requirements)
    reasoning_type = plan.get("type", "definition")
    
    expansions = []
    
    # Build expansions based on reasoning type and missing types
    if reasoning_type == "method_process":
        if "Method / Intervention" in missing_types:
            expansions.extend([
                "What are alternative methods?",
                "What are additional techniques?",
                "What are different procedures?",
                "What are other approaches?",
            ])
    
    elif reasoning_type == "comparison":
        if "Concept / Entity" in missing_types or "Effect / Outcome" in missing_types:
            expansions.extend([
                "What are related concepts?",
                "What are measurable outcomes?",
                "What are different aspects?",
            ])
    
    elif reasoning_type == "causal_mechanism":
        if "Cause / Factor" in missing_types:
            expansions.append("What are the underlying causes?")
        if "Effect / Outcome" in missing_types:
            expansions.append("What are the resulting effects?")
        if "Cause / Factor" in missing_types and "Effect / Outcome" in missing_types:
            expansions.append("What is the causal chain?")
    
    elif reasoning_type == "definition":
        if "Concept / Entity" in missing_types:
            expansions.extend([
                "What is this concept?",
                "What are key components?",
            ])
        if "Context / Location" in missing_types or "Effect / Outcome" in missing_types:
            expansions.extend([
                "What are its properties?",
                "What are its characteristics?",
                "What are its applications?",
            ])
    
    elif reasoning_type == "optimization":
        if "Method / Intervention" in missing_types:
            expansions.append("What are better methods?")
        if "Effect / Outcome" in missing_types:
            expansions.append("What improvements are possible?")
    
    # Build expanded query
    expanded = original_query
    if expansions:
        expanded += " " + " ".join(expansions[:2])  # Add top 2 expansions
    
    return expanded


def build_retry_retrieval_query(
    original_query: str,
    missing_requirements: List[str],
) -> str:
    """
    Build a focused retrieval query to fill gaps.
    
    This is used for the retrieval retry phase — it emphasizes
    missing node types and relevant dimensions.
    
    Parameters
    ----------
    original_query : str
        Original user query.
    missing_requirements : list[str]
        Missing requirements from coverage validation.
    
    Returns
    -------
    str
        Retrieval query focused on filling gaps.
    """
    return expand_query_with_context(original_query, missing_requirements, {})


def should_retry(
    compile_result,
    max_retries: int = 2,
    current_retry_count: int = 0,
) -> bool:
    """
    Decide whether to trigger adaptive retry.
    
    Retry if:
    - Status is "insufficient"
    - It's within max retry limit
    - There are specific missing requirements
    
    Parameters
    ----------
    compile_result : CompileResult
        Result from compile_answer().
    max_retries : int
        Maximum number of retries.
    current_retry_count : int
        Current retry attempt number.
    
    Returns
    -------
    bool
        Whether to retry.
    """
    if current_retry_count >= max_retries:
        return False
    
    if compile_result.status != "insufficient":
        return False
    
    if not compile_result.missing_requirements:
        return False
    
    return True


def plan_retry_actions(
    compile_result,
    original_query: str,
    plan: Dict,
) -> Dict:
    """
    Plan the retry sequence given failed coverage validation.
    
    Returns:
    {
        "should_retry": bool,
        "retrieval_query": str,
        "entity_type_priorities": list[str],
        "reason": str,
    }
    
    Parameters
    ----------
    compile_result : CompileResult
        Result from compile_answer().
    original_query : str
        Original user query.
    plan : dict
        Current query plan.
    
    Returns
    -------
    dict
        Retry plan with actions and parameters.
    """
    if compile_result.status != "insufficient":
        return {
            "should_retry": False,
            "reason": "Answer coverage is sufficient",
        }
    
    missing = compile_result.missing_requirements
    missing_types = _extract_missing_entity_types(missing)
    expanded_query = expand_query_with_context(original_query, missing, plan)
    
    return {
        "should_retry": True,
        "retrieval_query": expanded_query,
        "entity_type_priorities": missing_types,
        "reason": f"Insufficient coverage: {missing[0] if missing else 'unknown'}",
        "retry_hints": [
            f"Try focusing on: {', '.join(missing_types)}",
            f"Missing: {missing[0]}",
        ],
    }


def merge_graphs(
    original_graph: nx.DiGraph,
    new_graph: nx.DiGraph,
) -> nx.DiGraph:
    """
    Merge two graphs intelligently, avoiding duplicates.
    
    Nodes are merged by ID; edges are preserved if not duplicate.
    
    Parameters
    ----------
    original_graph : nx.DiGraph
        The original graph.
    new_graph : nx.DiGraph
        The newly built graph from retry.
    
    Returns
    -------
    nx.DiGraph
        Merged graph.
    """
    merged = original_graph.copy()
    
    # Add new nodes
    for node_id in new_graph.nodes():
        if node_id not in merged.nodes():
            merged.add_node(node_id, **new_graph.nodes[node_id])
    
    # Add new edges
    for source, target in new_graph.edges():
        if not merged.has_edge(source, target):
            merged.add_edge(source, target, **new_graph.get_edge_data(source, target))
    
    return merged


def format_retry_summary(
    retry_count: int,
    original_coverage: float,
    new_coverage: float,
    original_node_count: int,
    new_node_count: int,
) -> str:
    """
    Format a summary of the retry outcome.
    
    Parameters
    ----------
    retry_count : int
        Retry attempt number.
    original_coverage : float
        Original coverage score.
    new_coverage : float
        New coverage score after retry.
    original_node_count : int
        Original graph node count.
    new_node_count : int
        New graph node count after retry.
    
    Returns
    -------
    str
        Formatted summary.
    """
    coverage_improvement = (new_coverage - original_coverage) * 100
    node_growth = new_node_count - original_node_count
    
    return (
        f"Retry #{retry_count}: Coverage improved from {original_coverage:.1%} to {new_coverage:.1%} "
        f"(+{coverage_improvement:.0f}%). Graph grew from {original_node_count} to {new_node_count} nodes "
        f"(+{node_growth} new)."
    )
