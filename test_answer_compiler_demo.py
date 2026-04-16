"""
Demo & Testing: Answer Compiler in Action

This script demonstrates how the Graph → Answer Compiler works with
concrete examples of different query types.

Usage:
    python test_answer_compiler_demo.py

Shows:
- Node ranking for different reasoning types
- Coverage validation
- Answer structure selection
- Adaptive retry mechanism
"""

import networkx as nx
from pipeline.answer_compiler import (
    compile_answer,
    rank_nodes_for_answer,
    select_top_nodes,
    build_node_explanation,
    validate_coverage,
    CompileResult,
)
from pipeline.adaptive_retry import plan_retry_actions, format_retry_summary


def build_sample_method_graph():
    """Build a sample graph for a method/process query."""
    graph = nx.DiGraph()
    
    # Nodes: methods and effects
    nodes = [
        ("evaporation", {"label": "Evaporation", "type": "Method / Intervention", "source_paper_id": "paper1"}),
        ("crystallization", {"label": "Crystallization", "type": "Method / Intervention", "source_paper_id": "paper2"}),
        ("precipitation", {"label": "Precipitation", "type": "Method / Intervention", "source_paper_id": "paper1"}),
        ("solvent_extraction", {"label": "Solvent Extraction", "type": "Method / Intervention", "source_paper_id": "paper3"}),
        ("separation", {"label": "Separation", "type": "Effect / Outcome", "source_paper_id": "paper2"}),
        ("purification", {"label": "Purification", "type": "Effect / Outcome", "source_paper_id": "paper1"}),
        ("lithium", {"label": "Lithium", "type": "Concept / Entity", "source_paper_id": "paper1"}),
    ]
    
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    
    # Edges: relations
    edges = [
        ("evaporation", "crystallization", {"relation_type": "leads_to", "source_paper_id": "paper1"}),
        ("crystallization", "separation", {"relation_type": "produces", "source_paper_id": "paper2"}),
        ("precipitation", "purification", {"relation_type": "enables", "source_paper_id": "paper1"}),
        ("solvent_extraction", "separation", {"relation_type": "produces", "source_paper_id": "paper3"}),
        ("separation", "lithium", {"relation_type": "extracts", "source_paper_id": "paper2"}),
    ]
    
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    
    return graph


def build_sample_causal_graph():
    """Build a sample graph for a causal/mechanism query."""
    graph = nx.DiGraph()
    
    nodes = [
        ("drought", {"label": "Drought", "type": "Cause / Factor", "source_paper_id": "paper1"}),
        ("high_temp", {"label": "High Temperature", "type": "Cause / Factor", "source_paper_id": "paper2"}),
        ("soil_degradation", {"label": "Soil Degradation", "type": "Problem / Condition", "source_paper_id": "paper1"}),
        ("crop_stress", {"label": "Crop Stress", "type": "Effect / Outcome", "source_paper_id": "paper2"}),
        ("reduced_yield", {"label": "Reduced Yield", "type": "Effect / Outcome", "source_paper_id": "paper3"}),
        ("osmotic_adjustment", {"label": "Osmotic Adjustment", "type": "Method / Intervention", "source_paper_id": "paper1"}),
    ]
    
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    
    edges = [
        ("drought", "crop_stress", {"relation_type": "causes", "source_paper_id": "paper1"}),
        ("high_temp", "crop_stress", {"relation_type": "causes", "source_paper_id": "paper2"}),
        ("crop_stress", "soil_degradation", {"relation_type": "leads_to", "source_paper_id": "paper2"}),
        ("soil_degradation", "reduced_yield", {"relation_type": "causes", "source_paper_id": "paper3"}),
        ("osmotic_adjustment", "crop_stress", {"relation_type": "mitigates", "source_paper_id": "paper1"}),
    ]
    
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    
    return graph


def build_sample_definition_graph():
    """Build a sample graph for a definition query."""
    graph = nx.DiGraph()
    
    nodes = [
        ("photosynthesis", {"label": "Photosynthesis", "type": "Concept / Entity", "source_paper_id": "paper1"}),
        ("light_reaction", {"label": "Light Reaction", "type": "Method / Intervention", "source_paper_id": "paper1"}),
        ("dark_reaction", {"label": "Dark Reaction (Calvin Cycle)", "type": "Method / Intervention", "source_paper_id": "paper2"}),
        ("chlorophyll", {"label": "Chlorophyll", "type": "Concept / Entity", "source_paper_id": "paper1"}),
        ("carbohydrate", {"label": "Carbohydrate", "type": "Effect / Outcome", "source_paper_id": "paper2"}),
        ("oxygen", {"label": "Oxygen Release", "type": "Effect / Outcome", "source_paper_id": "paper1"}),
    ]
    
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    
    edges = [
        ("photosynthesis", "light_reaction", {"relation_type": "part_of", "source_paper_id": "paper1"}),
        ("photosynthesis", "dark_reaction", {"relation_type": "part_of", "source_paper_id": "paper2"}),
        ("light_reaction", "oxygen", {"relation_type": "produces", "source_paper_id": "paper1"}),
        ("dark_reaction", "carbohydrate", {"relation_type": "produces", "source_paper_id": "paper2"}),
        ("chlorophyll", "photosynthesis", {"relation_type": "enables", "source_paper_id": "paper1"}),
    ]
    
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    
    return graph


def demo_method_query():
    """Demonstrate answer compilation for a method query."""
    print("\n" + "="*70)
    print("DEMO 1: METHOD/PROCESS QUERY")
    print("="*70)
    
    query = "How is lithium extracted from brine?"
    plan = {
        "type": "method_process",
        "entities": ["lithium", "brine"],
        "complexity": "medium",
        "required_nodes": 5,
    }
    papers = [{"paper_id": f"paper{i}", "title": f"Paper {i}"} for i in range(1, 4)]
    
    graph = build_sample_method_graph()
    
    print(f"\n📋 Query: {query}")
    print(f"📋 Plan type: {plan['type']}")
    print(f"📋 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Show node ranking
    print("\n🔍 Node Ranking:")
    ranked = rank_nodes_for_answer(graph, plan, papers)
    for i, node in enumerate(ranked[:5], 1):
        print(f"  {i}. {node.node_label:20s} (type={node.node_type:20s}, score={node.score:.3f})")
        print(f"     degree={node.degree:.2f}, align={node.query_alignment:.2f}, support={node.paper_support:.2f}")
    
    # Compile answer
    print("\n✍️ Compiling answer...")
    result = compile_answer(graph, plan, papers)
    
    print(f"\n📊 Compilation Result:")
    print(f"  Status: {result.status}")
    print(f"  Coverage: {result.coverage:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Selected nodes: {len(result.selected_nodes)}")
    print(f"  Reasoning paths: {result.reasoning_paths_used}")
    print(f"  Papers cited: {result.papers_cited}")
    
    print(f"\n📄 Generated Answer:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    
    if result.missing_requirements:
        print(f"\n⚠️  Missing requirements: {result.missing_requirements}")
    
    return result


def demo_causal_query():
    """Demonstrate answer compilation for a causal query."""
    print("\n" + "="*70)
    print("DEMO 2: CAUSAL/MECHANISM QUERY")
    print("="*70)
    
    query = "Why do droughts reduce crop yields?"
    plan = {
        "type": "causal_mechanism",
        "entities": ["drought", "crop_yield"],
        "complexity": "medium",
        "required_nodes": 5,
    }
    papers = [{"paper_id": f"paper{i}", "title": f"Paper {i}"} for i in range(1, 4)]
    
    graph = build_sample_causal_graph()
    
    print(f"\n📋 Query: {query}")
    print(f"📋 Plan type: {plan['type']}")
    print(f"📋 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Compile answer
    print("\n✍️ Compiling answer...")
    result = compile_answer(graph, plan, papers)
    
    print(f"\n📊 Compilation Result:")
    print(f"  Status: {result.status}")
    print(f"  Coverage: {result.coverage:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Structure: {result.structure_type}")
    
    print(f"\n📄 Generated Answer:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    
    return result


def demo_definition_query():
    """Demonstrate answer compilation for a definition query."""
    print("\n" + "="*70)
    print("DEMO 3: DEFINITION QUERY")
    print("="*70)
    
    query = "What is photosynthesis?"
    plan = {
        "type": "definition",
        "entities": ["photosynthesis"],
        "complexity": "low",
        "required_nodes": 4,
    }
    papers = [{"paper_id": f"paper{i}", "title": f"Paper {i}"} for i in range(1, 3)]
    
    graph = build_sample_definition_graph()
    
    print(f"\n📋 Query: {query}")
    print(f"📋 Plan type: {plan['type']}")
    print(f"📋 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Compile answer
    print("\n✍️ Compiling answer...")
    result = compile_answer(graph, plan, papers)
    
    print(f"\n📊 Compilation Result:")
    print(f"  Status: {result.status}")
    print(f"  Coverage: {result.coverage:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")
    
    print(f"\n📄 Generated Answer:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    
    return result


def demo_sparse_graph_retry():
    """Demonstrate adaptive retry on sparse graph."""
    print("\n" + "="*70)
    print("DEMO 4: ADAPTIVE RETRY (Sparse Graph)")
    print("="*70)
    
    # Create sparse graph (only 1 method node - insufficient)
    graph = nx.DiGraph()
    graph.add_node("evaporation", label="Evaporation", type="Method / Intervention", source_paper_id="paper1")
    graph.add_node("lithium", label="Lithium", type="Concept / Entity", source_paper_id="paper1")
    
    query = "How is lithium extracted?"
    plan = {
        "type": "method_process",
        "entities": ["lithium"],
        "complexity": "low",
        "required_nodes": 5,
    }
    papers = [{"paper_id": "paper1", "title": "Paper 1"}]
    
    print(f"\n📋 Query: {query}")
    print(f"📋 Initial graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # First compilation (will fail)
    result_1 = compile_answer(graph, plan, papers)
    print(f"\n🔴 First attempt:")
    print(f"  Status: {result_1.status}")
    print(f"  Coverage: {result_1.coverage:.1%}")
    print(f"  Missing: {result_1.missing_requirements}")
    
    # Plan retry
    if result_1.status == "insufficient":
        retry_plan = plan_retry_actions(result_1, query, plan)
        print(f"\n♻️  Retry Plan:")
        print(f"  Should retry: {retry_plan.get('should_retry')}")
        print(f"  Expanded query: {retry_plan.get('retrieval_query')}")
        print(f"  Missing types to prioritize: {retry_plan.get('entity_type_priorities')}")
        print(f"  Reason: {retry_plan.get('reason')}")
    
    return result_1


def main():
    """Run all demos."""
    print("\n" + "█" * 70)
    print("█ GRAPH → ANSWER COMPILER: DEMO & TESTING")
    print("█" * 70)
    
    results = []
    
    try:
        results.append(("Method Query", demo_method_query()))
    except Exception as e:
        print(f"❌ Error in method demo: {e}")
    
    try:
        results.append(("Causal Query", demo_causal_query()))
    except Exception as e:
        print(f"❌ Error in causal demo: {e}")
    
    try:
        results.append(("Definition Query", demo_definition_query()))
    except Exception as e:
        print(f"❌ Error in definition demo: {e}")
    
    try:
        results.append(("Sparse Graph Retry", demo_sparse_graph_retry()))
    except Exception as e:
        print(f"❌ Error in retry demo: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    success_count = sum(1 for _, r in results if r.status == "success")
    insufficient_count = sum(1 for _, r in results if r.status == "insufficient")
    print(f"✓ Successful compilations: {success_count}")
    print(f"⚠️  Insufficient coverage: {insufficient_count}")
    print(f"📊 Average coverage: {sum(r.coverage for _, r in results) / len(results):.1%}")
    print(f"📊 Average confidence: {sum(r.confidence for _, r in results) / len(results):.1%}")
    
    print("\n✅ All demos completed!\n")


if __name__ == "__main__":
    main()
