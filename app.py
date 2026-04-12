import os
import tempfile
import time
from typing import Dict, List, Tuple

import networkx as nx
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network

from main import run_pipeline
from models.paper import Paper
from pipeline.paper_filter import PaperFilter
from pipeline.paper_retriever import PaperRetriever
from utils.session_state import SessionState

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(
    page_title="Dynamic GraphRAG Explorer",
    page_icon="🧠",
    layout="wide",
)

NODE_COLORS = {
    "Concept / Entity": "#2E7D32",
    "Problem / Condition": "#C62828",
    "Method / Intervention": "#1565C0",
    "Context / Location": "#6A1B9A",
    "Cause / Factor": "#EF6C00",
    "Effect / Outcome": "#00838F",
    "Unknown": "#546E7A",
}


@st.cache_data(show_spinner=False)
def cached_retrieve_papers(question: str, domain: str, top_n: int) -> List[dict]:
    papers = PaperRetriever(top_n=top_n).retrieve(question, domain)
    return [p.to_dict() for p in papers]


@st.cache_data(show_spinner=False)
def cached_filter_papers(question: str, paper_dicts: List[dict], top_k: int) -> List[dict]:
    papers = [Paper.from_dict(p) for p in paper_dicts]
    filtered = PaperFilter().filter(papers, question, top_k=top_k)
    return [p.to_dict() for p in filtered]


def truncate_text(text: str, max_len: int = 260) -> str:
    if not text:
        return "No abstract available."
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def run_full_pipeline(question: str, top_n: int, top_k: int) -> Dict:
    t0 = time.perf_counter()
    details = run_pipeline(
        query=question,
        fast_mode=True,
        session_state=st.session_state["pipeline_session"],
        return_details=True,
        top_n_override=top_n,
        top_k_override=top_k,
    )
    total_elapsed = time.perf_counter() - t0

    retrieved_papers = details.get("retrieved_papers", [])

    # Rebuild entities/relations snapshot from session extraction cache for debug display.
    entities = []
    relations = []
    for e_list, r_list in st.session_state["pipeline_session"].extraction_cache.values():
        entities.extend(e_list)
        relations.extend(r_list)

    timings = {
        "total": total_elapsed,
        "domain_detection": details.get("runtime_metrics", {}).get("domain_detection_seconds", 0.0),
        "retrieval": details.get("runtime_metrics", {}).get("retrieval_seconds", 0.0),
        "filtering": details.get("runtime_metrics", {}).get("filtering_seconds", 0.0),
        "entity_extraction": details.get("runtime_metrics", {}).get("entity_extraction_seconds", 0.0),
        "graph_build": details.get("runtime_metrics", {}).get("graph_build_seconds", 0.0),
        "graph_retrieval": details.get("runtime_metrics", {}).get("graph_retrieval_seconds", 0.0),
        "answer_generation": details.get("runtime_metrics", {}).get("answer_generation_seconds", 0.0),
        "graph_expansion_iters": details.get("expansion_iters", 0),
        "retrieval_skips": details.get("retrieval_skips", 0),
    }

    return {
        "question": question,
        "domain": details["domain"],
        "retrieved_papers": retrieved_papers,
        "filtered_papers": details["filtered_papers"],
        "entities": entities,
        "relations": relations,
        "graph": details["graph"],
        "retrieval": details["retrieval"],
        "answer_dict": details["answer_dict"],
        "timings": timings,
    }


def extract_reasoning_paths(subgraph: nx.DiGraph, seed_ids: List[str], limit: int = 6) -> List[List[str]]:
    if subgraph.number_of_nodes() == 0:
        return []

    undirected = subgraph.to_undirected()
    paths: List[List[str]] = []

    unique_seeds = [s for s in seed_ids if s in undirected]
    if len(unique_seeds) >= 2:
        anchor = unique_seeds[0]
        for target in unique_seeds[1:]:
            if len(paths) >= limit:
                break
            if nx.has_path(undirected, anchor, target):
                p = nx.shortest_path(undirected, source=anchor, target=target)
                if 2 <= len(p) <= 6:
                    paths.append(p)

    if not paths:
        for src, dst in subgraph.edges():
            if len(paths) >= limit:
                break
            paths.append([src, dst])

    return paths[:limit]


def path_to_label_text(path: List[str], graph: nx.DiGraph) -> str:
    labels = [graph.nodes[n].get("label", n) for n in path]
    return " -> ".join(labels)


def build_node_support_index(
    graph: nx.DiGraph,
    papers: List[Paper],
) -> Dict[str, List[Paper]]:
    by_id = {p.paper_id: p for p in papers}
    support: Dict[str, List[Paper]] = {nid: [] for nid in graph.nodes()}

    for nid, attrs in graph.nodes(data=True):
        pid = attrs.get("source_paper_id")
        if pid and pid in by_id:
            support[nid].append(by_id[pid])

    for src, dst, attrs in graph.edges(data=True):
        pid = attrs.get("source_paper_id")
        if pid and pid in by_id:
            paper = by_id[pid]
            if paper not in support[src]:
                support[src].append(paper)
            if paper not in support[dst]:
                support[dst].append(paper)

    return support


def render_papers_section(papers: List[Paper], title: str) -> None:
    st.subheader(title)
    if not papers:
        st.info("No papers available.")
        return

    for idx, paper in enumerate(papers, start=1):
        year = paper.year if paper.year is not None else "Unknown"
        header = f"{idx}. {paper.title} ({year})"
        with st.expander(header, expanded=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**Source:** {paper.source or 'N/A'}")
            c2.markdown(f"**Citations:** {paper.citation_count}")
            score = f"{paper.relevance_score:.4f}" if paper.relevance_score else "N/A"
            c3.markdown(f"**Relevance:** {score}")
            st.write(truncate_text(paper.abstract, max_len=420))


def render_graph_pyvis(
    graph: nx.DiGraph,
    seed_ids: List[str],
    reasoning_paths: List[List[str]],
) -> None:
    if graph.number_of_nodes() == 0:
        st.warning("Graph is empty for this query.")
        return

    highlighted_nodes = set(seed_ids)
    for p in reasoning_paths:
        highlighted_nodes.update(p)

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1F2937",
        directed=True,
    )
    net.barnes_hut(gravity=-20000, central_gravity=0.2, spring_length=140, spring_strength=0.06)

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", node_id)
        node_type = attrs.get("type", "Unknown")
        degree = int(graph.degree(node_id))

        base_size = 14 + min(degree * 2, 20)
        is_seed = node_id in seed_ids
        is_reasoning = node_id in highlighted_nodes

        color = NODE_COLORS.get(node_type, NODE_COLORS["Unknown"])
        border_width = 4 if is_seed else (2 if is_reasoning else 1)

        title = (
            f"<b>{label}</b><br>"
            f"Type: {node_type}<br>"
            f"Node ID: {node_id}<br>"
            f"Degree: {degree}<br>"
            f"Seed Node: {'Yes' if is_seed else 'No'}"
        )

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=base_size,
            borderWidth=border_width,
            shape="dot",
        )

    for src, dst, attrs in graph.edges(data=True):
        relation = attrs.get("relation_type", "related_to")
        net.add_edge(src, dst, label=relation, title=relation, arrows="to")

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          },
          "edges": {
            "smooth": false,
            "font": {
              "size": 11,
              "align": "top"
            },
            "color": {
              "color": "#90A4AE"
            }
          },
          "nodes": {
            "font": {
              "size": 13,
              "face": "Tahoma"
            }
          }
        }
        """
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=720, scrolling=True)


def render_node_explorer(
    graph: nx.DiGraph,
    papers: List[Paper],
    seed_ids: List[str],
) -> None:
    st.subheader("Node Explorer")
    if graph.number_of_nodes() == 0:
        st.info("No nodes available for inspection.")
        return

    support_index = build_node_support_index(graph, papers)

    node_options: List[Tuple[str, str]] = []
    for nid, attrs in graph.nodes(data=True):
        label = attrs.get("label", nid)
        node_options.append((f"{label} [{nid}]", nid))
    node_options.sort(key=lambda x: x[0].lower())

    selected_display = st.selectbox(
        "Select a node",
        options=[n[0] for n in node_options],
        index=0,
    )
    selected_node = dict(node_options)[selected_display]

    attrs = graph.nodes[selected_node]
    label = attrs.get("label", selected_node)
    node_type = attrs.get("type", "Unknown")

    is_seed = selected_node in seed_ids
    neighbors = sorted(
        set(graph.successors(selected_node)).union(set(graph.predecessors(selected_node)))
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Node", label)
    c2.metric("Type", node_type)
    c3.metric("Connected Nodes", len(neighbors))

    st.markdown(f"**Seed node:** {'Yes' if is_seed else 'No'}")

    st.markdown("**Connected Nodes**")
    if neighbors:
        for n in neighbors:
            n_label = graph.nodes[n].get("label", n)
            st.markdown(f"- {n_label} ({n})")
    else:
        st.write("No connected nodes.")

    st.markdown("**Supporting Papers**")
    node_papers = support_index.get(selected_node, [])
    if node_papers:
        for p in node_papers[:8]:
            st.markdown(f"- {p.title} ({p.year if p.year else 'Unknown'})")
    else:
        st.write("No directly linked supporting papers found.")


def render_answer(answer_dict: Dict) -> None:
    st.subheader("Generated Answer")
    answer = answer_dict.get("answer", "")
    if not answer:
        st.warning("No answer generated.")
        return

    st.markdown(answer)
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Model: {answer_dict.get('model', 'N/A')}")
    c2.caption(f"Papers used: {answer_dict.get('num_papers_used', 0)}")
    if answer_dict.get("low_confidence"):
        c3.caption("Evidence quality: Low")
    else:
        c3.caption("Evidence quality: Normal")


def render_provenance(
    answer_dict: Dict,
    filtered_papers: List[Paper],
    subgraph: nx.DiGraph,
    seed_ids: List[str],
    reasoning_paths: List[List[str]],
) -> None:
    """Render an answer provenance panel: cited papers and contributing graph paths."""
    st.subheader("Answer Provenance")

    answer = answer_dict.get("answer", "")

    # --- Cited papers ---
    import re as _re
    cited_titles = _re.findall(r"\[Paper:\s*([^\]]+)\]", answer, flags=_re.IGNORECASE)
    cited_titles_lower = {t.strip().lower()[:60] for t in cited_titles}

    with st.expander("Papers cited in answer", expanded=True):
        if not cited_titles:
            st.info("No explicit [Paper: ...] citations found in the answer.")
        else:
            for title in sorted(set(cited_titles)):
                # Try to match cited title against filtered papers
                matched = next(
                    (p for p in filtered_papers if title.strip().lower()[:60] in p.title.lower()[:60]),
                    None,
                )
                if matched:
                    year = matched.year if matched.year else "Unknown"
                    st.markdown(f"- **{matched.title}** ({year}) — *{matched.source or 'N/A'}*")
                else:
                    st.markdown(f"- {title} *(not matched in retrieved papers)*")

    # --- Papers supporting seed nodes ---
    with st.expander("Papers supporting key graph nodes", expanded=False):
        if subgraph.number_of_nodes() == 0 or not filtered_papers:
            st.info("No graph data available.")
        else:
            support_index = build_node_support_index(subgraph, filtered_papers)
            seen_titles: set = set()
            for nid in seed_ids[:8]:
                if nid not in subgraph.nodes:
                    continue
                label = subgraph.nodes[nid].get("label", nid)
                papers_for_node = support_index.get(nid, [])
                if not papers_for_node:
                    continue
                st.markdown(f"**{label}**")
                for p in papers_for_node[:3]:
                    if p.title not in seen_titles:
                        seen_titles.add(p.title)
                        year = p.year if p.year else "Unknown"
                        st.markdown(f"  - {p.title} ({year})")

    # --- Reasoning path summary ---
    with st.expander("Graph reasoning paths used", expanded=False):
        if not reasoning_paths:
            st.info("No multi-hop reasoning paths were identified.")
        else:
            for i, path in enumerate(reasoning_paths, 1):
                path_labels = [subgraph.nodes[n].get("label", n) for n in path if n in subgraph.nodes]
                st.markdown(f"{i}. " + " → ".join(path_labels))


def ensure_state() -> None:
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "pipeline_session" not in st.session_state:
        st.session_state["pipeline_session"] = SessionState()


def main() -> None:
    ensure_state()

    st.title("Dynamic GraphRAG Explorer")
    st.caption("Interactive scientific QA with retrieval, graph reasoning, and grounded generation")

    with st.container(border=True):
        st.markdown("### Question Input")
        default_q = "What fungal diseases affect wheat crops in humid environments?"
        question = st.text_input("Ask a scientific question", value=default_q)
        c_cfg1, c_cfg2 = st.columns(2)
        top_n = c_cfg1.slider("Retrieve papers", min_value=8, max_value=30, value=25, step=2)
        top_k = c_cfg2.slider("Papers for extraction", min_value=2, max_value=12, value=8, step=1)
        run_clicked = st.button("Run Query", type="primary")

    if run_clicked:
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Running Dynamic GraphRAG pipeline..."):
            try:
                st.session_state["results"] = run_full_pipeline(
                    question.strip(),
                    top_n=top_n,
                    top_k=top_k,
                )
                st.success("Pipeline completed successfully.")
            except Exception as exc:
                st.session_state["results"] = None
                st.exception(exc)
                st.stop()

    results = st.session_state.get("results")
    if not results:
        st.info("Run a query to see papers, graph reasoning, and generated answer.")
        return

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domain", results["domain"])
    c2.metric("Retrieved Papers", len(results["retrieved_papers"]))
    c3.metric("Filtered Papers", len(results["filtered_papers"]))
    c4.metric("Graph Nodes", results["graph"].number_of_nodes())

    timings = results.get("timings", {})
    if timings:
        st.caption(
            "Run time: "
            f"{timings.get('total', 0.0):.1f}s total | "
            f"retrieval {timings.get('retrieval', 0.0):.1f}s | "
            f"filtering {timings.get('filtering', 0.0):.1f}s | "
            f"entity extraction {timings.get('entity_extraction', 0.0):.1f}s | "
            f"graph {timings.get('graph_build', 0.0):.1f}s+{timings.get('graph_retrieval', 0.0):.1f}s | "
            f"answer generation {timings.get('answer_generation', 0.0):.1f}s"
        )

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        render_papers_section(results["retrieved_papers"], "Top Retrieved Papers")
    with right:
        render_papers_section(results["filtered_papers"], "Top Filtered Papers")

    st.markdown("---")
    st.subheader("Extraction Debug Info")
    with st.expander("Show extracted entities & relations by paper", expanded=False):
        entities = results.get("entities", [])
        relations = results.get("relations", [])
        filtered_papers = results.get("filtered_papers", [])
        
        if entities:
            st.markdown(f"**Total: {len(entities)} unique entities**")
            for entity in sorted(entities, key=lambda e: (e.type, e.label))[:20]:
                st.markdown(f"- [{entity.type}] {entity.label}")
        else:
            st.warning("No entities were extracted. Try increasing paper count or checking paper quality.")
        
        if relations:
            st.markdown(f"**Total: {len(relations)} relations**")
            for rel in sorted(relations, key=lambda r: r.relation_type)[:15]:
                st.markdown(f"- {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}")
        else:
            st.info("No relations extracted yet. More entities needed for relations.")

    st.markdown("---")
    st.subheader("Knowledge Graph")

    retrieval = results["retrieval"]
    subgraph = retrieval["subgraph"]
    full_graph = results["graph"]
    seed_ids = retrieval.get("seed_entities", [])
    reasoning_paths = extract_reasoning_paths(subgraph, seed_ids)

    if full_graph.number_of_nodes() > subgraph.number_of_nodes():
        st.info(
            "You are seeing the retrieved subgraph, which can be much smaller than the full graph. "
            f"Full graph has {full_graph.number_of_nodes()} nodes; retrieved subgraph has {subgraph.number_of_nodes()}."
        )

    stats_a, stats_b, stats_c = st.columns(3)
    stats_a.metric("Subgraph Nodes", retrieval.get("num_nodes", 0))
    stats_b.metric("Subgraph Edges", retrieval.get("num_edges", 0))
    stats_c.metric("Seed Nodes", len(seed_ids))

    graph_view = st.radio(
        "Graph View",
        options=["Retrieved Subgraph", "Full Graph"],
        horizontal=True,
    )
    graph_to_render = subgraph if graph_view == "Retrieved Subgraph" else full_graph

    render_graph_pyvis(graph_to_render, seed_ids, reasoning_paths)

    st.markdown("---")
    st.subheader("Reasoning Paths")
    if reasoning_paths:
        for path in reasoning_paths:
            st.markdown(f"- {path_to_label_text(path, subgraph)}")
    else:
        st.write("No explicit multi-hop paths were found for this query.")

    st.markdown("---")
    render_node_explorer(subgraph, results["filtered_papers"], seed_ids)

    st.markdown("---")
    render_answer(results["answer_dict"])

    st.markdown("---")
    render_provenance(
        answer_dict=results["answer_dict"],
        filtered_papers=results["filtered_papers"],
        subgraph=subgraph,
        seed_ids=seed_ids,
        reasoning_paths=reasoning_paths,
    )


if __name__ == "__main__":
    main()
