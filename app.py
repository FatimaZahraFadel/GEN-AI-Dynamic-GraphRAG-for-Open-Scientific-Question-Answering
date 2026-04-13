"""
Dynamic GraphRAG Explorer — Streamlit dashboard.

Interactive scientific QA with retrieval, graph reasoning, and grounded generation.
"""

import os
import re
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


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dynamic GraphRAG Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — white theme with subtle depth ──────────────────────────────

_CUSTOM_CSS = """
<style>
/* ── Global background & typography ────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1a1a2e;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #e8e8e8;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #1a1a2e !important;
}

/* ── Card mixin ────────────────────────────────────────────────── */
.card {
    background: #ffffff;
    border: 1px solid #eaeaea;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.07); }

/* ── Section heading ───────────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin: 0 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f0f0f0;
    letter-spacing: -0.01em;
}

/* ── Metric strip ──────────────────────────────────────────────── */
.metric-strip {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-item {
    flex: 1 1 0;
    min-width: 120px;
    background: #f8f9fa;
    border: 1px solid #eaeaea;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.metric-item .metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}
.metric-item .metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* ── Timing bar ────────────────────────────────────────────────── */
.timing-bar {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 0.6rem 0;
}
.timing-chip {
    font-size: 0.73rem;
    font-weight: 500;
    color: #4b5563;
    background: #f3f4f6;
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    white-space: nowrap;
}
.timing-chip strong { color: #1a1a2e; }

/* ── Paper card ────────────────────────────────────────────────── */
.paper-card {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    transition: background 0.15s;
}
.paper-card:hover { background: #f5f5f5; }
.paper-card .paper-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1a1a2e;
    margin-bottom: 0.3rem;
}
.paper-card .paper-meta {
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 0.35rem;
}
.paper-card .paper-abstract {
    font-size: 0.8rem;
    color: #374151;
    line-height: 1.5;
}

/* ── Legend chip ────────────────────────────────────────────────── */
.legend-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 0.75rem 0; }
.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 500;
    color: #374151;
    background: #f8f9fa;
    border: 1px solid #eaeaea;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Path badge ────────────────────────────────────────────────── */
.path-badge {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #1f2937;
}
.path-badge .path-arrow { color: #9ca3af; font-weight: 600; }

/* ── Answer block ──────────────────────────────────────────────── */
.answer-block {
    background: #f8fafb;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #1e293b;
}

/* ── Empty state ───────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    color: #9ca3af;
}
.empty-state .empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-state .empty-text { font-size: 0.95rem; }

/* ── Streamlit overrides ───────────────────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stExpander"] {
    border: 1px solid #eaeaea !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
div.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #f0f0f0;
}
div.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.6rem 1.1rem;
    border-radius: 8px 8px 0 0;
    background: #fff5f5;
    color: #b91c1c;
    border: 1px solid #fecaca;
    margin-right: 0.25rem;
}
div.stTabs [data-baseweb="tab"]:hover {
    background: #fee2e2;
    color: #991b1b;
}
div.stTabs [aria-selected="true"] {
    background: #dc2626 !important;
    color: #ffffff !important;
    border-color: #dc2626 !important;
}

/* Ensure Subgraph/Full Graph radio labels remain visible on white background */
[data-baseweb="radio"] {
    color: #111111 !important;
}
[data-baseweb="radio"] span {
    color: #111111 !important;
}

header[data-testid="stHeader"] { background: transparent; }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────────────────────

NODE_COLORS = {
    "Concept / Entity": "#2E7D32",
    "Problem / Condition": "#C62828",
    "Method / Intervention": "#1565C0",
    "Context / Location": "#6A1B9A",
    "Cause / Factor": "#EF6C00",
    "Effect / Outcome": "#00838F",
    "Unknown": "#546E7A",
}

_DEFAULT_QUESTION = "What fungal diseases affect wheat crops in humid environments?"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int = 260) -> str:
    if not text:
        return "No abstract available."
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def _html(raw: str) -> None:
    """Shorthand for unsafe HTML markdown."""
    st.markdown(raw, unsafe_allow_html=True)


# ── Data / pipeline functions ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_retrieve_papers(question: str, domain: str, top_n: int) -> List[dict]:
    papers = PaperRetriever(top_n=top_n).retrieve(question, domain)
    return [p.to_dict() for p in papers]


@st.cache_data(show_spinner=False)
def cached_filter_papers(question: str, paper_dicts: List[dict], top_k: int) -> List[dict]:
    papers = [Paper.from_dict(p) for p in paper_dicts]
    filtered = PaperFilter().filter(papers, question, top_k=top_k)
    return [p.to_dict() for p in filtered]


def run_full_pipeline(question: str, top_n: int, top_k: int) -> Dict:
    """Execute the full Dynamic GraphRAG pipeline and package results."""
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

    # Rebuild entities/relations snapshot from session extraction cache.
    entities, relations = [], []
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


# ── Graph helpers ────────────────────────────────────────────────────────────

def extract_reasoning_paths(
    subgraph: nx.DiGraph, seed_ids: List[str], limit: int = 6,
) -> List[List[str]]:
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


def build_node_support_index(
    graph: nx.DiGraph, papers: List[Paper],
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


# ── Render: Sidebar ──────────────────────────────────────────────────────────

def render_sidebar() -> Tuple[str, bool]:
    """Draw the sidebar with question input and run button. Returns (question, run_clicked)."""
    with st.sidebar:
        _html('<p style="font-size:1.4rem;font-weight:700;margin-bottom:0.2rem;">Dynamic GraphRAG</p>')
        _html('<p style="font-size:0.78rem;color:#6b7280;margin-bottom:1.5rem;">'
              'Scientific QA with retrieval, graph reasoning & grounded generation</p>')

        _html('<p class="section-title">Query</p>')
        question = st.text_area(
            "Ask a scientific question",
            value=_DEFAULT_QUESTION,
            height=100,
            label_visibility="collapsed",
            placeholder="Enter a scientific question...",
        )

        run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

        st.divider()

        # Legend
        _html('<p class="section-title">Node Type Legend</p>')
        legend_html = '<div class="legend-row">'
        for label, color in NODE_COLORS.items():
            legend_html += (
                f'<span class="legend-chip">'
                f'<span class="legend-dot" style="background:{color};"></span>{label}</span>'
            )
        legend_html += "</div>"
        _html(legend_html)

    return question, run_clicked


# ── Render: Metrics strip ────────────────────────────────────────────────────

def render_metrics_strip(results: Dict) -> None:
    """Top-level metrics rendered as styled HTML cards."""
    timings = results.get("timings", {})
    items = [
        (results["domain"], "Domain"),
        (str(len(results["retrieved_papers"])), "Retrieved"),
        (str(len(results["filtered_papers"])), "Filtered"),
        (str(results["graph"].number_of_nodes()), "Graph Nodes"),
        (str(results["retrieval"]["subgraph"].number_of_edges()), "Graph Edges"),
        (f'{timings.get("total", 0.0):.1f}s', "Total Time"),
    ]
    cards = ""
    for value, label in items:
        cards += (
            f'<div class="metric-item">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>'
        )
    _html(f'<div class="metric-strip">{cards}</div>')


# ── Render: Timing chips ────────────────────────────────────────────────────

def render_timing_chips(timings: Dict) -> None:
    stages = [
        ("Domain", "domain_detection"),
        ("Retrieval", "retrieval"),
        ("Filtering", "filtering"),
        ("Extraction", "entity_extraction"),
        ("Graph Build", "graph_build"),
        ("Graph Retrieval", "graph_retrieval"),
        ("Answer Gen", "answer_generation"),
    ]
    chips = ""
    for label, key in stages:
        val = timings.get(key, 0.0)
        chips += f'<span class="timing-chip">{label} <strong>{val:.1f}s</strong></span>'
    _html(f'<div class="timing-bar">{chips}</div>')


# ── Render: Papers ───────────────────────────────────────────────────────────

def render_papers_list(papers: List[Paper], show_relevance: bool = False) -> None:
    """Render a list of papers as styled cards."""
    if not papers:
        _html('<div class="empty-state"><div class="empty-text">No papers available.</div></div>')
        return

    for idx, paper in enumerate(papers, 1):
        year = paper.year if paper.year is not None else "n.d."
        source = paper.source or "N/A"
        cites = paper.citation_count or 0
        meta_parts = [f"{year}", f"{source}", f"{cites} citations"]
        if show_relevance and paper.relevance_score:
            meta_parts.append(f"relevance {paper.relevance_score:.3f}")
        meta = " &middot; ".join(meta_parts)
        abstract = _truncate(paper.abstract, 350)

        _html(
            f'<div class="paper-card">'
            f'<div class="paper-title">{idx}. {paper.title}</div>'
            f'<div class="paper-meta">{meta}</div>'
            f'<div class="paper-abstract">{abstract}</div>'
            f'</div>'
        )


# ── Render: Knowledge graph (pyvis) ─────────────────────────────────────────

def render_graph_pyvis(
    graph: nx.DiGraph,
    seed_ids: List[str],
    reasoning_paths: List[List[str]],
) -> None:
    if graph.number_of_nodes() == 0:
        _html('<div class="empty-state">'
              '<div class="empty-icon">&#x26A0;</div>'
              '<div class="empty-text">Graph is empty for this query.</div></div>')
        return

    highlighted_nodes = set(seed_ids)
    for p in reasoning_paths:
        highlighted_nodes.update(p)

    net = Network(
        height="650px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1F2937",
        directed=True,
    )
    net.barnes_hut(
        gravity=-20000, central_gravity=0.2,
        spring_length=140, spring_strength=0.06,
    )

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
            f"Seed: {'Yes' if is_seed else 'No'}"
        )

        net.add_node(
            node_id, label=label, title=title, color=color,
            size=base_size, borderWidth=border_width, shape="dot",
        )

    for src, dst, attrs in graph.edges(data=True):
        relation = attrs.get("relation_type", "related_to")
        net.add_edge(src, dst, label=relation, title=relation, arrows="to")

    net.set_options("""
    {
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": true, "keyboard": true},
      "physics": {"enabled": true, "stabilization": {"enabled": true, "iterations": 200}},
      "edges": {"smooth": false, "font": {"size": 11, "align": "top"}, "color": {"color": "#b0bec5"}},
      "nodes": {"font": {"size": 13, "face": "Tahoma"}}
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=670, scrolling=True)


# ── Render: Reasoning paths ─────────────────────────────────────────────────

def render_reasoning_paths(
    paths: List[List[str]], subgraph: nx.DiGraph,
) -> None:
    if not paths:
        st.caption("No multi-hop reasoning paths found for this query.")
        return

    for path in paths:
        labels = [subgraph.nodes[n].get("label", n) for n in path if n in subgraph.nodes]
        arrow_text = f' <span class="path-arrow">&rarr;</span> '.join(labels)
        _html(f'<div class="path-badge">{arrow_text}</div>')


# ── Render: Node explorer ───────────────────────────────────────────────────

def render_node_explorer(
    graph: nx.DiGraph, papers: List[Paper], seed_ids: List[str],
) -> None:
    if graph.number_of_nodes() == 0:
        st.caption("No nodes available for inspection.")
        return

    support_index = build_node_support_index(graph, papers)

    node_options: List[Tuple[str, str]] = []
    for nid, attrs in graph.nodes(data=True):
        label = attrs.get("label", nid)
        node_options.append((f"{label} [{nid}]", nid))
    node_options.sort(key=lambda x: x[0].lower())

    selected_display = st.selectbox(
        "Select a node to inspect",
        options=[n[0] for n in node_options],
        index=0,
        label_visibility="collapsed",
    )
    selected_node = dict(node_options)[selected_display]

    attrs = graph.nodes[selected_node]
    label = attrs.get("label", selected_node)
    node_type = attrs.get("type", "Unknown")
    is_seed = selected_node in seed_ids
    neighbors = sorted(
        set(graph.successors(selected_node)).union(set(graph.predecessors(selected_node)))
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Node", label)
    c2.metric("Type", node_type)
    c3.metric("Connections", len(neighbors))
    c4.metric("Seed Node", "Yes" if is_seed else "No")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Connected Nodes**")
        if neighbors:
            for n in neighbors:
                n_label = graph.nodes[n].get("label", n)
                st.markdown(f"- {n_label}")
        else:
            st.caption("No connected nodes.")

    with col_right:
        st.markdown("**Supporting Papers**")
        node_papers = support_index.get(selected_node, [])
        if node_papers:
            for p in node_papers[:8]:
                st.markdown(f"- {p.title} ({p.year if p.year else 'n.d.'})")
        else:
            st.caption("No directly linked papers found.")


# ── Render: Answer ───────────────────────────────────────────────────────────

def render_answer(answer_dict: Dict) -> None:
    answer = answer_dict.get("answer", "")
    if not answer:
        _html('<div class="empty-state">'
              '<div class="empty-icon">&#x2753;</div>'
              '<div class="empty-text">No answer was generated.</div></div>')
        return

    _html(f'<div class="answer-block">{answer}</div>')

    # Metadata row
    model = answer_dict.get("model", "N/A")
    n_papers = answer_dict.get("num_papers_used", 0)
    quality = "Low" if answer_dict.get("low_confidence") else "Normal"
    _html(
        f'<div style="display:flex;gap:1.5rem;margin-top:0.75rem;font-size:0.78rem;color:#6b7280;">'
        f'<span>Model: <strong>{model}</strong></span>'
        f'<span>Papers used: <strong>{n_papers}</strong></span>'
        f'<span>Evidence quality: <strong>{quality}</strong></span>'
        f'</div>'
    )


# ── Render: Provenance ──────────────────────────────────────────────────────

def render_provenance(
    answer_dict: Dict,
    filtered_papers: List[Paper],
    subgraph: nx.DiGraph,
    seed_ids: List[str],
    reasoning_paths: List[List[str]],
) -> None:
    answer = answer_dict.get("answer", "")

    # Cited papers
    cited_titles = re.findall(r"\[Paper:\s*([^\]]+)\]", answer, flags=re.IGNORECASE)

    with st.expander("Papers cited in answer", expanded=True):
        if not cited_titles:
            st.caption("No explicit [Paper: ...] citations found in the answer.")
        else:
            for title in sorted(set(cited_titles)):
                matched = next(
                    (p for p in filtered_papers if title.strip().lower()[:60] in p.title.lower()[:60]),
                    None,
                )
                if matched:
                    year = matched.year if matched.year else "n.d."
                    st.markdown(f"- **{matched.title}** ({year}) - *{matched.source or 'N/A'}*")
                else:
                    st.markdown(f"- {title} *(not matched in retrieved papers)*")

    # Papers supporting seed nodes
    with st.expander("Papers supporting key graph nodes", expanded=False):
        if subgraph.number_of_nodes() == 0 or not filtered_papers:
            st.caption("No graph data available.")
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
                        year = p.year if p.year else "n.d."
                        st.markdown(f"  - {p.title} ({year})")

    # Reasoning paths
    with st.expander("Graph reasoning paths used", expanded=False):
        if not reasoning_paths:
            st.caption("No multi-hop reasoning paths were identified.")
        else:
            for i, path in enumerate(reasoning_paths, 1):
                path_labels = [subgraph.nodes[n].get("label", n) for n in path if n in subgraph.nodes]
                st.markdown(f"{i}. " + " -> ".join(path_labels))


# ── Render: Extraction debug ────────────────────────────────────────────────

def render_extraction_debug(entities: list, relations: list) -> None:
    with st.expander("Extracted entities & relations", expanded=False):
        col_e, col_r = st.columns(2)
        with col_e:
            st.markdown(f"**Entities** ({len(entities)})")
            if entities:
                for entity in sorted(entities, key=lambda e: (e.type, e.label))[:20]:
                    st.markdown(f"- `{entity.type}` {entity.label}")
            else:
                st.caption("No entities extracted.")
        with col_r:
            st.markdown(f"**Relations** ({len(relations)})")
            if relations:
                for rel in sorted(relations, key=lambda r: r.relation_type)[:15]:
                    st.markdown(f"- {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}")
            else:
                st.caption("No relations extracted.")


# ── State management ─────────────────────────────────────────────────────────

def ensure_state() -> None:
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "pipeline_session" not in st.session_state:
        st.session_state["pipeline_session"] = SessionState()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_state()

    # ── Sidebar ──────────────────────────────────────────────────────
    question, run_clicked = render_sidebar()

    # ── Run pipeline ─────────────────────────────────────────────────
    if run_clicked:
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Running Dynamic GraphRAG pipeline..."):
            try:
                st.session_state["results"] = run_full_pipeline(
                    question.strip(), top_n=25, top_k=8,
                )
            except Exception as exc:
                st.session_state["results"] = None
                st.exception(exc)
                st.stop()

    results = st.session_state.get("results")

    # ── Empty state ──────────────────────────────────────────────────
    if not results:
        _html(
            '<div class="empty-state" style="padding:5rem 2rem;">'
            '<div class="empty-icon">&#x1F52C;</div>'
            '<div class="empty-text">Enter a scientific question and click '
            '<strong>Run Pipeline</strong> to explore papers, knowledge graphs, '
            'and generated answers.</div></div>'
        )
        return

    # ── Results ──────────────────────────────────────────────────────

    # Metrics strip
    render_metrics_strip(results)

    # Timing chips
    render_timing_chips(results.get("timings", {}))

    # Prepare graph data
    retrieval = results["retrieval"]
    subgraph = retrieval["subgraph"]
    full_graph = results["graph"]
    seed_ids = retrieval.get("seed_entities", [])
    reasoning_paths = extract_reasoning_paths(subgraph, seed_ids)

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_answer, tab_graph, tab_papers, tab_debug = st.tabs([
        "Answer", "Knowledge Graph", "Papers", "Debug",
    ])

    # ── Tab: Answer ──────────────────────────────────────────────────
    with tab_answer:
        _html('<p class="section-title">Generated Answer</p>')
        render_answer(results["answer_dict"])

        st.markdown("")
        _html('<p class="section-title">Provenance</p>')
        render_provenance(
            answer_dict=results["answer_dict"],
            filtered_papers=results["filtered_papers"],
            subgraph=subgraph,
            seed_ids=seed_ids,
            reasoning_paths=reasoning_paths,
        )

    # ── Tab: Knowledge Graph ─────────────────────────────────────────
    with tab_graph:
        # Graph view toggle + info
        col_toggle, col_info = st.columns([1, 2])
        with col_toggle:
            graph_view = st.radio(
                "View",
                options=["Subgraph", "Full Graph"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with col_info:
            if full_graph.number_of_nodes() > subgraph.number_of_nodes():
                st.caption(
                    f"Full graph: {full_graph.number_of_nodes()} nodes / "
                    f"Retrieved subgraph: {subgraph.number_of_nodes()} nodes"
                )

        graph_to_render = subgraph if graph_view == "Subgraph" else full_graph

        with st.spinner("Rendering graph..."):
            render_graph_pyvis(graph_to_render, seed_ids, reasoning_paths)

        # Reasoning paths & node explorer side by side
        col_paths, col_explorer = st.columns(2)

        with col_paths:
            _html('<p class="section-title">Reasoning Paths</p>')
            render_reasoning_paths(reasoning_paths, subgraph)

        with col_explorer:
            _html('<p class="section-title">Node Explorer</p>')
            render_node_explorer(subgraph, results["filtered_papers"], seed_ids)

    # ── Tab: Papers ──────────────────────────────────────────────────
    with tab_papers:
        col_ret, col_filt = st.columns(2)
        with col_ret:
            _html('<p class="section-title">Retrieved Papers</p>')
            render_papers_list(results["retrieved_papers"])
        with col_filt:
            _html('<p class="section-title">Filtered Papers</p>')
            render_papers_list(results["filtered_papers"], show_relevance=True)

    # ── Tab: Debug ───────────────────────────────────────────────────
    with tab_debug:
        _html('<p class="section-title">Extraction Debug</p>')
        render_extraction_debug(results.get("entities", []), results.get("relations", []))


if __name__ == "__main__":
    main()
