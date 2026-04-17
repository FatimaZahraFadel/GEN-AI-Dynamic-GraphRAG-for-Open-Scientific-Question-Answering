# Dynamic GraphRAG for Open Scientific Question Answering

Dynamic GraphRAG is an end-to-end scientific QA system that retrieves live papers,
builds a query-specific knowledge graph, retrieves a focused subgraph, and generates
grounded answers with confidence and coverage signals.

## Quick Start

```bash
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

## What Is Implemented

- Plan-driven pipeline with `QueryPlanner` (reasoning type, entities, extraction/retrieval priorities)
- Robust domain detection (`DomainDetector`) with consensus strategies
- Multi-source paper retrieval from Europe PMC, arXiv, and OpenAlex
- LLM query expansion + dual retrieval (original query + expanded query)
- Plan-aware reranking using:
  - semantic similarity (`alpha=0.6`)
  - plan keyword overlap (`beta=0.25`)
  - metadata quality (citation + recency, `gamma=0.15`)
- Hybrid paper filtering with semantic, lexical, metadata, and intent alignment
- Domain-agnostic entity/relation extraction with multi-tier fallback
- Query-aware graph construction and subgraph retrieval with reasoning paths
- Confidence/coverage scoring with adaptive expansion loop (up to 2 iterations)
- LLM answer generation with low-confidence handling
- Graph-only Answer Compiler module + demo script
- Streamlit app for interactive exploration and graph visualization
- Evaluation harness with 4 ablation modes

## Pipeline Overview

```
User Query
    |
    v
0) QueryPlanner        -> reasoning type + entities + graph requirements
    |
    v
1) DomainDetector      -> domain + subdomain
    |
    v
2) PaperRetriever      -> Europe PMC + arXiv + OpenAlex
                        -> query expansion + dual retrieval + dedup + rerank
    |
    v
3) PaperFilter         -> hybrid ranking + consistency assessment
    |
    v
4) EntityExtractor     -> typed entities/relations (Groq with fallback)
    |
    v
5) GraphBuilder        -> query-focused directed knowledge graph
    |
    v
6) GraphRetriever      -> relevant subgraph + reasoning paths + validation
    |
    +--> (if low confidence / low coverage) Adaptive Retry (max 2 iters)
    |
    v
7) AnswerGenerator     -> grounded final answer
    |
    v
Confidence/Coverage + Metrics + Artifacts
```

## Repository Structure

```
.
├── app.py                           # Streamlit UI (interactive GraphRAG explorer)
├── main.py                          # Pipeline orchestration and run_pipeline()
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py                  # Runtime defaults and feature toggles
├── pipeline/
│   ├── query_planner.py             # Stage 0
│   ├── domain_detector.py           # Stage 1
│   ├── paper_retriever.py           # Stage 2
│   ├── paper_filter.py              # Stage 3
│   ├── entity_extractor.py          # Stage 4
│   ├── graph_builder.py             # Stage 5
│   ├── graph_retriever.py           # Stage 6
│   ├── answer_generator.py          # Stage 7
│   ├── answer_compiler.py           # Graph-only compiler / validator module
│   ├── adaptive_retry.py            # Confidence-driven expansion helpers
│   ├── confidence_scorer.py         # Confidence and coverage scoring
│   └── embedding_service.py         # Shared embedding service/cache
├── models/
│   ├── paper.py
│   └── graph_node.py
├── utils/
│   ├── logger.py
│   ├── helpers.py
│   └── session_state.py             # Session-level caching/reuse
├── evaluation/
│   └── evaluator.py                 # Ablation + metrics evaluation harness
├── docs/
│   ├── report.tex
│   └── report.pdf
└── test_answer_compiler_demo.py     # Answer compiler demonstrations
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy environment template:

```bash
# Windows PowerShell
Copy-Item .env.example .env
# macOS/Linux
# cp .env.example .env
```

3. Fill required values in `.env`.

## Environment Variables

### Required

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used by LLM stages (planner, retrieval expansion, extraction, generation, evaluation judge) |

### Optional

| Variable | Description |
|---|---|
| `OPENALEX_EMAIL` | Contact email for OpenAlex polite-pool requests |
| `USE_OLLAMA_EXTRACTION_FALLBACK` | `1` to allow extraction fallback to local Ollama |
| `OLLAMA_BASE_URL` | Ollama endpoint (default `http://localhost:11434`) |
| `OLLAMA_EXTRACTION_MODEL` | Local extraction model name |
| `USE_OLLAMA_PRIMARY` | `1` to route planner/retriever/answer/eval calls to Ollama first |
| `OLLAMA_GENERAL_MODEL` | Local model for general LLM stages |
| `OLLAMA_TIMEOUT_SECONDS` | Timeout for Ollama requests |
| `ENTITY_EXTRACTION_MODEL` | Override extraction model |
| `ENTITY_EXTRACTION_FAST_MODEL` | Override fast extraction model |
| `QUERY_PLANNER_MODEL` | Override planner model |
| `QUERY_PLANNER_MAX_TOKENS` | Planner token budget |
| `QUERY_PLANNER_TEMPERATURE` | Planner temperature |
| `PLAN_AWARE_RETRIEVAL_ALPHA` | Semantic similarity weight (default `0.6`) |
| `PLAN_AWARE_RETRIEVAL_BETA` | Plan keyword overlap weight (default `0.25`) |
| `PLAN_AWARE_RETRIEVAL_GAMMA` | Metadata quality weight (default `0.15`) |

Note: `.env.example` currently includes `SEMANTIC_SCHOLAR_API_KEY` for legacy compatibility, but the active retriever implementation uses Europe PMC, arXiv, and OpenAlex.

## Running

### 1) Run pipeline from CLI

```bash
python main.py
```

### 2) Run Streamlit app

```bash
streamlit run app.py
```

### 3) Run Answer Compiler demo

```bash
python test_answer_compiler_demo.py
```

### Runtime Expectations

- Full pipeline runs are typically slower than classic RAG because retrieval,
    extraction, graph construction, and validation all run at inference time.
- Typical end-to-end latency is around 15 to 45 seconds, depending on query
    complexity and external API responsiveness.

## Evaluation

The evaluation harness compares these modes:

- `classic_rag`
- `graphrag_base`
- `graphrag_reasoning_only`
- `graphrag_full`

It reports metrics including ROUGE-1/ROUGE-2, keyword coverage, semantic similarity,
answer length, and optional LLM-judge scoring.

Use `evaluation/evaluator.py` to run benchmark evaluations.

## Core Configuration (`config/settings.py`)

Most runtime behavior is controlled in `config/settings.py`, including:

- domain list (`DOMAINS`)
- retrieval/filter limits (`TOP_N_PAPERS`, `FAST_MODE_TOP_N_PAPERS`, `FAST_MODE_TOP_K_PAPERS`)
- extraction concurrency and model controls
- graph confidence threshold + expansion limits
- planner budgets and temperatures
- plan-aware retrieval weights (`PLAN_AWARE_RETRIEVAL_ALPHA/BETA/GAMMA`)

## Notes

- The system is domain-agnostic by design, with current domain detectors tuned for:
  Agriculture, Geoscience, Computer Science, Supply Chain, and Environment.
- Session-aware caching is implemented (`utils/session_state.py`) to speed up follow-up queries.
- Report artifacts are available in `docs/report.tex` and `docs/report.pdf`.
- Known limitations and tradeoffs are documented in `docs/report.tex`
    (see the "Limitations" section).
