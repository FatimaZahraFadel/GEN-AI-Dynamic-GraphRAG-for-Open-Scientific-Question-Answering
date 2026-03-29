# Dynamic GraphRAG for Open Scientific Question Answering

A pipeline that answers scientific questions by dynamically building a knowledge
graph from freshly retrieved academic papers and using it as retrieval context for
an LLM.

## Pipeline Overview

```
User Query
    │
    ▼
1. DomainDetector    — classify query into scientific domain(s)
    │
    ▼
2. PaperRetriever    — fetch papers from Semantic Scholar & OpenAlex
    │
    ▼
3. PaperFilter       — embed & rank papers by semantic similarity
    │
    ▼
4. EntityExtractor   — extract concepts, methods, datasets via LLM
    │
    ▼
5. GraphBuilder      — build knowledge graph (papers + entities + edges)
    │
    ▼
6. GraphRetriever    — retrieve query-relevant subgraph
    │
    ▼
7. AnswerGenerator   — generate grounded answer via RAG prompt
    │
    ▼
  Answer
```

## Setup

1. **Clone / download** the project.

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables** — copy `.env.example` to `.env` and fill
   in your API keys:

   ```bash
   cp .env.example .env
   ```

   | Variable | Description |
   |---|---|
   | `OPENAI_API_KEY` | OpenAI API key for LLM calls |
   | `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key (optional but recommended) |
   | `OPENALEX_EMAIL` | Your email for OpenAlex polite-pool access |

4. **Run the pipeline**:

   ```bash
   python main.py
   ```

## Project Structure

```
dynamic-graphrag/
├── main.py                   # Entry point & pipeline orchestration
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py           # Global constants (domains, limits, model names)
├── pipeline/
│   ├── domain_detector.py    # Stage 1 — query classification
│   ├── paper_retriever.py    # Stage 2 — paper retrieval from APIs
│   ├── paper_filter.py       # Stage 3 — embedding-based filtering
│   ├── entity_extractor.py   # Stage 4 — LLM entity extraction
│   ├── graph_builder.py      # Stage 5 — knowledge graph construction
│   ├── graph_retriever.py    # Stage 6 — subgraph retrieval
│   └── answer_generator.py   # Stage 7 — RAG answer generation
├── models/
│   ├── paper.py              # Paper dataclass
│   └── graph_node.py         # GraphNode dataclass
├── utils/
│   ├── logger.py             # Logging utilities
│   └── helpers.py            # General helpers (chunking, similarity, etc.)
└── evaluation/
    └── evaluator.py          # ROUGE / BERTScore / exact-match / LLM-judge
```

## Configuration

Edit `config/settings.py` to adjust:

- `DOMAINS` — list of supported scientific domains
- `TOP_N_PAPERS` — papers retained after filtering
- `MAX_GRAPH_NODES` — node cap for the knowledge graph
- `EMBEDDING_MODEL` — sentence-transformers model name

## Evaluation

Pass a JSON dataset of `{"question": "...", "answer": "..."}` records to
`run_evaluation()` in `main.py`, or call the `Evaluator` class directly.
