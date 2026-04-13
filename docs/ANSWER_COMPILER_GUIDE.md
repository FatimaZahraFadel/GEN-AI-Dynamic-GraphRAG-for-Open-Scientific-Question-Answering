# Graph → Answer Compiler: Implementation Guide

## Overview

This document describes the **Graph → Answer Compiler** upgrade to the Dynamic GraphRAG system. The compiler transforms knowledge graphs into structured, reasoning-based answers with automatic coverage validation—**without** falling back to paper summaries.

---

## 🎯 Key Principles

### 1. **Graph-Only Answers**
- No paper summarization fallback
- Answers derived purely from graph structure, node types, and relations
- Explicit rejection if coverage is insufficient

### 2. **Domain-Agnostic Design**
- No hardcoded keyword lists ("lithium", "fertilizer", etc.)
- Reasoning based on generic entity types (`Method / Intervention`, `Cause / Factor`, etc.)
- Dynamic ranking via graph centrality and semantic similarity

### 3. **Coverage-Gated Answering**
- Before finalizing, validates that graph meets minimum requirements
- If insufficient, triggers adaptive retry with expanded queries
- Max 2 retries per query to avoid infinite loops

### 4. **Plan-Driven Answer Structure**
- Query plan (reasoning type) determines node selection priorities
- Output format (method list, comparison table, causal chain, definition, etc.) is auto-selected
- Natural language explanations built from graph relations

---

## 📦 New Modules

### 1. `pipeline/answer_compiler.py`

Main entry point: `compile_answer(graph, plan, papers)`

#### Core Functions:

**Node Ranking & Selection**
- `rank_nodes_for_answer()` — Rank nodes by:
  - Degree centrality (40%)
  - Query alignment / closeness to seed entities (30%)
  - Paper support count (30%)
- `select_top_nodes()` — Priority-based selection:
  - For `method`: prioritize "Method / Intervention" nodes
  - For `comparison`: prioritize "Effect / Outcome", "Concept / Entity"
  - For `causal`: prioritize "Cause / Factor", "Effect / Outcome"
  - etc.

**Explanation Extraction**
- `build_node_explanation()` — Convert graph edges to natural language:
  - E.g., `solvent_extraction --[extracts]--> lithium` becomes:
    > "Solvent extraction is used to extract lithium using chemical solvents."

**Structured Answer Construction**
- `build_method_answer()` — Format: numbered methods with explanations
- `build_comparison_answer()` — Format: Markdown table
- `build_causal_answer()` — Format: cause → mechanism → effect chain
- `build_definition_answer()` — Format: definition + related concepts
- `build_explanation_answer()` — Generic fallback format

**Coverage Validation**
- `validate_coverage()` — Check if graph meets minimum requirements per reasoning type:
  - `method_process`: ≥ 3 distinct Method nodes
  - `comparison`: ≥ 2 comparable entities
  - `causal_mechanism`: ≥ 1 cause + 1 effect + ≥ 1 reasoning path
  - `definition`: ≥ 1 concept + ≥ 2 properties
  - `optimization`: ≥ 2 methods + ≥ 1 effect

#### Return Type: `CompileResult`
```python
@dataclass
class CompileResult:
    status: str                    # "success", "insufficient", "error"
    answer: Optional[str]          # Final answer text
    structure_type: str            # "method", "comparison", "causal", etc.
    selected_nodes: List[SelectedNode]  # Nodes used
    reasoning_paths_used: int      # Count of multi-hop paths
    papers_cited: int              # Unique papers supporting the answer
    coverage: float                # Coverage score [0, 1]
    confidence: float              # Confidence score [0, 1]
    warnings: List[str]            # Issues during compilation
    missing_requirements: List[str] # Why validation failed (if status="insufficient")
```

### 2. `pipeline/adaptive_retry.py`

Adaptive retry mechanism when coverage validation fails.

#### Core Functions:

**Retry Planning**
- `plan_retry_actions()` — Analyzes missing requirements and generates:
  - Expanded query with focus on missing entity types
  - Entity type priorities for the retry
  - Reason/explanation for the retry
- `expand_query_with_context()` — Contextually expands query:
  - If missing methods: add "alternative methods", "techniques", "procedures"
  - If missing causes: add "underlying causes", "causal chain"
  - If missing effects: add "resulting effects", "outcomes"

**Graph Merging**
- `merge_graphs()` — Intelligently combine original and retry graphs:
  - Preserves nodes from both graphs
  - Avoids duplicate edges
  - Maintains edge metadata

**Formatted Summaries**
- `format_retry_summary()` — Logs retry outcomes:
  - Coverage improvement percentage
  - Node count growth
  - Coverage before/after

---

## 🔄 Integration into Main Pipeline

### Stage 7: Answer Generation (Modified)

**Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COMPILE ANSWER (Graph → Answer Compiler)                │
│    Input: subgraph, plan, papers                           │
│    Output: CompileResult with answer + coverage            │
└─────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │ Coverage validation success? │
         └───────────────────────────────┘
           ✓ YES          │          ✗ NO
           │              └─────────────────────────┐
           ↓                                        ↓
    ┌──────────────────┐          ┌──────────────────────────────┐
    │ Return compiled  │          │ 2. PLAN RETRY                │
    │ answer          │          │    ✓ Retry count < 2?        │
    │                 │          │    ✓ Missing requirements?    │
    │ ✓ NO fallback   │          │    ✓ Enable expansion?       │
    │ ✓ Only graph    │          │                              │
    └──────────────────┘          └──────────────────────────────┘
                                         │
                                    ✗ Conditions not met
                                    │ Fallback to LLM
                                    ↓
                              ┌──────────────────┐
                              │ LLM Answer Gen   │
                              │ (fallback only)  │
                              └──────────────────┘

                                    ✓ Conditions met
                                    ↓
                        ┌──────────────────────────────┐
                        │ 3. RETRY RETRIEVAL           │
                        │    - Expanded query          │
                        │    - Entity type priorities  │
                        └──────────────────────────────┘
                                    ↓
                        ┌──────────────────────────────┐
                        │ 4. EXTRACT + REBUILD GRAPH   │
                        │    - More targeted extraction│
                        │    - Merge with orig graph   │
                        └──────────────────────────────┘
                                    ↓
                        ┌──────────────────────────────┐
                        │ 5. RECOMPILE ANSWER          │
                        │    - Coverage check again    │
                        │    - If still insufficient   │
                        │      → another retry (max 2) │
                        └──────────────────────────────┘
```

### Key Changes in `main.py`:

1. **Import new modules:**
   ```python
   from pipeline.answer_compiler import compile_answer, CompileResult
   from pipeline.adaptive_retry import plan_retry_actions, merge_graphs, format_retry_summary
   ```

2. **Stage 7 orchestration:**
   - Call `compile_answer()` first
   - Check `compile_result.status`
   - If `"insufficient"` and retries available:
     - Re-retrieve with expanded query
     - Re-extract with entity type priorities
     - Merge graphs
     - Recompile answer
   - If `"insufficient"` after retries or status `"error"`:
     - Fall back to LLM generator **only as last resort**
   - If `"success"`:
     - Use compiled answer (no LLM needed)

3. **Updated result payload:**
   ```python
   answer_dict = {
       "answer": compile_result.answer,
       "compiler_used": True,  # Track which system generated answer
       "compiler_status": compile_result.status,
       "structure_type": compile_result.structure_type,
       "selected_nodes": [...],
       "reasoning_paths": compile_result.reasoning_paths_used,
       "papers_cited": compile_result.papers_cited,
       "coverage": compile_result.coverage,
       "confidence": compile_result.confidence,
   }
   ```

---

## 📊 Example: Method Query

### Input:
```
Query: "How is lithium extracted from brine?"
Plan type: "method_process"
Entities: ["lithium", "brine"]
Expected node types: ["Method / Intervention", "Effect / Outcome", ...]
```

### Graph Nodes (from extraction):
```
evaporation (Method) → degree=3, papers=2
precipitation (Method) → degree=2, papers=1
crystallization (Method) → degree=4, papers=3
solvent_extraction (Method) → degree=2, papers=1
separation (Effect) → degree=5, papers=4
purity (Effect) → degree=2, papers=2
```

### Ranking:
```
1. crystallization  → score = 0.4×(4/5) + 0.3×0.8 + 0.3×0.6 = 0.68
2. evaporation      → score = 0.4×(3/5) + 0.3×0.7 + 0.3×0.4 = 0.53
3. precipitation    → score = 0.4×(2/5) + 0.3×0.6 + 0.3×0.2 = 0.38
4. separation       → score = 0.4×(5/5) + 0.3×0.5 + 0.3×0.8 = 0.64 (low priority as Effect)
5. solvent_extract… → score = 0.4×(2/5) + 0.3×0.8 + 0.3×0.2 = 0.40
```

### Top-K Selection (k=3, prioritize Method nodes):
```
✓ crystallization
✓ evaporation
✓ precipitation
```

### Explanation Building:
```
crystallization:
  - incoming: evaporation --[leads_to]--> crystallization
  - outgoing: crystallization --[produces]--> separation
  - explanation: "Evaporation leads to crystallization, which produces high-purity lithium separation."

evaporation:
  - outgoing: evaporation --[leads_to]--> crystallization
  - explanation: "Evaporation leads to crystallization."

precipitation:
  - incoming: acidification --[causes]--> precipitation
  - outgoing: precipitation --[enables]--> separation
  - explanation: "Acidification causes precipitation, which enables further separation."
```

### Coverage Validation:
```
Check: method_process → need ≥ 3 Method nodes
Found: crystallization, evaporation, precipitation, solvent_extraction = 4 ✓
Status: SUFFICIENT
```

### Final Answer:
```markdown
**Common methods/approaches:**

1. Crystallization
   - Evaporation leads to crystallization, which produces high-purity lithium separation.

2. Evaporation
   - Evaporation leads to crystallization.

3. Precipitation
   - Acidification causes precipitation, which enables further separation.

**Confidence:** 78%
**Coverage:** 100%
**Selected nodes:** 3 methods, 2 effects
**Reasoning paths:** 4
```

---

## 🧪 Testing & Validation

### Unit Tests (Recommended Add-ons):

```python
# Test node ranking
def test_rank_nodes_for_answer():
    ...

# Test coverage validation per reasoning type
def test_validate_coverage_method():
    ...

def test_validate_coverage_causal():
    ...

# Test explanation extraction from relations
def test_build_node_explanation():
    ...

# Test adaptive retry planning
def test_plan_retry_actions():
    ...
```

### Integration Tests:

```python
# End-to-end: answer compiler on sample graphs
def test_compile_answer_method_query():
    graph = build_sample_graph("method")
    plan = {"type": "method_process", "entities": [...]}
    result = compile_answer(graph, plan, papers)
    assert result.status == "success"
    assert "method" in result.answer.lower()

def test_compile_answer_insufficient_coverage():
    graph = build_sparse_graph()  # Only 1 method node
    plan = {"type": "method_process", ...}
    result = compile_answer(graph, plan, papers)
    assert result.status == "insufficient"
    assert len(result.missing_requirements) > 0
```

### Example Queries for Live Testing:

1. **Definition:** "What is photosynthesis?"
2. **Method:** "How can we reduce soil erosion?"
3. **Comparison:** "What are the differences between RNA and DNA?"
4. **Causal:** "Why do some plants grow faster than others?"
5. **Optimization:** "What's the best way to improve crop yields?"

---

## ⚙️ Configuration & Constants

### In `config/settings.py`:

```python
# Answer compiler settings (add these if not present)
ANSWER_COMPILER_TOP_K_NODES = 5        # Top-K nodes to select
ANSWER_COMPILER_MAX_RETRIES = 2        # Max adaptive retry attempts
ANSWER_COMPILER_DEGREE_WEIGHT = 0.4    # Weight for degree centrality
ANSWER_COMPILER_ALIGNMENT_WEIGHT = 0.3 # Weight for query alignment
ANSWER_COMPILER_SUPPORT_WEIGHT = 0.3   # Weight for paper support

ADAPTIVE_RETRY_ENABLED = True          # Enable retry mechanism
ADAPTIVE_RETRY_EXPAND_SCALE = 0.8      # Scale for expansion retrieval
```

---

## 🚀 Deployment Checklist

- [ ] `pipeline/answer_compiler.py` created and syntax-checked
- [ ] `pipeline/adaptive_retry.py` created and syntax-checked
- [ ] `main.py` imports updated
- [ ] Stage 7 answer generation replaced with compiler + retry loop
- [ ] Result payload includes compiler metrics
- [ ] All imports tested: `from pipeline.answer_compiler import compile_answer`
- [ ] Streamlit app launched and tested
- [ ] Sample queries run through full pipeline
- [ ] Coverage validation works for all 5 reasoning types
- [ ] Adaptive retry triggered and graphs merged correctly

---

## 📝 Key Differences from Previous System

| Aspect | Previous | New |
|--------|----------|-----|
| **Answer Source** | LLM summarizes papers | Graph relations + node types |
| **Fallback** | Paper-based summary | LLM-only as last resort |
| **Coverage Check** | Implicit (LLM generated anyway) | Explicit validation; rejects weak graphs |
| **Retry on Weak Graph** | Run LLM repeatedly | Expand query, re-extract, re-rank |
| **Answer Format** | Free-form text | Structured per reasoning type |
| **Explanation Source** | Paper abstracts | Graph edges converted to text |
| **Node Selection** | Not explicit | Ranked by centrality + alignment |

---

## 🎓 Design Rationale

### Why Separate Ranking Criteria?

- **Degree Centrality (40%):** Nodes with many neighbors are likely central concepts
- **Query Alignment (30%):** Closeness to seed entities (query terms) ensures relevance
- **Paper Support (30%):** Nodes mentioned in multiple papers are more reliable

### Why Coverage Validation by Reasoning Type?

Different question types need different graph structures:
- Methods need multiple Method nodes but can work with few Effects
- Causal questions need Causes, Effects, AND paths between them
- Definitions need a central concept plus context properties

### Why Max 2 Retries?

- Prevents infinite loops
- Diminishing returns: 2nd retry often doesn't improve much
- Most missing information caught in first retry

---

## 📚 References

- **Query Planner**: `pipeline/query_planner.py` produces plan with reasoning type
- **Graph Builder**: `pipeline/graph_builder.py` builds graph with typed nodes/edges
- **Graph Retriever**: `pipeline/graph_retriever.py` extracts query-relevant subgraph
- **Entity Extractor**: `pipeline/entity_extractor.py` extracts typed entities + relations

---

## 🤝 Integration Points

The compiler integrates smoothly with existing pipeline:

1. **Input from Graph Retriever**: Subgraph + context triples
2. **Input from Query Planner**: Plan with reasoning type
3. **Input from Paper Filter**: List of supporting papers
4. **Output to Main**: CompileResult with answer + metadata
5. **Fallback to Answer Generator**: Only if compiler fails (rare)

---

## 🔍 Debugging

### Enable Debug Logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("answer_compiler")
```

### Check Compilation Status:

```python
result = compile_answer(graph, plan, papers)
print(f"Status: {result.status}")
print(f"Coverage: {result.coverage:.1%}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Selected nodes: {[n.label for n in result.selected_nodes]}")
print(f"Missing requirements: {result.missing_requirements}")
```

### Inspect Retry Process:

Main pipeline logs `retry #N/2` messages with coverage improvements.

---

## 📞 Support & Troubleshooting

### Issue: "Status: insufficient, Coverage: 30%"
→ Graph too sparse. Check: Did retrieval get enough papers? Are entity types correct?

### Issue: "LLM fallback used"
→ Answer compiler couldn't reach 70%+ coverage even after 2 retries. Graph structure is weak for this query.

### Issue: "No nodes selected"
→ Graph nodes don't match priority types for this reasoning. Check query plan accuracy.

---

**End of Document**
