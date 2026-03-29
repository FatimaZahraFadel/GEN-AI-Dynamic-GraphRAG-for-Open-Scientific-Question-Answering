"""
Dynamic GraphRAG — main entry point.

This module wires together all pipeline stages and demonstrates the full
end-to-end flow for answering a scientific question via graph-augmented
retrieval.  Each stage is called in sequence; intermediate results are
passed through to the next stage.

Pipeline stages
---------------
1. DomainDetector   — classify the query into one or more scientific domains
2. PaperRetriever   — fetch candidate papers from Semantic Scholar / OpenAlex
3. PaperFilter      — embed, score, and rank papers; keep top-N
4. EntityExtractor  — extract concepts / methods / datasets from each paper
5. GraphBuilder     — build a knowledge graph from papers + entities
6. GraphRetriever   — retrieve a query-relevant subgraph
7. AnswerGenerator  — generate a grounded answer conditioned on the subgraph
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Import pipeline stages ---
from pipeline.domain_detector import DomainDetector
from pipeline.paper_retriever import PaperRetriever
from pipeline.paper_filter import PaperFilter
from pipeline.entity_extractor import EntityExtractor
from pipeline.graph_builder import GraphBuilder
from pipeline.graph_retriever import GraphRetriever
from pipeline.answer_generator import AnswerGenerator

# --- Import evaluation ---
from evaluation.evaluator import Evaluator

# --- Import config ---
from config.settings import DOMAINS, TOP_N_PAPERS, MAX_GRAPH_NODES, EMBEDDING_MODEL

# --- Import logger ---
from utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(query: str) -> str:
    """
    Execute the full Dynamic GraphRAG pipeline for a single user query.

    Args:
        query: Natural-language scientific question.

    Returns:
        str: Generated answer grounded in the retrieved knowledge graph.
    """

    logger.info("Starting Dynamic GraphRAG pipeline")
    logger.info(f"Query: {query}")

    # ------------------------------------------------------------------
    # Stage 1: Domain Detection
    # Classify the query into one or more scientific domains so that
    # retrieval can be scoped to the relevant literature.
    # ------------------------------------------------------------------
    detector = DomainDetector(domains=DOMAINS)
    # domains: List[str] — e.g. ["computer_science", "biology"]
    domains = detector.detect(query)
    logger.info(f"Detected domains: {domains}")

    # ------------------------------------------------------------------
    # Stage 2: Paper Retrieval
    # Query Semantic Scholar and OpenAlex for candidate papers.
    # Both APIs are tried; results are merged and deduplicated.
    # ------------------------------------------------------------------
    retriever = PaperRetriever(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        openalex_email=os.getenv("OPENALEX_EMAIL"),
        top_n=TOP_N_PAPERS,
    )
    # papers: List[Paper]
    papers = retriever.retrieve(query=query, domain=domains[0] if domains else "")
    logger.info(f"Retrieved {len(papers)} candidate papers")

    # ------------------------------------------------------------------
    # Stage 3: Paper Filtering
    # Embed the query and each paper (title + abstract), compute cosine
    # similarity scores, optionally boost by citation count, keep top-N.
    # ------------------------------------------------------------------
    paper_filter = PaperFilter(embedding_model=EMBEDDING_MODEL, top_n=TOP_N_PAPERS)
    # filtered_papers: List[Paper] — sorted by relevance_score desc
    filtered_papers = paper_filter.filter(papers=papers, query=query)
    logger.info(f"Filtered to {len(filtered_papers)} papers")

    # ------------------------------------------------------------------
    # Stage 4: Entity Extraction
    # For each filtered paper, send its title+abstract to the LLM and
    # extract structured entities (concepts, methods, datasets, tasks).
    # ------------------------------------------------------------------
    extractor = EntityExtractor()
    # entity_map: Dict[paper_id -> List[entity_label]]
    entity_map = extractor.extract(filtered_papers)
    total_entities = sum(len(v) for v in entity_map.values())
    logger.info(f"Extracted {total_entities} entities across {len(entity_map)} papers")

    # ------------------------------------------------------------------
    # Stage 5: Graph Construction
    # Create paper nodes and entity nodes; draw edges for:
    #   - paper ↔ entity  (containment / mentions)
    #   - paper → paper   (citation links, if available)
    #   - paper ↔ paper   (high semantic similarity)
    # Prune to MAX_GRAPH_NODES if necessary.
    # ------------------------------------------------------------------
    builder = GraphBuilder(max_nodes=MAX_GRAPH_NODES)
    # graph: nx.DiGraph
    graph = builder.build(papers=filtered_papers, entity_map=entity_map)
    logger.info(
        f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # ------------------------------------------------------------------
    # Stage 6: Graph Retrieval
    # Find seed nodes closest to the query embedding, then expand via BFS
    # to include neighboring nodes up to hop_depth hops away.
    # The subgraph is then linearized into a context string.
    # ------------------------------------------------------------------
    graph_retriever = GraphRetriever(
        embedding_model=EMBEDDING_MODEL,
        top_k_nodes=10,
        hop_depth=2,
    )
    # subgraph: nx.DiGraph — query-relevant portion of the knowledge graph
    subgraph = graph_retriever.retrieve(graph=graph, query=query)
    # context: str — textual representation for the LLM prompt
    context = graph_retriever.subgraph_to_context(subgraph)
    logger.info(
        f"Retrieved subgraph: {subgraph.number_of_nodes()} nodes, "
        f"{subgraph.number_of_edges()} edges"
    )

    # ------------------------------------------------------------------
    # Stage 7: Answer Generation
    # Combine the graph context with the original query in a RAG-style
    # prompt and call the LLM to produce the final grounded answer.
    # ------------------------------------------------------------------
    generator = AnswerGenerator(model="gpt-4o", temperature=0.2)
    # answer: str
    answer = generator.generate(query=query, context=context)
    logger.info("Answer generated successfully")

    return answer


def run_evaluation(dataset_path: str) -> None:
    """
    Run the pipeline over a QA dataset and report evaluation metrics.

    Args:
        dataset_path: Path to a JSON file with {"question", "answer"} records.
    """
    evaluator = Evaluator(metrics=["rouge", "bertscore", "exact_match"])
    scores = evaluator.evaluate_dataset(dataset_path)
    print("Evaluation results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    # Example question — replace or wire to a CLI / web interface
    sample_query = (
        "What are the most effective transformer-based methods for "
        "protein structure prediction?"
    )

    answer = run_pipeline(sample_query)
    print("\n=== Generated Answer ===")
    print(answer)
