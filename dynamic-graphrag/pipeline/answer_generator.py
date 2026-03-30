"""
Stage 7 — Answer Generator: generate a grounded, cited answer via Groq LLM.

Takes the user question, the graph context triples produced by GraphRetriever,
and the filtered paper list, then assembles a RAG-style prompt and calls the
Groq LLM to produce a factual, citation-aware answer.
"""

import os
from typing import Dict, List, Optional

import networkx as nx
from dotenv import load_dotenv
from groq import Groq

from config.settings import SPARSE_GRAPH_THRESHOLD
from models.paper import Paper
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_MAX_PAPERS = 5
_ABSTRACT_TRUNCATE = 300


class AnswerGenerator:
    """
    Generates a grounded natural-language answer conditioned on the subgraph
    context and supporting paper abstracts retrieved for the current query.

    The LLM is instructed to answer solely from the provided context, cite
    paper titles using ``[Paper: title]`` notation, and avoid speculation.

    Attributes
    ----------
    model : str
        Groq model identifier used for generation.
    temperature : float
        Sampling temperature. Lower values produce more deterministic output.
    _client : Groq or None
        Authenticated Groq API client (lazy-initialised on first call).
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
    ) -> None:
        """
        Initialise the answer generator.

        Parameters
        ----------
        model : str
            Groq model to use. Defaults to ``llama-3.1-8b-instant``.
        temperature : float
            LLM sampling temperature. Defaults to 0.2 (near-deterministic).
        """
        self.model = model
        self.temperature = temperature
        self._client: Groq | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context_text: str,
        papers: List[Paper],
        subgraph: Optional[nx.DiGraph] = None,
    ) -> Dict:
        """
        Main entry point: build the prompt, call Groq, and return results.

        If ``subgraph`` is provided and has fewer than
        ``SPARSE_GRAPH_THRESHOLD`` edges, the top-3 paper abstracts are
        appended directly to ``context_text`` before prompt assembly so the
        LLM has sufficient grounding material.

        Parameters
        ----------
        question : str
            Original natural-language question from the user.
        context_text : str
            Graph triple context produced by :class:`GraphRetriever`.
        papers : list[Paper]
            Filtered papers whose abstracts provide additional evidence.
        subgraph : nx.DiGraph or None
            Retrieved subgraph; used only to check sparseness. Optional.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``question``       (str)  — original question
            - ``answer``         (str)  — generated answer text
            - ``prompt_used``    (str)  — full prompt sent to the LLM
            - ``model``          (str)  — Groq model identifier used
            - ``num_papers_used``(int)  — number of papers included in prompt
        """
        # Sparse graph check — supplement with raw abstracts when needed
        if subgraph is not None and subgraph.number_of_edges() < SPARSE_GRAPH_THRESHOLD:
            logger.warning(
                f"Sparse graph detected ({subgraph.number_of_edges()} edges < "
                f"{SPARSE_GRAPH_THRESHOLD}) — supplementing with raw abstracts."
            )
            supplement_lines = ["\n--- Supplementary Abstract Context ---"]
            for i, p in enumerate(papers[:3], 1):
                snippet = (p.abstract or "")[:400].strip()
                supplement_lines.append(f"[Supp {i}] {p.title}\n{snippet}")
            context_text = context_text + "\n" + "\n\n".join(supplement_lines)

        prompt = self.build_prompt(question, context_text, papers)
        answer = self._call_llm(prompt)

        num_papers = min(len(papers), _MAX_PAPERS)
        logger.info(
            f"Answer generated ({len(answer)} chars) using "
            f"{num_papers} paper(s) and model '{self.model}'."
        )

        return {
            "question": question,
            "answer": answer,
            "prompt_used": prompt,
            "model": self.model,
            "num_papers_used": num_papers,
        }

    def build_prompt(
        self,
        question: str,
        context_text: str,
        papers: List[Paper],
    ) -> str:
        """
        Assemble the full RAG prompt for the LLM.

        The prompt contains three sections:
        1. **Instructions** — how the model must behave (context-only, cite
           with ``[Paper: title]``, no speculation).
        2. **Graph context** — the triple-formatted subgraph text.
        3. **Supporting papers** — up to ``_MAX_PAPERS`` paper titles and
           truncated abstracts (300 chars each).

        Parameters
        ----------
        question : str
            User question.
        context_text : str
            Serialised knowledge graph triples from :class:`GraphRetriever`.
        papers : list[Paper]
            Papers to include as supporting evidence (first 5 used).

        Returns
        -------
        str
            Complete prompt string ready to send to the Groq API.
        """
        # Build the paper evidence block
        selected_papers = papers[:_MAX_PAPERS]
        paper_lines: List[str] = []
        for i, paper in enumerate(selected_papers, 1):
            abstract_snippet = (paper.abstract or "")[:_ABSTRACT_TRUNCATE].strip()
            if len(paper.abstract or "") > _ABSTRACT_TRUNCATE:
                abstract_snippet += "..."
            paper_lines.append(
                f"[{i}] Title: {paper.title}\n"
                f"    Abstract: {abstract_snippet}"
            )
        papers_block = "\n\n".join(paper_lines) if paper_lines else "No papers available."

        prompt = f"""You are a scientific question-answering assistant.

Your task is to answer the question below using ONLY the information provided in the Graph Context and Supporting Papers sections. Do not use any outside knowledge or make speculative claims.

Instructions:
- Answer in clear, concise scientific language.
- Reference specific entities and relationships from the graph triples where relevant.
- Cite supporting papers using the format [Paper: <title>] when you draw from them.
- If the context does not contain enough information to answer fully, state what is known and acknowledge the limitation.
- Do not speculate beyond what the context supports.

---
QUESTION:
{question}

---
GRAPH CONTEXT (knowledge graph triples extracted from scientific literature):
{context_text}

---
SUPPORTING PAPERS:
{papers_block}

---
ANSWER:"""

        logger.info(
            f"Prompt built: {len(selected_papers)} paper(s), "
            f"{len(context_text)} chars of graph context."
        )
        return prompt

    def format_answer(self, answer_dict: Dict) -> str:
        """
        Format the answer dictionary into a clean, readable terminal string.

        Parameters
        ----------
        answer_dict : dict
            Output of :meth:`generate`.

        Returns
        -------
        str
            Human-readable multi-line string with question, answer, metadata.
        """
        separator = "=" * 70
        return (
            f"\n{separator}\n"
            f"  QUESTION\n"
            f"{separator}\n"
            f"  {answer_dict['question']}\n\n"
            f"{separator}\n"
            f"  ANSWER\n"
            f"{separator}\n"
            f"{answer_dict['answer']}\n\n"
            f"{separator}\n"
            f"  Model       : {answer_dict['model']}\n"
            f"  Papers used : {answer_dict['num_papers_used']}\n"
            f"{separator}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Send the prompt to the Groq chat-completion API and return the answer.

        Parameters
        ----------
        prompt : str
            Fully assembled prompt string.

        Returns
        -------
        str
            Raw answer text from the LLM, stripped of leading/trailing whitespace.

        Raises
        ------
        EnvironmentError
            If ``GROQ_API_KEY`` is missing from the environment.
        Exception
            Propagates any Groq API error after logging it.
        """
        client = self._get_client()
        logger.info(f"Calling Groq API (model={self.model}, temp={self.temperature})…")

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error during answer generation: {e}")
            raise

    def _get_client(self) -> Groq:
        """
        Return the cached Groq client, creating it on first call.

        Returns
        -------
        Groq
            Authenticated Groq API client.

        Raises
        ------
        EnvironmentError
            If ``GROQ_API_KEY`` is not set in the environment.
        """
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file."
                )
            self._client = Groq(api_key=api_key)
        return self._client


# ---------------------------------------------------------------------------
# Complete end-to-end pipeline smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    from dotenv import load_dotenv
    load_dotenv()

    from pipeline.domain_detector import DomainDetector
    from pipeline.paper_retriever import PaperRetriever
    from pipeline.paper_filter import PaperFilter
    from pipeline.entity_extractor import EntityExtractor
    from pipeline.graph_builder import GraphBuilder
    from pipeline.graph_retriever import GraphRetriever

    QUESTION = "What fungal diseases affect wheat crops in humid environments?"

    print()
    print("=" * 70)
    print("  DYNAMIC GraphRAG — Full Pipeline")
    print("=" * 70)
    print(f"  Question: {QUESTION}")
    print()

    # Stage 1 — domain
    domain = DomainDetector().classify(QUESTION, method="keyword")
    print(f"  [1] Domain      : {domain}")

    # Stage 2 — retrieve papers
    raw_papers = PaperRetriever(top_n=20).retrieve(QUESTION, domain)
    print(f"  [2] Retrieved   : {len(raw_papers)} papers")

    # Stage 3 — filter + rank
    filtered = PaperFilter().filter(raw_papers, QUESTION, top_k=5)
    print(f"  [3] Filtered    : {len(filtered)} papers")

    # Stage 4 — extract entities & relations
    entities, relations = EntityExtractor().extract(filtered)
    print(f"  [4] Entities    : {len(entities)}   Relations: {len(relations)}")

    # Stage 5 — build graph
    graph = GraphBuilder().build(entities, relations)
    print(f"  [5] Graph       : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Stage 6 — retrieve subgraph + context
    retrieval = GraphRetriever().retrieve(QUESTION, graph)
    print(f"  [6] Subgraph    : {retrieval['num_nodes']} nodes, {retrieval['num_edges']} edges")

    # Stage 7 — generate answer
    generator = AnswerGenerator()
    answer_dict = generator.generate(
        question=QUESTION,
        context_text=retrieval["context_text"],
        papers=filtered,
    )

    print(generator.format_answer(answer_dict))
