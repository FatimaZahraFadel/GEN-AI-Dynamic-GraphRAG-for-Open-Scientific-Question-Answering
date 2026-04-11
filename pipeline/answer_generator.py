"""
Stage 7 — Answer Generator: generate a grounded, cited answer via Groq LLM.

Takes the user question, the graph context triples produced by GraphRetriever,
and the filtered paper list, then assembles a RAG-style prompt and calls the
Groq LLM to produce a factual, citation-aware answer.
"""

import os
import re
from typing import Dict, List, Optional

import networkx as nx
from dotenv import load_dotenv
from groq import Groq

from config.settings import (
    ANSWER_PROMPT_TOKEN_BUDGET,
    MAX_ABSTRACT_TOKENS,
    MAX_CONTEXT_TRIPLES,
    SPARSE_GRAPH_THRESHOLD,
)
from models.paper import Paper
from utils.helpers import truncate_text
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_MAX_PAPERS = 5
_ABSTRACT_TRUNCATE = 300
_INTENT_MIN_ITEMS = {
    "solution": 6,
    "list": 6,
    "comparison": 4,
    "process": 4,
    "cause": 4,
    "explanation": 4,
    "fact": 4,
}


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
        self.llm_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context_text: str,
        papers: List[Paper],
        subgraph: Optional[nx.DiGraph] = None,
        reasoning_steps: Optional[List[str]] = None,
        reasoning_paths: Optional[List[Dict]] = None,
        intent: str = "fact",
        evidence_assessment: Optional[Dict] = None,
        low_confidence_mode: bool = False,
        runtime_metrics: Optional[Dict[str, float]] = None,
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
                snippet = truncate_text((p.abstract or "").strip(), 220)
                supplement_lines.append(f"[Supp {i}] {p.title}\n{snippet}")
            context_text = context_text + "\n" + "\n\n".join(supplement_lines)

        prompt = self.build_prompt(
            question,
            context_text,
            papers,
            reasoning_steps=reasoning_steps or [],
            reasoning_paths=reasoning_paths or [],
            intent=intent,
            evidence_assessment=evidence_assessment or {},
            low_confidence_mode=low_confidence_mode,
        )
        if runtime_metrics is not None:
            runtime_metrics["llm_calls"] = runtime_metrics.get("llm_calls", 0) + 1
        answer = self._call_llm(prompt)

        quality_issues = self._detect_quality_issues(answer, intent=intent, question=question)
        if quality_issues:
            logger.warning(
                "Answer quality gate triggered: %s. Running repair pass.",
                "; ".join(quality_issues),
            )
            repair_prompt = self._build_repair_prompt(
                question=question,
                intent=intent,
                initial_answer=answer,
                quality_issues=quality_issues,
                context_text=context_text,
                papers=selected_papers,
            )
            if runtime_metrics is not None:
                runtime_metrics["llm_calls"] = runtime_metrics.get("llm_calls", 0) + 1
            answer = self._call_llm(repair_prompt)

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
        reasoning_steps: List[str] | None = None,
        reasoning_paths: List[Dict] | None = None,
        intent: str = "fact",
        evidence_assessment: Dict | None = None,
        low_confidence_mode: bool = False,
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
        reasoning_steps = reasoning_steps or []
        reasoning_paths = reasoning_paths or []
        evidence_assessment = evidence_assessment or {}

        context_text = self._select_top_context_triples(
            question=question,
            context_text=context_text,
            max_triples=MAX_CONTEXT_TRIPLES,
        )

        # Build reasoning chain block from top evidence paths and controller steps.
        chain_lines: List[str] = []
        for i, path_info in enumerate(reasoning_paths[:5], 1):
            labels = path_info.get("labels", [])
            if not labels:
                continue
            chain_lines.append(
                f"Step {i}: {' -> '.join(labels)} means these entities form a linked evidence path."
            )
        for step in reasoning_steps[:4]:
            chain_lines.append(step)
        if not chain_lines:
            chain_lines.append("Step 1: No explicit multi-hop evidence path was found; rely on direct relations.")
        reasoning_chain_block = "\n".join(chain_lines[:6])

        # Build the paper evidence block
        selected_papers = papers[:_MAX_PAPERS]
        evidence_terms = self._extract_candidate_terms(question, selected_papers)
        paper_lines: List[str] = []
        for i, paper in enumerate(selected_papers, 1):
            abstract_snippet = truncate_text(
                (paper.abstract or "").strip(),
                MAX_ABSTRACT_TOKENS,
            )
            if len(abstract_snippet) > _ABSTRACT_TRUNCATE:
                abstract_snippet = abstract_snippet[:_ABSTRACT_TRUNCATE].strip()
                abstract_snippet += "..."
            paper_lines.append(
                f"[{i}] Title: {paper.title}\n"
                f"    Abstract: {abstract_snippet}"
            )
        papers_block = "\n\n".join(paper_lines) if paper_lines else "No papers available."
        evidence_terms_block = ", ".join(evidence_terms[:12]) if evidence_terms else "None detected from titles/abstracts."
        evidence_diag_block = (
            f"consistency_score={evidence_assessment.get('consistency_score', 0.0)}; "
            f"is_consistent={evidence_assessment.get('is_consistent', False)}; "
            f"avg_pairwise_overlap={evidence_assessment.get('avg_pairwise_overlap', 0.0)}"
        )
        confidence_instruction = ""
        if low_confidence_mode:
            confidence_instruction = (
                "- Confidence is low. Use cautious language, separate strong vs weak evidence, "
                "and explicitly list uncertainties.\n"
            )
        effective_mode = self._infer_effective_mode(intent=intent, question=question)
        min_items = _INTENT_MIN_ITEMS.get(effective_mode, _INTENT_MIN_ITEMS.get(intent, 4))
        intent_instruction = self._build_intent_instruction(intent=effective_mode, min_items=min_items)

        prompt = f"""You are a scientific question-answering assistant.

Your task is to answer the question below using ONLY the information provided in the Graph Context and Supporting Papers sections. Do not use any outside knowledge or make speculative claims.

Instructions:
- Answer immediately with the direct answer; do not start with generic intro text.
- Organize output using clear section headings and bullet points.
- Keep every item concrete and actionable; avoid vague advice.
- Cite supporting papers using the format [Paper: <title>] when you draw from them.
- Prefer terms that appear in Graph Context, Candidate Evidence Terms, or Supporting Papers; do not invent entities.
- Remove duplicates and weak graph nodes; keep only clear, relevant concepts.
- Do not speculate beyond what the context supports.
{intent_instruction}{confidence_instruction}- Include trade-offs or common mistakes where helpful.
- End with: Practical Optimization Workflow (3-5 concise steps) when relevant.

Output format:
- Direct Answer:
    - ...
- Category 1:
    - Technique -> What it is. When to use it. (optionally trade-off)
- Category 2:
    - Technique -> What it is. When to use it. (optionally trade-off)
- Practical Optimization Workflow:
    1. ...
    2. ...
    3. ...

---
QUESTION:
{question}

---
REASONING CHAIN (controller guidance):
{reasoning_chain_block}

---
GRAPH CONTEXT (knowledge graph triples extracted from scientific literature):
{context_text}

---
CANDIDATE EVIDENCE TERMS (auto-mined from evidence text):
{evidence_terms_block}

---
EVIDENCE DIAGNOSTICS:
{evidence_diag_block}

---
SUPPORTING PAPERS:
{papers_block}

---
ANSWER:"""

        # Hard token-budget guard.
        prompt = truncate_text(prompt, ANSWER_PROMPT_TOKEN_BUDGET)

        logger.info(
            f"Prompt built: {len(selected_papers)} paper(s), "
            f"{len(context_text)} chars of graph context."
        )
        return prompt

    def _select_top_context_triples(
        self,
        question: str,
        context_text: str,
        max_triples: int,
    ) -> str:
        """Keep the most question-relevant relation lines from graph context."""
        if max_triples <= 0:
            return ""

        lines = context_text.splitlines()
        triple_lines = [line for line in lines if "--[" in line and "]-->" in line]
        if len(triple_lines) <= max_triples:
            return context_text

        q_tokens = set(re.findall(r"\b[a-z]{3,}\b", question.lower()))

        def _score(line: str) -> int:
            tokens = set(re.findall(r"\b[a-z]{3,}\b", line.lower()))
            return len(tokens & q_tokens)

        top_lines = sorted(triple_lines, key=_score, reverse=True)[:max_triples]
        top_line_set = set(top_lines)

        out: List[str] = []
        for line in lines:
            if "--[" in line and "]-->" in line and line not in top_line_set:
                continue
            out.append(line)
        return "\n".join(out)

    def _extract_candidate_terms(self, question: str, papers: List[Paper]) -> List[str]:
        """Extract question-aligned evidence terms from paper titles/abstracts."""
        q_tokens = set(re.findall(r"\b[a-z]{4,}\b", (question or "").lower()))
        stop = {
            "what", "which", "when", "where", "does", "that", "this",
            "from", "with", "into", "about", "their", "there", "these",
        }
        q_tokens = {t for t in q_tokens if t not in stop}
        if not q_tokens:
            return []

        mentions: Dict[str, int] = {}
        for paper in papers:
            text = f"{paper.title}. {paper.abstract or ''}"
            tokens = re.findall(r"\b[a-z][a-z\-]{3,}\b", text.lower())
            for tok in tokens:
                if tok in q_tokens:
                    mentions[tok] = mentions.get(tok, 0) + 1

        ranked = sorted(mentions.items(), key=lambda x: (-x[1], x[0]))
        return [name for name, _ in ranked]

    def _build_intent_instruction(self, intent: str, min_items: int) -> str:
        """Build intent-specific output requirements for stronger answer quality."""
        if intent == "solution":
            return (
                f"- This is a solution question. Provide at least {min_items} concrete techniques.\\n"
                "- Group techniques into categories (model-level, training, data, hyperparameter tuning).\\n"
                "- For each technique include: what it is (1 sentence) and when to use it (1 sentence).\\n"
                "- Do not repeat the same concept with different wording.\\n"
            )
        if intent == "comparison":
            return (
                f"- This is a comparison question. Provide at least {min_items} compared items.\\n"
                "- For each item include key difference and trade-off.\\n"
            )
        if intent in {"process", "cause", "explanation"}:
            return (
                f"- Provide at least {min_items} core points and connect cause-effect or mechanism clearly.\\n"
            )
        if intent == "list":
            return (
                f"- This is a list question. Provide at least {min_items} non-duplicate items.\\n"
            )
        return (
            f"- Provide at least {min_items} concrete points if the question asks for multiple methods/issues.\\n"
        )

    def _infer_effective_mode(self, intent: str, question: str) -> str:
        """Infer stronger answer mode from question wording when needed."""
        q = (question or "").lower()

        list_signals = [
            "what are", "list", "techniques", "methods", "strategies", "approaches",
            "best practices", "ways to", "how to optimize", "optimization techniques",
        ]
        solution_signals = [
            "fix", "improve", "optimize", "reduce", "mitigate", "enhance",
        ]

        if any(sig in q for sig in list_signals):
            if any(sig in q for sig in solution_signals):
                return "solution"
            return "list"

        if intent in _INTENT_MIN_ITEMS:
            return intent
        return "fact"

    def _count_list_items(self, text: str) -> int:
        """Count bullet/numbered list items in generated answer."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        item_re = re.compile(r"^(?:[-*]|\d+[.)])\s+")
        arrow_re = re.compile(r"^[A-Za-z].*->")
        return sum(1 for ln in lines if item_re.match(ln) or arrow_re.match(ln))

    def _detect_quality_issues(self, answer: str, intent: str, question: str) -> List[str]:
        """Return human-readable quality violations for auto-repair."""
        issues: List[str] = []
        text = (answer or "").strip()
        if not text:
            return ["empty_answer"]

        effective_mode = self._infer_effective_mode(intent=intent, question=question)
        min_items = _INTENT_MIN_ITEMS.get(effective_mode, 4)
        item_count = self._count_list_items(text)
        if effective_mode in {"solution", "list", "comparison", "process", "cause", "explanation"} and item_count < min_items:
            issues.append(f"insufficient_items:{item_count}<{min_items}")

        lower = text.lower()
        if effective_mode in {"solution", "list"}:
            if "practical optimization workflow" not in lower:
                issues.append("missing_workflow_section")
            category_markers = ["model", "training", "data", "hyperparameter"]
            if sum(1 for m in category_markers if m in lower) < 2:
                issues.append("missing_category_structure")

        if len(text) < 500:
            issues.append("too_short")

        return issues

    def _build_repair_prompt(
        self,
        question: str,
        intent: str,
        initial_answer: str,
        quality_issues: List[str],
        context_text: str,
        papers: List[Paper],
    ) -> str:
        """Build a strict rewrite prompt to repair low-quality drafts."""
        min_items = _INTENT_MIN_ITEMS.get(intent, 4)
        paper_titles = "\n".join(f"- {p.title}" for p in papers[:_MAX_PAPERS]) or "- None"
        return f"""Rewrite the draft answer to satisfy all constraints exactly.

Question: {question}
Intent: {intent}
Detected quality issues: {', '.join(quality_issues)}

Hard constraints:
- Provide at least {min_items} concrete items.
- Use clear category headings and bullet points.
- For each item include: what it is + when to use it.
- Include practical trade-offs/common mistakes briefly.
- Keep grounded to provided graph context and papers; no unsupported claims.
- Avoid duplicate concepts.
- End with section title exactly: Practical Optimization Workflow
- Under that section include 3-5 numbered steps.

Graph Context:
{context_text}

Supporting Papers:
{paper_titles}

Draft answer to improve:
{initial_answer}

Return only the improved final answer."""

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
            self.llm_calls += 1
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
