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
import requests
from dotenv import load_dotenv
from groq import Groq

from config.settings import (
    ANSWER_PROMPT_TOKEN_BUDGET,
    MAX_ABSTRACT_TOKENS,
    MAX_CONTEXT_TRIPLES,
    OLLAMA_BASE_URL,
    OLLAMA_GENERAL_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    QUERY_PLANNER_MODEL,
    SPARSE_GRAPH_THRESHOLD,
    USE_OLLAMA_PRIMARY,
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
        model: str = QUERY_PLANNER_MODEL,
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
        self._use_ollama_primary = USE_OLLAMA_PRIMARY
        self._ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self._ollama_model = OLLAMA_GENERAL_MODEL
        self._ollama_timeout = OLLAMA_TIMEOUT_SECONDS
        self.llm_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context_text: str,
        papers: List[Paper],
        plan: Optional[Dict] = None,
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
            plan=plan or {},
            reasoning_steps=reasoning_steps or [],
            reasoning_paths=reasoning_paths or [],
            intent=intent,
            evidence_assessment=evidence_assessment or {},
            low_confidence_mode=low_confidence_mode,
        )
        selected_papers = papers[:_MAX_PAPERS]
        if runtime_metrics is not None:
            runtime_metrics["llm_calls"] = runtime_metrics.get("llm_calls", 0) + 1
        try:
            answer = self._call_llm(prompt)
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(
                    "Rate limit encountered during answer generation; returning fallback answer."
                )
                answer = self._build_rate_limited_fallback_answer(
                    question=question,
                    papers=selected_papers,
                    reasoning_paths=reasoning_paths or [],
                )
            else:
                raise

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
            try:
                answer = self._call_llm(repair_prompt)
            except Exception as e:
                if self._is_rate_limit_error(e):
                    logger.warning(
                        "Rate limit encountered during repair pass; keeping first-pass answer."
                    )
                else:
                    raise

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
            "low_confidence": low_confidence_mode,
        }

    def build_prompt(
        self,
        question: str,
        context_text: str,
        papers: List[Paper],
        plan: Dict | None = None,
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
        plan = plan or {}

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
        reasoning_explanations = self._reasoning_paths_to_explanations(reasoning_paths[:5])

        # Build the paper evidence block
        selected_papers = papers[:_MAX_PAPERS]
        evidence_terms = self._extract_candidate_terms(question, selected_papers)
        paper_lines: List[str] = []
        summary_lines: List[str] = []
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
            summary = self._build_extractive_summary(question, paper.abstract or "")
            summary_lines.append(
                f"[{i}] {paper.title}\n"
                f"    Summary: {summary}"
            )
        papers_block = "\n\n".join(paper_lines) if paper_lines else "No papers available."
        summaries_block = "\n\n".join(summary_lines) if summary_lines else "No text summaries available."
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
        plan_type = str(plan.get("type", "")).strip().lower()
        effective_mode = self._infer_effective_mode(intent=plan_type or intent, question=question)
        min_items = _INTENT_MIN_ITEMS.get(effective_mode, _INTENT_MIN_ITEMS.get(intent, 4))
        intent_instruction = self._build_intent_instruction(intent=effective_mode, min_items=min_items)

        prompt = f"""You are a scientific question-answering assistant covering any scientific domain.

Your task is to answer the question below using ONLY the information provided in the Graph Context and Supporting Papers sections. Do not use any outside knowledge or make speculative claims.

Instructions:
- Answer immediately and directly; do not start with generic filler text.
- Organize output using clear section headings (## bold) and bullet points appropriate to the question domain.
- Keep every item concrete and evidence-backed; cite the specific paper or graph triple that supports it.
- Base explanations on graph reasoning paths, not only abstract summaries.
- Cite supporting papers using the format [Paper: <title>] when you draw from them.
- Use terminology that appears in Graph Context, Candidate Evidence Terms, or Supporting Papers; do not invent entities.
- Do not speculate beyond what the context supports.
{intent_instruction}{confidence_instruction}- Include limitations, uncertainties, or conditions where evidence is mixed or sparse.
- Close with a brief **Summary** (2-3 sentences) consolidating the key finding.

---
QUESTION:
{question}

---
REASONING CHAIN (controller guidance):
{reasoning_chain_block}

REASONING EXPLANATIONS (path-to-language):
{reasoning_explanations}

---
GRAPH CONTEXT (knowledge graph triples extracted from scientific literature):
{context_text}

---
ORIGINAL TEXT SUMMARIES (extractive summaries from filtered papers):
{summaries_block}

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

    def _reasoning_paths_to_explanations(self, reasoning_paths: List[Dict]) -> str:
        """Convert graph paths into explicit causal/explanatory sentences."""
        if not reasoning_paths:
            return "No multi-hop reasoning path available."

        lines: List[str] = []
        for idx, path in enumerate(reasoning_paths, 1):
            labels = path.get("labels", [])
            if len(labels) < 2:
                continue
            if len(labels) == 2:
                lines.append(f"Path {idx}: {labels[0]} is connected to {labels[1]}.")
            else:
                links = [f"{labels[i]} influences {labels[i+1]}" for i in range(len(labels) - 1)]
                lines.append(f"Path {idx}: " + ", then ".join(links) + ".")

        return "\n".join(lines) if lines else "No multi-hop reasoning path available."

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

    def _build_extractive_summary(self, question: str, abstract: str, max_sentences: int = 2) -> str:
        """Build lightweight query-focused extractive summary from abstract text."""
        text = (abstract or "").strip()
        if not text:
            return "No abstract content available."

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return truncate_text(text, 220)

        q_terms = set(re.findall(r"\b[a-z][a-z\-]{3,}\b", (question or "").lower()))
        stop = {
            "what", "which", "when", "where", "that", "this", "there", "their",
            "with", "from", "into", "about", "using", "based",
        }
        q_terms = {t for t in q_terms if t not in stop}

        scored: List[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            toks = set(re.findall(r"\b[a-z][a-z\-]{3,}\b", sent.lower()))
            overlap = len(toks & q_terms) / max(len(q_terms), 1) if q_terms else 0.0
            length_bonus = min(len(sent) / 220.0, 1.0)
            score = (0.8 * overlap) + (0.2 * length_bonus)
            scored.append((score, idx, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = sorted(scored[:max_sentences], key=lambda x: x[1])
        summary = " ".join(s for _, _, s in chosen)
        return truncate_text(summary, 260)

    def _build_intent_instruction(self, intent: str, min_items: int) -> str:
        """Build intent-specific output requirements for stronger answer quality."""
        if intent == "solution":
            return (
                f"- This is a solution/methods question. Provide at least {min_items} concrete techniques or approaches.\\n"
                "- Group related techniques under thematic section headings (e.g. by mechanism, scale, or application context).\\n"
                "- For each technique include: what it is (1 sentence) and the evidence or condition supporting it (1 sentence).\\n"
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
                f"- This is a list question. Provide at least {min_items} non-duplicate, domain-specific items.\\n"
                "- For each item give a brief description grounded in the provided evidence.\\n"
            )
        return (
            f"- Provide at least {min_items} concrete, evidence-grounded points.\\n"
        )

    def _infer_effective_mode(self, intent: str, question: str) -> str:
        """Infer stronger answer mode from question wording when needed."""
        q = (question or "").lower()
        mapped = {
            "method/process": "process",
            "causal/mechanism": "cause",
            "optimization/recommendation": "solution",
            "comparison": "comparison",
            "definition": "fact",
        }
        if intent in mapped:
            intent = mapped[intent]

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

        if effective_mode in {"solution", "list"}:
            # Check for section/heading structure — any heading pattern counts,
            # not a domain-specific phrase.
            has_sections = bool(
                re.search(r"(?:^|\n)#{1,3}\s+\w", text)          # markdown headings
                or re.search(r"(?:^|\n)[A-Z][A-Za-z ]{3,}:\s*\n", text)  # "Title:\n" style
                or re.search(r"(?:^|\n)\*\*[^*]+\*\*", text)     # bold headers
            )
            if not has_sections and item_count < 3:
                issues.append("missing_structure")

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
- Use clear category headings (bold or markdown ##) and bullet points.
- For each item include: what it is + evidence or mechanism from the papers.
- Include relevant caveats, limitations, or conditions where applicable.
- Keep grounded to provided graph context and papers; no unsupported claims.
- Avoid duplicate concepts.
- End with a brief summary section (2-3 sentences) consolidating the key finding.

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
        if self._use_ollama_primary:
            logger.info(
                "Calling Ollama API (model=%s, temp=%s)...",
                self._ollama_model,
                self.temperature,
            )
            try:
                ollama_text = self._call_ollama(prompt)
                if ollama_text:
                    return ollama_text
            except Exception as e:
                logger.warning("Ollama primary call failed; falling back to Groq: %s", e)

        client = self._get_client()
        logger.info("Calling Groq API (model=%s, temp=%s)...", self.model, self.temperature)

        try:
            self.llm_calls += 1
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=ANSWER_PROMPT_TOKEN_BUDGET,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Groq API error during answer generation: %s", e)
            raise

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama and return generated text."""
        self.llm_calls += 1
        response = requests.post(
            f"{self._ollama_base_url}/api/generate",
            json={
                "model": self._ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 1024,
                },
            },
            timeout=self._ollama_timeout,
        )
        response.raise_for_status()
        payload = response.json() or {}
        text = (payload.get("response") or "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Return True when an exception looks like a provider rate-limit error."""
        msg = str(error).lower()
        markers = [
            "rate limit",
            "rate_limit_exceeded",
            "error code: 429",
            "tokens per day",
            "tokens per minute",
            "tpm",
            "tpd",
        ]
        return any(marker in msg for marker in markers)

    def _build_rate_limited_fallback_answer(
        self,
        question: str,
        papers: List[Paper],
        reasoning_paths: List[Dict],
    ) -> str:
        """
        Build a deterministic, domain-neutral grounded answer when LLM calls
        are rate-limited.

        Assembles the answer directly from paper titles and abstracts so the
        content is always topically relevant to the question.
        """
        lines: List[str] = []
        lines.append(f"**Partial answer to:** {question}")
        lines.append(
            "\n*Note: LLM generation was rate-limited. "
            "The following summary was assembled directly from retrieved papers.*\n"
        )

        if not papers:
            lines.append("No papers were retrieved for this query.")
            return "\n".join(lines)

        lines.append("**Key findings from retrieved papers:**\n")
        for i, paper in enumerate(papers[:6], 1):
            title = paper.title or "Untitled"
            year = paper.year if paper.year else "n.d."
            snippet = (paper.abstract or "").strip()
            # Take first 2 sentences of abstract as summary
            sentences = re.split(r"(?<=[.!?])\s+", snippet)
            summary = " ".join(sentences[:2]) if sentences else ""
            summary = truncate_text(summary, 280)
            lines.append(f"{i}. **{title}** ({year})")
            if summary:
                lines.append(f"   {summary}")
            lines.append("")

        if reasoning_paths:
            lines.append("**Graph reasoning paths identified:**")
            for p in reasoning_paths[:3]:
                labels = p.get("labels", [])
                if labels:
                    lines.append(f"- {' → '.join(labels)}")
            lines.append("")

        lines.append("**Sources:**")
        for paper in papers[:5]:
            lines.append(f"- [Paper: {paper.title}]")

        return "\n".join(lines)

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
