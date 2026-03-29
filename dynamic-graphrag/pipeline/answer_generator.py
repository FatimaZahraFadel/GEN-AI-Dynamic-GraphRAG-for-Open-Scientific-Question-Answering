"""
Stage 7 — Answer Generator: synthesize a grounded answer from graph context.
"""

from typing import List, Optional

import networkx as nx


class AnswerGenerator:
    """
    Generates a natural-language answer to the user query by conditioning
    an LLM on the context extracted from the retrieved knowledge subgraph.

    The generator formats a RAG-style prompt that includes the linearized
    subgraph context and the original question, then calls the LLM API.

    Attributes:
        model: LLM model identifier for answer generation.
        max_context_tokens: Token budget for the graph context section of the prompt.
        temperature: Sampling temperature for the LLM.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_context_tokens: int = 3000,
        temperature: float = 0.2,
    ) -> None:
        """
        Initialize the answer generator.

        Args:
            model: OpenAI model to use for generation.
            max_context_tokens: Maximum tokens allocated to graph context.
            temperature: LLM sampling temperature. Lower = more deterministic.
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature

    def generate(self, query: str, context: str) -> str:
        """
        Generate a grounded answer given the query and the graph context string.

        Args:
            query: Original user question.
            context: Linearized subgraph context produced by GraphRetriever.

        Returns:
            str: Generated answer text.
        """
        pass

    def generate_with_citations(self, query: str, context: str, paper_ids: List[str]) -> dict:
        """
        Generate an answer and return it alongside source paper references.

        Args:
            query: Original user question.
            context: Linearized subgraph context.
            paper_ids: IDs of papers included in the context.

        Returns:
            dict: Dictionary with keys "answer" (str) and "citations" (List[str]).
        """
        pass

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Assemble the full prompt sent to the LLM.

        Args:
            query: User question.
            context: Graph context string.

        Returns:
            str: Complete prompt string.
        """
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Send the prompt to the LLM API and return the raw response text.

        Args:
            prompt: Fully assembled prompt.

        Returns:
            str: Raw text output from the LLM.
        """
        pass
