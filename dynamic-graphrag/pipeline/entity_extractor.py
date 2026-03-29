"""
Stage 4 — Entity Extractor: identify scientific entities from paper texts.
"""

from typing import Dict, List

from models.paper import Paper


class EntityExtractor:
    """
    Extracts named entities (concepts, methods, datasets, tasks) from the
    title and abstract of each paper using an LLM-based extraction prompt.

    Extracted entities are later used as nodes in the knowledge graph.

    Attributes:
        model: LLM model identifier used for entity extraction.
        entity_types: Categories of entities to extract.
    """

    DEFAULT_ENTITY_TYPES = ["concept", "method", "dataset", "task", "metric"]

    def __init__(self, model: str = "gpt-4o-mini", entity_types: List[str] = None) -> None:
        """
        Initialize the entity extractor.

        Args:
            model: LLM model used for extraction prompts.
            entity_types: Entity categories to extract. Defaults to DEFAULT_ENTITY_TYPES.
        """
        self.model = model
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES

    def extract(self, papers: List[Paper]) -> Dict[str, List[str]]:
        """
        Extract entities from all papers and return a mapping of paper_id
        to its list of extracted entity strings.

        Args:
            papers: Papers whose texts will be processed.

        Returns:
            Dict[str, List[str]]: Mapping of paper_id -> list of entity labels.
        """
        pass

    def extract_from_paper(self, paper: Paper) -> List[str]:
        """
        Extract entities from a single paper's title and abstract.

        Args:
            paper: The paper to process.

        Returns:
            List[str]: Extracted entity labels found in the paper.
        """
        pass

    def _build_extraction_prompt(self, text: str) -> str:
        """
        Build the LLM prompt used to extract entities from a text snippet.

        Args:
            text: Concatenated title and abstract of a paper.

        Returns:
            str: Formatted extraction prompt.
        """
        pass

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse the raw LLM output into a clean list of entity strings.

        Args:
            response: Raw text response from the LLM.

        Returns:
            List[str]: Cleaned and deduplicated entity labels.
        """
        pass
