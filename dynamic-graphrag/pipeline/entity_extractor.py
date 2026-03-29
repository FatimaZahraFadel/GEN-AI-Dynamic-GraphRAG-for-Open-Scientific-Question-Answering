"""
Stage 4 — Entity Extractor: extract entities and relationships from paper abstracts.

For each Paper, the abstract is sent to a Groq LLM which returns a JSON object
containing a list of entities and a list of directed relations between them.
The results are converted into Entity and Relation dataclass instances that
feed directly into the graph-building stage.
"""

import json
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq

from models.graph_node import Entity, Relation
from models.paper import Paper
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "Crop / Organism",
    "Disease / Condition",
    "Treatment / Method",
    "Environment / Location",
    "Cause / Factor",
    "Effect / Outcome",
]

RELATION_TYPES = [
    "affects",
    "treats",
    "occurs_in",
    "caused_by",
    "leads_to",
    "detected_by",
    "studied_in",
]


def _slugify(text: str) -> str:
    """
    Convert a label to a lowercase, underscore-separated slug.

    All non-alphanumeric characters (except hyphens) are replaced by
    underscores; leading/trailing underscores are stripped; runs of
    multiple underscores are collapsed.

    Parameters
    ----------
    text : str
        Raw label string, e.g. ``"Rust Disease"``.

    Returns
    -------
    str
        Slug string, e.g. ``"rust_disease"``.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


class EntityExtractor:
    """
    Extracts scientific entities and their relationships from paper abstracts
    using a Groq LLM (llama3-8b-8192).

    The extractor sends each abstract through a structured JSON-extraction
    prompt.  Responses are parsed safely; papers whose LLM call fails or
    returns malformed JSON are skipped with a warning.

    Attributes
    ----------
    model : str
        Groq model identifier. Defaults to ``"llama3-8b-8192"``.
    _client : Groq
        Authenticated Groq API client (lazy-initialised on first use).
    """

    def __init__(self, model: str = "llama3-8b-8192") -> None:
        """
        Initialise the entity extractor.

        Parameters
        ----------
        model : str
            Groq model to use for extraction. Defaults to
            ``"llama3-8b-8192"``.
        """
        self.model = model
        self._client: Groq | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, papers: List[Paper]
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from all papers.

        Calls :meth:`extract_from_abstract` for each paper, deduplicates
        entities by ``id`` (keeping the first occurrence), and collects all
        relations.

        Parameters
        ----------
        papers : list[Paper]
            Filtered papers from the previous pipeline stage.

        Returns
        -------
        tuple[list[Entity], list[Relation]]
            ``(all_entities, all_relations)`` — deduplicated entities and the
            full set of relations across all papers.
        """
        all_entities: List[Entity] = []
        all_relations: List[Relation] = []
        seen_entity_ids: set[str] = set()

        for paper in papers:
            entities, relations = self.extract_from_abstract(paper)

            for entity in entities:
                if entity.id not in seen_entity_ids:
                    all_entities.append(entity)
                    seen_entity_ids.add(entity.id)

            all_relations.extend(relations)

        logger.info(
            f"Extraction complete: {len(all_entities)} unique entities, "
            f"{len(all_relations)} relations across {len(papers)} papers."
        )
        return all_entities, all_relations

    def extract_from_abstract(
        self, paper: Paper
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a single paper's abstract.

        Sends the abstract to the Groq LLM via :meth:`build_extraction_prompt`,
        parses the JSON response, and converts raw dicts into :class:`Entity`
        and :class:`Relation` objects.

        If the LLM call fails or returns unparseable JSON, the error is logged
        and ``([], [])`` is returned so the pipeline can continue.

        Parameters
        ----------
        paper : Paper
            The paper to process.

        Returns
        -------
        tuple[list[Entity], list[Relation]]
            Entities and relations extracted from this paper's abstract.
        """
        if not paper.abstract or len(paper.abstract.strip()) < 50:
            logger.warning(
                f"Skipping paper '{paper.paper_id}': abstract too short."
            )
            return [], []

        prompt = self.build_extraction_prompt(paper.abstract)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(
                f"Groq API error for paper '{paper.paper_id}': {e}"
            )
            return [], []

        # Parse JSON — strip any accidental markdown fences first
        cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error for paper '{paper.paper_id}': {e}\n"
                f"Raw response: {raw_text[:300]}"
            )
            return [], []

        entities = self._parse_entities(data.get("entities", []), paper.paper_id)
        relations = self._parse_relations(
            data.get("relations", []), paper.paper_id
        )

        logger.info(
            f"Paper '{paper.paper_id[:30]}': "
            f"{len(entities)} entities, {len(relations)} relations extracted."
        )
        return entities, relations

    def build_extraction_prompt(self, abstract: str) -> str:
        """
        Build the structured JSON-extraction prompt for the LLM.

        The prompt lists the allowed entity types and relation types and
        instructs the model to return **only** a raw JSON object — no
        markdown, no explanation.

        Parameters
        ----------
        abstract : str
            The paper abstract to analyse.

        Returns
        -------
        str
            Fully formatted prompt string.
        """
        entity_types_str = ", ".join(f'"{t}"' for t in ENTITY_TYPES)
        relation_types_str = ", ".join(f'"{t}"' for t in RELATION_TYPES)

        return f"""You are a scientific information extraction system.

Extract entities and relationships from the abstract below.

Allowed entity types: {entity_types_str}
Allowed relation types: {relation_types_str}

Return ONLY a valid JSON object in this exact format — no markdown, no explanation, no extra text:
{{
  "entities": [
    {{"label": "Wheat", "type": "Crop / Organism"}},
    {{"label": "Rust", "type": "Disease / Condition"}}
  ],
  "relations": [
    {{"source": "Wheat", "target": "Rust", "relation": "affects"}}
  ]
}}

Abstract:
{abstract}"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_entities(
        self, raw_entities: list, paper_id: str
    ) -> List[Entity]:
        """
        Convert raw entity dicts from the LLM JSON into Entity objects.

        Entries missing a ``label`` field are silently skipped.  The ``type``
        field is accepted as-is; if absent it defaults to ``"Unknown"``.

        Parameters
        ----------
        raw_entities : list
            List of ``{"label": ..., "type": ...}`` dicts from the LLM.
        paper_id : str
            ID of the source paper (attached to every Entity).

        Returns
        -------
        list[Entity]
            Parsed and slugified Entity objects.
        """
        entities: List[Entity] = []
        for item in raw_entities:
            label = (item.get("label") or "").strip()
            if not label:
                continue
            entities.append(
                Entity(
                    id=_slugify(label),
                    label=label,
                    type=(item.get("type") or "Unknown").strip(),
                    source_paper_id=paper_id,
                )
            )
        return entities

    def _parse_relations(
        self, raw_relations: list, paper_id: str
    ) -> List[Relation]:
        """
        Convert raw relation dicts from the LLM JSON into Relation objects.

        Entries missing ``source``, ``target``, or ``relation`` are skipped.

        Parameters
        ----------
        raw_relations : list
            List of ``{"source": ..., "target": ..., "relation": ...}`` dicts.
        paper_id : str
            ID of the source paper (attached to every Relation).

        Returns
        -------
        list[Relation]
            Parsed Relation objects with slugified source/target IDs.
        """
        relations: List[Relation] = []
        for item in raw_relations:
            source = (item.get("source") or "").strip()
            target = (item.get("target") or "").strip()
            relation_type = (item.get("relation") or "").strip()
            if not (source and target and relation_type):
                continue
            relations.append(
                Relation(
                    source_id=_slugify(source),
                    target_id=_slugify(target),
                    relation_type=relation_type,
                    source_paper_id=paper_id,
                )
            )
        return relations

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
# Full pipeline smoke-test: DomainDetector -> Retriever -> Filter -> Extractor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io, warnings
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    from pipeline.domain_detector import DomainDetector
    from pipeline.paper_retriever import PaperRetriever
    from pipeline.paper_filter import PaperFilter

    QUESTION = "What fungal diseases affect wheat crops in humid environments?"
    TOP_K = 3  # keep small to limit LLM calls during testing

    print()
    print("=" * 70)
    print(f"  Question : {QUESTION}")
    print("=" * 70)

    # Stage 1 — domain detection
    domain = DomainDetector().classify(QUESTION, method="keyword")
    print(f"\n  [1] Domain detected : {domain}")

    # Stage 2 — paper retrieval
    retriever = PaperRetriever(top_n=20)
    raw_papers = retriever.retrieve(QUESTION, domain)
    print(f"  [2] Papers retrieved: {len(raw_papers)}")

    # Stage 3 — filter + rank
    filtered = PaperFilter().filter(raw_papers, QUESTION, top_k=TOP_K)
    print(f"  [3] Papers after filter: {len(filtered)}")

    # Stage 4 — entity extraction
    extractor = EntityExtractor()
    all_entities, all_relations = extractor.extract(filtered)

    # Print results grouped by paper
    print()
    print("=" * 70)
    print("  EXTRACTED ENTITIES & RELATIONS (grouped by paper)")
    print("=" * 70)

    for paper in filtered:
        paper_entities = [e for e in all_entities if e.source_paper_id == paper.paper_id]
        paper_relations = [r for r in all_relations if r.source_paper_id == paper.paper_id]

        print(f"\n  Paper : {paper.title[:65]}")
        print(f"  ID    : {paper.paper_id[:50]}")
        print(f"  Score : {paper.relevance_score:.4f}")

        if paper_entities:
            print(f"\n  Entities ({len(paper_entities)}):")
            for e in paper_entities:
                print(f"    - [{e.type}]  {e.label}  (id: {e.id})")
        else:
            print("  Entities: none extracted")

        if paper_relations:
            print(f"\n  Relations ({len(paper_relations)}):")
            for r in paper_relations:
                print(f"    - {r.source_id}  --[{r.relation_type}]-->  {r.target_id}")
        else:
            print("  Relations: none extracted")

        print()

    print("=" * 70)
    print(f"  TOTAL: {len(all_entities)} unique entities, {len(all_relations)} relations")
    print("=" * 70)
