"""
Stage 1 — Domain Detector: classify the scientific domain of a user query.

Supported domains: Agriculture, Geoscience, Supply Chain, Environment.
Three classification strategies are provided:
  - llm       : zero-shot classification via Groq (llama3-8b-8192)
  - keyword   : rule-based keyword matching with LLM fallback
  - embedding : cosine similarity against domain reference sentences
"""

import os
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import DOMAINS
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class DomainDetector:
    """
    Classifies a natural-language scientific question into one of the
    supported scientific domains using three interchangeable strategies.

    Strategies
    ----------
    llm       — Groq chat-completion (llama3-8b-8192) zero-shot prompt.
    keyword   — Keyword-count heuristic; falls back to LLM on no match.
    embedding — Cosine similarity between query and per-domain reference
                sentences using sentence-transformers (all-MiniLM-L6-v2).

    Attributes
    ----------
    domains : list of str
        Candidate domain labels drawn from config.settings.DOMAINS.
    model : str
        Groq model identifier used for LLM-based classification.
    _groq_client : Groq
        Authenticated Groq API client (lazy-initialised on first LLM call).
    _embedder : SentenceTransformer | None
        Sentence-transformer model (lazy-initialised on first embedding call).
    """

    # --- keyword dictionaries ------------------------------------------------

    _KEYWORDS: Dict[str, List[str]] = {
        "Agriculture": [
            "crop", "soil", "wheat", "fungal", "irrigation", "fertilizer",
            "pest", "harvest", "farm", "livestock", "rice", "maize", "sowing",
            "agroforestry", "pesticide", "drought", "yield", "tillage",
        ],
        "Geoscience": [
            "earthquake", "volcano", "tectonic", "mineral", "sediment",
            "erosion", "geology", "seismic", "lithosphere", "mantle", "fault",
            "stratigraphy", "geomorphology", "magma", "plate", "groundwater",
        ],
        "Supply Chain": [
            "logistics", "inventory", "supplier", "procurement", "warehouse",
            "distribution", "demand", "forecasting", "sourcing", "transport",
            "lead time", "order", "fulfilment", "vendor", "shipment",
        ],
        "Environment": [
            "climate", "pollution", "carbon", "biodiversity", "emission",
            "deforestation", "greenhouse", "ecosystem", "habitat", "waste",
            "renewable", "sustainability", "ozone", "particulate", "acid rain",
        ],
    }

    # --- representative sentences for embedding strategy --------------------

    _DOMAIN_REFERENCES: Dict[str, str] = {
        "Agriculture": (
            "Research on crop yields, soil health, irrigation systems, "
            "fertilizer use, pest management, and sustainable farming practices."
        ),
        "Geoscience": (
            "Study of earthquakes, volcanic activity, tectonic plate movement, "
            "mineral formation, sediment erosion, and geological processes."
        ),
        "Supply Chain": (
            "Analysis of logistics networks, inventory management, supplier "
            "procurement, warehouse distribution, and demand forecasting."
        ),
        "Environment": (
            "Investigation of climate change, carbon emissions, air and water "
            "pollution, biodiversity loss, and deforestation impacts."
        ),
    }

    def __init__(
        self,
        domains: List[str] = DOMAINS,
        model: str = "llama3-8b-8192",
    ) -> None:
        """
        Initialise the domain detector.

        Parameters
        ----------
        domains : list of str
            Candidate domain labels. Defaults to config.settings.DOMAINS.
        model : str
            Groq model identifier. Defaults to ``llama3-8b-8192``.
        """
        self.domains = domains
        self.model = model
        self._groq_client: Groq | None = None
        self._embedder: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, question: str, method: str = "llm") -> str:
        """
        Main entry point — classify a question using the chosen strategy.

        Parameters
        ----------
        question : str
            Natural-language scientific question from the user.
        method : str
            Classification strategy: ``"llm"``, ``"keyword"``, or
            ``"embedding"``. Defaults to ``"llm"``.

        Returns
        -------
        str
            One of the domain labels from ``self.domains``.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three supported strategies.
        """
        if method == "llm":
            result = self.llm_classify(question)
        elif method == "keyword":
            result = self.keyword_classify(question)
        elif method == "embedding":
            result = self.embedding_classify(question)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: 'llm', 'keyword', 'embedding'."
            )

        logger.info(f"[{method}] classified '{question[:60]}...' → {result}")
        return result

    def llm_classify(self, question: str) -> str:
        """
        Classify the question via a Groq chat-completion zero-shot prompt.

        The model is instructed to return **only** the domain name and nothing
        else, so the raw completion text is used directly after stripping.

        Parameters
        ----------
        question : str
            Scientific question to classify.

        Returns
        -------
        str
            One of the domain labels from ``self.domains``.
        """
        client = self._get_groq_client()
        prompt = self._build_llm_prompt(question)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )

        raw = response.choices[0].message.content.strip()
        return self._normalise_domain(raw)

    def keyword_classify(self, question: str) -> str:
        """
        Classify the question by counting domain keyword matches.

        Each word in the lowercased question is checked against per-domain
        keyword lists.  The domain with the highest count wins.  If no
        keyword matches at all, the method falls back to :meth:`llm_classify`.

        Parameters
        ----------
        question : str
            Scientific question to classify.

        Returns
        -------
        str
            One of the domain labels from ``self.domains``.
        """
        question_lower = question.lower()
        scores: Dict[str, int] = {domain: 0 for domain in self.domains}

        for domain in self.domains:
            keywords = self._KEYWORDS.get(domain, [])
            for kw in keywords:
                if kw in question_lower:
                    scores[domain] += 1

        best_domain = max(scores, key=lambda d: scores[d])
        if scores[best_domain] == 0:
            logger.debug("No keyword match found; falling back to LLM classification.")
            return self.llm_classify(question)

        return best_domain

    def embedding_classify(self, question: str) -> str:
        """
        Classify the question by cosine similarity to domain reference sentences.

        A pre-trained ``all-MiniLM-L6-v2`` sentence-transformer model is used
        to embed both the question and one representative sentence per domain.
        The domain whose reference embedding is most similar to the question
        embedding is returned.

        The model is loaded lazily and cached for subsequent calls.

        Parameters
        ----------
        question : str
            Scientific question to classify.

        Returns
        -------
        str
            One of the domain labels from ``self.domains``.
        """
        embedder = self._get_embedder()

        # Embed question and all reference sentences in a single batch
        reference_domains = list(self._DOMAIN_REFERENCES.keys())
        reference_texts = [self._DOMAIN_REFERENCES[d] for d in reference_domains]

        all_texts = [question] + reference_texts
        embeddings = embedder.encode(all_texts, convert_to_numpy=True)

        query_embedding = embeddings[0:1]          # shape (1, dim)
        domain_embeddings = embeddings[1:]          # shape (n_domains, dim)

        similarities = cosine_similarity(query_embedding, domain_embeddings)[0]

        best_idx = int(similarities.argmax())
        return reference_domains[best_idx]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, question: str) -> str:
        """
        Build the zero-shot classification prompt for the LLM.

        The prompt lists the allowed domain names and explicitly instructs
        the model to respond with the domain name only.

        Parameters
        ----------
        question : str
            User question to embed in the prompt.

        Returns
        -------
        str
            Formatted prompt string ready to send to the API.
        """
        domain_list = ", ".join(self.domains)
        return (
            f"You are a scientific domain classifier.\n"
            f"Classify the following question into exactly one of these domains: "
            f"{domain_list}.\n"
            f"Reply with the domain name only — no explanation, no punctuation, "
            f"no extra words.\n\n"
            f"Question: {question}"
        )

    def _normalise_domain(self, raw: str) -> str:
        """
        Map a raw LLM output string to the closest canonical domain label.

        Matching is case-insensitive.  If the raw string does not contain
        any known domain, the first domain in ``self.domains`` is returned
        as a safe default and a warning is logged.

        Parameters
        ----------
        raw : str
            Raw text returned by the LLM.

        Returns
        -------
        str
            Canonical domain label from ``self.domains``.
        """
        raw_lower = raw.lower()
        for domain in self.domains:
            if domain.lower() in raw_lower:
                return domain

        logger.warning(
            f"LLM returned unrecognised domain '{raw}'; "
            f"defaulting to '{self.domains[0]}'."
        )
        return self.domains[0]

    def _get_groq_client(self) -> Groq:
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
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file."
                )
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _get_embedder(self) -> SentenceTransformer:
        """
        Return the cached SentenceTransformer model, loading it on first call.

        Returns
        -------
        SentenceTransformer
            Loaded sentence-transformer model (all-MiniLM-L6-v2).
        """
        if self._embedder is None:
            logger.debug("Loading SentenceTransformer model 'all-MiniLM-L6-v2'…")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "What are the effects of drought stress on wheat crop yield?",
        "How do tectonic plate movements trigger major earthquakes?",
        "What strategies can reduce carbon emissions from deforestation?",
    ]

    detector = DomainDetector()

    for q in test_questions:
        print(f"\nQuestion : {q}")
        for method in ("llm", "keyword", "embedding"):
            domain = detector.classify(q, method=method)
            print(f"  [{method:9s}] → {domain}")
