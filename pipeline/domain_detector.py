"""
Stage 1 — Domain Detector: classify the scientific domain of a user query.

Supported domains: Agriculture, Geoscience, Computer Science, Supply Chain, Environment.
Three classification strategies are provided:
  - llm       : zero-shot classification via Groq (llama3-8b-8192)
  - keyword   : rule-based keyword matching with LLM fallback
  - embedding : cosine similarity against domain reference sentences
"""

import os
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq
import requests
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    DOMAINS,
    OLLAMA_BASE_URL,
    OLLAMA_GENERAL_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    USE_OLLAMA_PRIMARY,
)
from pipeline.embedding_service import EmbeddingService
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
    _embedding_service : EmbeddingService
        Shared sentence-transformer service reused across modules.
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
            # Mining, critical minerals, and resource extraction
            "lithium", "extraction", "brine", "geothermal", "mining", "ore",
            "rare earth", "pegmatite", "hydrometallurgy", "leaching", "smelting",
            "cobalt", "nickel", "copper", "zinc", "molybdenum", "tungsten",
            "critical mineral", "mineral resource", "reserve", "deposit",
        ],
        "Computer Science": [
            "machine learning", "deep learning", "underfitting", "overfitting",
            "neural network", "model", "training", "feature engineering",
            "regularization", "cross-validation", "gradient", "optimizer",
            "transformer", "llm", "classification", "regression",
            "computer vision", "nlp", "hyperparameter", "dataset",
        ],
        "Supply Chain": [
            "supply chain", "supply chains", "supply chain management",
            "logistics", "inventory", "supplier", "procurement", "warehouse",
            "distribution", "demand", "forecasting", "sourcing", "transport",
            "lead time", "order", "fulfilment", "vendor", "shipment",
            "delay", "delays", "disruption", "disruptions", "bottleneck",
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
            "mineral formation, sediment erosion, geological processes, lithium "
            "and rare-earth extraction, geothermal brines, mining, critical "
            "mineral resources, and hydrometallurgy."
        ),
        "Computer Science": (
            "Research on machine learning, deep learning, model training, "
            "optimization, algorithms, artificial intelligence, and data-driven systems."
        ),
        "Supply Chain": (
            "Analysis of supply chain management, logistics networks, inventory "
            "management, supplier procurement, warehouse distribution, demand "
            "forecasting, delays, disruptions, and bottlenecks."
        ),
        "Environment": (
            "Investigation of climate change, carbon emissions, air and water "
            "pollution, biodiversity loss, and deforestation impacts."
        ),
    }

    _SUBDOMAIN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
        "Agriculture": {
            "plant pathology": ["disease", "pathogen", "fungal", "blight", "rust", "pest"],
            "crop management": ["fertilizer", "irrigation", "yield", "tillage", "harvest"],
            "soil systems": ["soil", "rhizosphere", "microbiome", "organic carbon"],
        },
        "Geoscience": {
            "cosmology": ["universe", "cosmology", "dark matter", "galaxy", "big bang"],
            "earth systems": ["climate", "atmosphere", "ocean", "hydrology", "groundwater"],
            "solid earth": ["earthquake", "tectonic", "fault", "volcano", "stratigraphy"],
        },
        "Computer Science": {
            "machine learning": ["machine learning", "model", "training", "underfitting", "overfitting"],
            "deep learning": ["deep learning", "neural network", "transformer", "representation", "backpropagation"],
            "data and optimization": ["feature", "regularization", "optimizer", "gradient", "hyperparameter"],
        },
        "Supply Chain": {
            "demand planning": ["demand", "forecast", "planning", "inventory", "stockout"],
            "operations logistics": ["logistics", "warehouse", "shipment", "transport", "lead time",
                                     "supply chain", "delay", "bottleneck", "distribution", "freight"],
            "procurement risk": ["supplier", "procurement", "sourcing", "risk", "disruption",
                                 "shortage", "resilience", "contingency"],
        },
        "Environment": {
            "climate impacts": ["climate", "warming", "emissions", "carbon", "mitigation"],
            "pollution": ["pollution", "air quality", "water quality", "contaminant", "waste"],
            "ecosystems": ["biodiversity", "ecosystem", "habitat", "deforestation", "conservation"],
        },
    }

    def __init__(
        self,
        domains: List[str] = DOMAINS,
        model: str = "llama-3.1-8b-instant",
        embedding_service: EmbeddingService | None = None,
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
        self._embedding_service = (
            embedding_service or EmbeddingService.get_instance()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_robust(self, question: str) -> str:
        """
        Robust two-signal classification: keyword + embedding consensus.

        Strategy
        --------
        1. Run keyword classification to get a candidate domain and its
           confidence margin (difference between top-1 and top-2 scores).
        2. Run embedding classification for an independent signal.
        3. If both agree → return that domain.
        4. If they disagree and keyword margin is strong (≥ 2 hits ahead) →
           trust keyword (strong lexical signal beats embedding for specific
           technical terms like "lithium", "seismic").
        5. Otherwise → trust embedding (richer semantic reference sentences
           handle ambiguous or cross-domain queries better).
        6. Never fall back to a hardcoded default: if keyword returns no hits,
           embedding is used exclusively.

        This avoids the failure mode where a single generic word like "methods"
        gives Computer Science one keyword hit and wrongly wins over Geoscience.
        """
        q_lower = question.lower()

        # --- Keyword scores ---
        kw_scores: Dict[str, int] = {d: 0 for d in self.domains}
        for domain in self.domains:
            for kw in self._KEYWORDS.get(domain, []):
                if kw in q_lower:
                    kw_scores[domain] += 1

        sorted_kw = sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
        kw_winner, kw_top = sorted_kw[0]
        kw_second = sorted_kw[1][1] if len(sorted_kw) > 1 else 0
        kw_margin = kw_top - kw_second

        # --- Embedding score ---
        emb_winner = self.embedding_classify(question)

        # --- Consensus logic ---
        if kw_top == 0:
            # No keyword evidence at all — use embedding exclusively
            result = emb_winner
            logger.info(
                "classify_robust: no keyword hits → embedding='%s'", emb_winner
            )
        elif kw_winner == emb_winner:
            # Both agree
            result = kw_winner
            logger.info(
                "classify_robust: consensus='%s' (kw=%d, emb agrees)", kw_winner, kw_top
            )
        elif kw_margin >= 2:
            # Keyword has a strong lead — trust it (e.g. "lithium extraction" with
            # 5 Geoscience keyword hits vs 1 CS hit)
            result = kw_winner
            logger.info(
                "classify_robust: kw strong margin=%d → kw='%s' overrides emb='%s'",
                kw_margin, kw_winner, emb_winner,
            )
        else:
            # Weak or tied keyword signal — trust the richer embedding representation
            result = emb_winner
            logger.info(
                "classify_robust: weak kw margin=%d → emb='%s' overrides kw='%s'",
                kw_margin, emb_winner, kw_winner,
            )

        logger.info("classify_robust: '%s' → '%s'", question[:60], result)
        return result

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

    def classify_subdomain(self, question: str, domain: str) -> str:
        """
        Classify a finer-grained subdomain within the predicted top-level domain.

        Falls back to the detected top-level domain when confidence is low or no
        subdomain map exists for the given domain.
        """
        subdomain_map = self._SUBDOMAIN_KEYWORDS.get(domain, {})
        if not subdomain_map:
            return (domain or "general").lower()

        q = (question or "").lower()
        scores: Dict[str, int] = {sd: 0 for sd in subdomain_map}
        for subdomain, kws in subdomain_map.items():
            for kw in kws:
                if kw in q:
                    scores[subdomain] += 1

        best_subdomain = max(scores, key=lambda k: scores[k])
        if scores[best_subdomain] == 0:
            return (domain or "general").lower()
        return best_subdomain

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
        prompt = self._build_llm_prompt(question)

        raw = self._call_llm(prompt=prompt, max_tokens=16)
        return self._normalise_domain(raw)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call Ollama first when enabled, then fall back to Groq."""
        if USE_OLLAMA_PRIMARY:
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                    json={
                        "model": OLLAMA_GENERAL_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json() or {}
                text = (payload.get("response") or "").strip()
                if text:
                    return text
            except Exception as e:
                logger.warning("DomainDetector Ollama call failed, falling back to Groq: %s", e)

        client = self._get_groq_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

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
            logger.debug("No keyword match found; falling back to embedding classification.")
            return self.embedding_classify(question)

        # If keyword evidence is weak/close, prefer embedding disambiguation.
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) <= 1:
            return self.embedding_classify(question)

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
        # Embed question and all reference sentences in a single batch
        reference_domains = list(self._DOMAIN_REFERENCES.keys())
        reference_texts = [self._DOMAIN_REFERENCES[d] for d in reference_domains]

        all_texts = [question] + reference_texts
        embeddings = self._embedding_service.encode_batch(all_texts)

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
            "LLM returned unrecognised domain '%s'; falling back to embedding classification.",
            raw,
        )
        # Use embedding on the original query so the fallback is query-sensitive.
        # If the raw LLM output is too short to be meaningful, fall back to the
        # first configured domain rather than manufacturing a generic label.
        if len(raw) > 3:
            return self.embedding_classify(raw)
        return self.domains[0] if self.domains else "general"

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
