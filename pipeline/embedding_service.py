"""Shared embedding service used across pipeline stages."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of text embeddings held in memory.  Oldest entries are evicted
# once this limit is reached (FIFO / LRU-insert-order eviction via OrderedDict).
_EMBEDDING_CACHE_MAX_SIZE = 2000


class EmbeddingService:
    """
    Singleton wrapper around ``SentenceTransformer``.

    This avoids loading the same embedding model multiple times across
    ``DomainDetector``, ``PaperFilter``, and ``GraphRetriever``.
    """

    _instances: Dict[str, "EmbeddingService"] = {}
    _lock = Lock()
    _encode_calls: int = 0
    _embedded_texts: int = 0
    _cache_hits: int = 0
    _cache_misses: int = 0

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # OrderedDict gives O(1) move-to-end and FIFO eviction.
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = Lock()

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _cache_put(self, text: str, emb: np.ndarray) -> None:
        """Insert *text → emb* into the cache, evicting the oldest entry when full."""
        self._text_cache[text] = np.asarray(emb, dtype=float)
        if len(self._text_cache) > _EMBEDDING_CACHE_MAX_SIZE:
            self._text_cache.popitem(last=False)  # evict oldest

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL) -> "EmbeddingService":
        """Return a process-wide singleton for the requested model."""
        with cls._lock:
            if model_name not in cls._instances:
                logger.info("Loading shared SentenceTransformer model '%s'…", model_name)
                cls._instances[model_name] = cls(model_name=model_name)
            return cls._instances[model_name]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one text into a dense vector."""
        EmbeddingService._encode_calls += 1
        with self._cache_lock:
            cached = self._text_cache.get(text)
        if cached is not None:
            EmbeddingService._cache_hits += 1
            return np.asarray(cached, dtype=float).copy()

        EmbeddingService._cache_misses += 1
        EmbeddingService._embedded_texts += 1
        embedded = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        with self._cache_lock:
            self._cache_put(text, embedded)
        return np.asarray(embedded, dtype=float).copy()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into dense vectors."""
        if not texts:
            return np.empty((0, 0), dtype=float)
        EmbeddingService._encode_calls += 1

        with self._cache_lock:
            cached_rows = [self._text_cache.get(text) for text in texts]

        missing_indices = [idx for idx, row in enumerate(cached_rows) if row is None]
        if not missing_indices:
            EmbeddingService._cache_hits += len(texts)
            return np.vstack([np.asarray(row, dtype=float) for row in cached_rows])

        missing_texts = [texts[idx] for idx in missing_indices]
        EmbeddingService._cache_misses += len(missing_texts)
        EmbeddingService._embedded_texts += len(missing_texts)
        missing_embs = self._model.encode(missing_texts, convert_to_numpy=True, show_progress_bar=False)

        with self._cache_lock:
            for idx, emb in zip(missing_indices, missing_embs):
                self._cache_put(texts[idx], emb)
                cached_rows[idx] = self._text_cache[texts[idx]]

        return np.vstack([np.asarray(row, dtype=float) for row in cached_rows])

    @classmethod
    def get_metrics(cls) -> Dict[str, int]:
        """Return cumulative embedding usage counters."""
        return {
            "embedding_calls": cls._encode_calls,
            "embedded_texts": cls._embedded_texts,
            "cache_hits": cls._cache_hits,
            "cache_misses": cls._cache_misses,
        }

    @classmethod
    def reset_metrics(cls) -> None:
        """Reset cumulative embedding usage counters."""
        cls._encode_calls = 0
        cls._embedded_texts = 0
        cls._cache_hits = 0
        cls._cache_misses = 0
