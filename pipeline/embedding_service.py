"""Shared embedding service used across pipeline stages."""

from __future__ import annotations

from threading import Lock
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


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

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

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
        EmbeddingService._embedded_texts += 1
        return self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into dense vectors."""
        if not texts:
            return np.empty((0, 0), dtype=float)
        EmbeddingService._encode_calls += 1
        EmbeddingService._embedded_texts += len(texts)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @classmethod
    def get_metrics(cls) -> Dict[str, int]:
        """Return cumulative embedding usage counters."""
        return {
            "embedding_calls": cls._encode_calls,
            "embedded_texts": cls._embedded_texts,
        }

    @classmethod
    def reset_metrics(cls) -> None:
        """Reset cumulative embedding usage counters."""
        cls._encode_calls = 0
        cls._embedded_texts = 0
