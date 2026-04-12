"""Session-scoped state/cache for incremental GraphRAG execution."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config.settings import (
    FOLLOWUP_SKIP_RETRIEVAL_SIMILARITY,
    SESSION_CACHE_TTL_SECONDS,
    SESSION_QUERY_SIMILARITY_THRESHOLD,
)
from models.graph_node import Entity, Relation
from models.paper import Paper

# Max entries kept in bounded session caches.  Prevents unbounded memory growth
# during long-running Streamlit sessions with many queries.
_MAX_PAPERS_CACHE = 500
_MAX_EXTRACTION_CACHE = 300


@dataclass
class SessionState:
    """Holds reusable per-session artifacts for follow-up queries."""

    domain: Optional[str] = None
    graph: Optional[nx.DiGraph] = None
    # OrderedDict gives FIFO eviction when max size is reached.
    papers_by_id: OrderedDict = field(default_factory=OrderedDict)
    extraction_cache: OrderedDict = field(default_factory=OrderedDict)
    query_embeddings: List[np.ndarray] = field(default_factory=list)
    query_texts: List[str] = field(default_factory=list)
    query_timestamps: List[float] = field(default_factory=list)
    last_similarity: float = 0.0

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def is_expired(self) -> bool:
        """Return True when session cache is stale based on TTL."""
        if not self.query_timestamps:
            return False
        return (time.time() - self.query_timestamps[-1]) > SESSION_CACHE_TTL_SECONDS

    def should_reuse_graph(self, query_embedding: np.ndarray) -> bool:
        """Decide whether existing graph can be reused for the new query."""
        if self.graph is None or not self.query_embeddings:
            self.last_similarity = 0.0
            return False

        last_embedding = self.query_embeddings[-1]
        similarity = self._cosine(query_embedding, last_embedding)
        self.last_similarity = similarity
        return (not self.is_expired()) and similarity >= SESSION_QUERY_SIMILARITY_THRESHOLD

    def should_skip_retrieval(self) -> bool:
        """Return True when similarity is high enough to skip fresh retrieval."""
        return self.last_similarity >= FOLLOWUP_SKIP_RETRIEVAL_SIMILARITY and bool(self.papers_by_id)

    def record_query(self, query: str, query_embedding: np.ndarray) -> None:
        """Append current query metadata to the session history."""
        self.query_texts.append(query)
        self.query_embeddings.append(query_embedding)
        self.query_timestamps.append(time.time())

    def cache_papers(self, papers: List[Paper]) -> None:
        """Upsert papers into the bounded paper cache by ``paper_id``."""
        for paper in papers:
            if paper.paper_id:
                # Move to end (most-recently-used) if already present.
                self.papers_by_id.pop(paper.paper_id, None)
                self.papers_by_id[paper.paper_id] = paper
                if len(self.papers_by_id) > _MAX_PAPERS_CACHE:
                    self.papers_by_id.popitem(last=False)

    def get_cached_papers(self) -> List[Paper]:
        """Return all cached papers as a list."""
        return list(self.papers_by_id.values())

    def cache_extraction(
        self,
        key: str,
        value: Tuple[List[Entity], List[Relation]],
    ) -> None:
        """Insert an extraction result into the bounded extraction cache."""
        self.extraction_cache.pop(key, None)
        self.extraction_cache[key] = value
        if len(self.extraction_cache) > _MAX_EXTRACTION_CACHE:
            self.extraction_cache.popitem(last=False)

    def get_extraction(
        self, key: str
    ) -> Optional[Tuple[List[Entity], List[Relation]]]:
        """Retrieve a cached extraction result, or None if not present."""
        return self.extraction_cache.get(key)
