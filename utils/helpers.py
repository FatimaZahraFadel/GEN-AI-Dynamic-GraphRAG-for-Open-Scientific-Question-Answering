"""
General-purpose helper utilities for the Dynamic GraphRAG pipeline.
"""

import math
from typing import Any, Dict, List


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into fixed-size chunks.

    Args:
        lst: The list to partition.
        chunk_size: Maximum number of elements per chunk.

    Returns:
        List[List[Any]]: List of sub-lists, each at most chunk_size elements.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten(nested: List[List[Any]]) -> List[Any]:
    """
    Flatten one level of nesting from a list of lists.

    Args:
        nested: A list whose elements are themselves lists.

    Returns:
        List[Any]: Single flat list with all elements.
    """
    return [item for sublist in nested for item in sublist]


def deduplicate_by_key(items: List[Dict], key: str) -> List[Dict]:
    """
    Remove duplicate dictionaries that share the same value for a given key.
    The first occurrence is kept.

    Args:
        items: List of dictionaries to deduplicate.
        key: Dictionary key whose value is used for uniqueness.

    Returns:
        List[Dict]: Deduplicated list preserving insertion order.
    """
    seen = set()
    deduped: List[Dict] = []
    for item in items:
        value = item.get(key)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(item)
    return deduped


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute the cosine similarity between two dense vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        float: Cosine similarity in the range [-1, 1].
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def truncate_text(text: str, max_tokens: int, encoding: str = "cl100k_base") -> str:
    """
    Truncate text to a maximum number of tokens using tiktoken.

    Args:
        text: Input text to truncate.
        max_tokens: Maximum allowed number of tokens.
        encoding: Tiktoken encoding name. Defaults to "cl100k_base".

    Returns:
        str: Truncated text string.
    """
    if max_tokens <= 0:
        return ""
    if not text:
        return ""

    try:
        import tiktoken  # optional dependency

        enc = tiktoken.get_encoding(encoding)
        token_ids = enc.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return enc.decode(token_ids[:max_tokens])
    except Exception:
        # Fallback approximation if tiktoken is unavailable.
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])
