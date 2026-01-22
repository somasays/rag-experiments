"""
Chunking Strategy Registry

This module provides a registry of all available chunking strategies.
Each strategy implements the ChunkingStrategy interface.

Available Strategies:
    - token: Fixed token count per chunk (may break mid-sentence)
    - sentence: Respects sentence boundaries (LlamaIndex default)
    - recursive: Hierarchical separators (LangChain industry standard)
    - semantic: Embedding-based semantic boundaries

Usage:
    >>> from chunking.strategies import STRATEGIES, get_strategy
    >>>
    >>> # Get strategy by name
    >>> chunker = get_strategy("token", chunk_size=256)
    >>>
    >>> # Or instantiate directly
    >>> from chunking.strategies import TokenChunker
    >>> chunker = TokenChunker(chunk_size=256, chunk_overlap=50)
    >>>
    >>> # Chunk documents
    >>> nodes = chunker.chunk(documents)

Extending:
    To add a custom strategy:
    1. Create a new file in chunking/strategies/
    2. Implement the ChunkingStrategy interface
    3. Register it in this __init__.py:

    >>> from .my_strategy import MyCustomChunker
    >>> STRATEGIES["my_custom"] = MyCustomChunker
"""

from typing import Any

from .base import ChunkingStrategy
from .token import TokenChunker
from .sentence import SentenceChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

# Strategy Registry
# Maps strategy name -> strategy class
STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    "token": TokenChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}


def get_strategy(
    name: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    **kwargs: Any,
) -> ChunkingStrategy:
    """
    Factory function to get a chunking strategy by name.

    Args:
        name: Strategy name (token, sentence, recursive, semantic)
        chunk_size: Target chunk size (interpretation varies by strategy)
        chunk_overlap: Overlap between chunks
        **kwargs: Additional strategy-specific arguments

    Returns:
        Instantiated ChunkingStrategy

    Raises:
        ValueError: If strategy name is not recognized

    Example:
        >>> chunker = get_strategy("token", chunk_size=256)
        >>> nodes = chunker.chunk(documents)
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )

    strategy_class = STRATEGIES[name]
    return strategy_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)


def list_strategies() -> list[dict[str, str]]:
    """
    List all available chunking strategies with descriptions.

    Returns:
        List of dicts with 'name' and 'description' keys

    Example:
        >>> for s in list_strategies():
        ...     print(f"{s['name']}: {s['description']}")
    """
    return [
        {
            "name": name,
            "description": cls.__doc__.split("\n")[0] if cls.__doc__ else "No description",
        }
        for name, cls in STRATEGIES.items()
    ]


__all__ = [
    "ChunkingStrategy",
    "TokenChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
]
