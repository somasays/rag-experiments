"""
Token-based chunking strategy.

This module implements chunking based on token count, which provides
the MOST CONSISTENT chunk sizes of all strategies. Essential for
controlled experiments where you need to isolate the effect of
chunk size from chunking strategy.

Why use token chunking?
=======================
1. PRECISE SIZE CONTROL: Unlike sentence chunking (which ignores
   chunk_size), token chunking produces chunks very close to the
   requested size.

2. FAIR COMPARISONS: When comparing strategies, use token chunking
   as your baseline - it's the only strategy that respects size config.

3. LLM CONTEXT: When your LLM has a token limit, token-based chunking
   maps directly to that constraint.

Our experiments found:
    - Requested: 1024 chars → Actual: ~934 chars (0.91x) - ACCURATE!
    - Compare to sentence: 1024 → 3677 (3.6x) - LIES!

Trade-offs:
    - May break mid-sentence or mid-word
    - Less semantic coherence than sentence chunking
    - But: this is a fair trade for controlled experiments

Key characteristics:
    - Fixed number of tokens per chunk
    - Most consistent chunk sizes (essential for experiments)
    - Good baseline for comparing other strategies
    - Respects your configuration (unlike sentence chunking!)
"""

from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode

from .base import ChunkingStrategy


class TokenChunker(ChunkingStrategy):
    """
    Chunk documents by token count.

    This strategy uses LlamaIndex's TokenTextSplitter to create chunks
    with a consistent number of tokens. Unlike character-based splitting,
    this ensures each chunk contains approximately the same amount of
    semantic content.

    Note: This strategy may break text mid-sentence or even mid-word,
    which can affect retrieval quality but provides the most consistent
    chunk sizes for controlled experiments.

    Attributes:
        chunk_size: Number of tokens per chunk (or derived from char target)
        chunk_overlap: Number of overlapping tokens between chunks
        token_chunk_size: Explicit token count (overrides chunk_size)

    Example:
        >>> chunker = TokenChunker(chunk_size=256, chunk_overlap=50)
        >>> nodes = chunker.chunk(documents)
        >>> print(f"Created {len(nodes)} chunks")

        # Or with explicit token size:
        >>> chunker = TokenChunker(token_chunk_size=500)
        >>> nodes = chunker.chunk(documents)
    """

    name = "token"

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        token_chunk_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the token chunker.

        Args:
            chunk_size: Target chunk size in characters. If token_chunk_size
                is not provided, this is converted to tokens using ~4 chars/token.
            chunk_overlap: Overlap in characters (converted to tokens).
            token_chunk_size: Explicit number of tokens per chunk. If provided,
                this overrides the chunk_size parameter.
            **kwargs: Additional arguments (unused, for interface compatibility)

        Note:
            The conversion ratio of ~4 characters per token is an approximation
            for English text. Actual token counts may vary by tokenizer.
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        # Convert character size to token count if not explicitly provided
        if token_chunk_size is not None:
            self._token_size = token_chunk_size
        else:
            # Approximate: ~4 characters per token for English text
            self._token_size = int(chunk_size / 4)

        # Calculate overlap in tokens (20% of chunk size, minimum 10)
        self._token_overlap = max(10, int(self._token_size * 0.2))

        # Initialize the underlying splitter
        self._splitter = TokenTextSplitter(
            chunk_size=self._token_size,
            chunk_overlap=self._token_overlap,
        )

    def chunk(self, documents: list[LlamaDocument]) -> list[TextNode]:
        """
        Chunk documents using token-based splitting.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects, each containing approximately
            `token_chunk_size` tokens

        Example:
            >>> chunker = TokenChunker(token_chunk_size=256)
            >>> docs = [Document(text="Long document text...")]
            >>> nodes = chunker.chunk(docs)
            >>> print(f"Average chunk size: {sum(len(n.text) for n in nodes) / len(nodes):.0f} chars")
        """
        return self._splitter.get_nodes_from_documents(documents)

    @property
    def token_size(self) -> int:
        """Return the actual token size used for chunking."""
        return self._token_size

    def __repr__(self) -> str:
        return (
            f"TokenChunker(token_size={self._token_size}, "
            f"token_overlap={self._token_overlap})"
        )
