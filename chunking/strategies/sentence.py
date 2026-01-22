"""
Sentence-boundary respecting chunking strategy.

This module implements chunking that respects sentence boundaries,
which is the default behavior in LlamaIndex. This approach preserves
semantic coherence by ensuring chunks end at sentence boundaries.

WARNING: THE DIRTY SECRET
=========================
LlamaIndex's SentenceSplitter IGNORES the chunk_size parameter!

When you request chunk_size=1024, you actually get ~3500-4000 char chunks.
This is 3-4x larger than requested.

Evidence from our experiments:
    - Requested: 1024 chars → Actual: 3677 chars (3.6x)
    - Requested: 3000 chars → Actual: 8150 chars (2.7x)

This confounds every benchmark comparing "chunking strategies at the
same chunk size." Sentence chunking "wins" because it's using bigger
chunks, not because the strategy is better.

For controlled experiments, use token or recursive chunking instead.

Key characteristics:
    - Respects sentence boundaries (good for semantic coherence)
    - Chunks MUCH larger than requested (bad for controlled experiments)
    - LlamaIndex default chunking behavior
    - "Winning" benchmark results are misleading
"""

from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .base import ChunkingStrategy


class SentenceChunker(ChunkingStrategy):
    """
    Chunk documents by sentence boundaries.

    WARNING: chunk_size parameter is NOT respected!
    ===============================================
    LlamaIndex's SentenceSplitter produces chunks 3-4x LARGER than requested.

    Our experiments found:
        - chunk_size=1024 → actual ~3677 chars (3.6x larger)
        - chunk_size=3000 → actual ~8150 chars (2.7x larger)

    This makes sentence chunking "win" most benchmarks - not because the
    strategy is better, but because larger chunks have higher recall.

    When to use:
        - Quick prototypes where exact size doesn't matter
        - When you want semantic coherence over size control
        - NOT for controlled experiments comparing strategies

    When NOT to use:
        - Controlled experiments (use token or recursive instead)
        - When you need predictable chunk sizes
        - When comparing "same size" across strategies

    Attributes:
        chunk_size: Target chunk size in characters (WARNING: IGNORED!)
        chunk_overlap: Overlap between chunks in characters

    Example:
        >>> chunker = SentenceChunker(chunk_size=1024)  # Will NOT be 1024!
        >>> nodes = chunker.chunk(documents)
        >>> actual_size = sum(len(n.text) for n in nodes) / len(nodes)
        >>> print(f"Requested: 1024, Actual: {actual_size:.0f}")  # ~3500
    """

    name = "sentence"

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        """
        Initialize the sentence chunker.

        WARNING: chunk_size is passed to LlamaIndex but largely IGNORED.
        Actual chunks will be 3-4x larger than requested.

        Args:
            chunk_size: Target chunk size in characters.
                WARNING: This is NOT respected! Actual chunks will be
                3-4x larger. This parameter exists for interface
                compatibility only.
            chunk_overlap: Number of characters to overlap between
                consecutive chunks. Overlap occurs at sentence boundaries.
            **kwargs: Additional arguments (unused, for interface compatibility)

        Note:
            If you need controlled chunk sizes, use TokenChunker or
            RecursiveChunker instead. This chunker is only suitable
            when exact chunk size doesn't matter.
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        # Initialize the underlying splitter
        # NOTE: Despite passing chunk_size, LlamaIndex's SentenceSplitter
        # produces chunks 3-4x larger than requested. This is documented
        # behavior but not widely understood. The splitter prioritizes
        # sentence boundaries over size constraints.
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[LlamaDocument]) -> list[TextNode]:
        """
        Chunk documents respecting sentence boundaries.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects, each ending at a sentence boundary

        Example:
            >>> chunker = SentenceChunker(chunk_size=1024)
            >>> docs = [Document(text="First sentence. Second sentence. Third.")]
            >>> nodes = chunker.chunk(docs)
            >>> # Each node ends at a period (sentence boundary)
        """
        return self._splitter.get_nodes_from_documents(documents)

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
