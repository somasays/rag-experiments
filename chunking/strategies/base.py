"""
Base class for chunking strategies.

This module defines the abstract interface that all chunking strategies must implement.
It ensures consistent behavior across different chunking approaches.
"""

from abc import ABC, abstractmethod
from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import TextNode


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.

    All chunking strategies must inherit from this class and implement
    the `chunk` method. This ensures a consistent interface across all
    strategies and enables easy extension.

    Attributes:
        name: Strategy identifier (e.g., "token", "sentence")
        chunk_size: Target chunk size (interpretation varies by strategy)
        chunk_overlap: Overlap between consecutive chunks

    Example:
        >>> class MyChunker(ChunkingStrategy):
        ...     name = "my_chunker"
        ...
        ...     def chunk(self, documents):
        ...         # Implementation here
        ...         return nodes
        >>>
        >>> chunker = MyChunker(chunk_size=512)
        >>> nodes = chunker.chunk(documents)
    """

    # Class attribute - override in subclasses
    name: str = "base"

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        **kwargs: Any,
    ):
        """
        Initialize the chunking strategy.

        Args:
            chunk_size: Target chunk size. Interpretation varies:
                - token: Number of tokens
                - sentence/recursive: Number of characters
                - semantic: Not directly used (determined by embeddings)
            chunk_overlap: Number of overlapping units between chunks
            **kwargs: Strategy-specific additional arguments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._extra_kwargs = kwargs

    @abstractmethod
    def chunk(self, documents: list[LlamaDocument]) -> list[TextNode]:
        """
        Chunk documents into smaller text nodes.

        This method must be implemented by all subclasses.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects (chunks)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement chunk()")

    def verify_chunking(
        self,
        documents: list[LlamaDocument],
        nodes: list[TextNode],
    ) -> dict[str, Any]:
        """
        Verify that chunking produced meaningful output.

        This method catches common experimental errors where documents
        are smaller than the chunk size, resulting in no actual chunking.

        Args:
            documents: Original documents before chunking
            nodes: Nodes/chunks after chunking

        Returns:
            Dict with chunking statistics:
                - doc_count: Number of input documents
                - chunk_count: Number of output chunks
                - chunks_per_doc: Average chunks per document
                - avg_doc_length: Average document length in characters
                - avg_chunk_length: Average chunk length in characters
                - chunking_occurred: True if chunks > documents

        Example:
            >>> stats = chunker.verify_chunking(documents, nodes)
            >>> if not stats["chunking_occurred"]:
            ...     print("Warning: No chunking occurred!")
        """
        doc_count = len(documents)
        chunk_count = len(nodes)

        # Calculate document lengths
        doc_lengths = [len(d.text) for d in documents]
        avg_doc_len = sum(doc_lengths) / doc_count if doc_count > 0 else 0

        # Calculate chunk lengths
        chunk_lengths = [len(n.text) for n in nodes]
        avg_chunk_len = sum(chunk_lengths) / chunk_count if chunk_count > 0 else 0

        chunking_occurred = chunk_count > doc_count

        if not chunking_occurred:
            print(f"WARNING: No chunking occurred for {self.name}!")
            print(f"  Documents: {doc_count}, Chunks: {chunk_count}")
            print(f"  Avg doc length: {avg_doc_len:.0f} chars")
            print("  Documents may be smaller than chunk size.")

        return {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "chunks_per_doc": chunk_count / doc_count if doc_count > 0 else 0,
            "avg_doc_length": round(avg_doc_len, 0),
            "avg_chunk_length": round(avg_chunk_len, 0),
            "chunking_occurred": chunking_occurred,
        }

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
