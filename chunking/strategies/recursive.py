"""
Recursive character-based chunking strategy.

This module implements the LangChain RecursiveCharacterTextSplitter,
which is the INDUSTRY STANDARD approach to text chunking.

Why use recursive chunking?
===========================
1. INDUSTRY STANDARD: This is what most production RAG systems use.
   LangChain recommends it as the default.

2. SEMANTIC BOUNDARIES: Tries to split at natural boundaries (paragraphs,
   sentences) rather than arbitrary character positions.

3. BETTER THAN SENTENCE: At least it TRIES to respect chunk_size,
   unlike sentence chunking which ignores it entirely.

How it works:
=============
The splitter uses a hierarchy of separators:
    1. "\\n\\n" - Paragraph boundaries (preferred)
    2. "\\n"   - Line breaks
    3. ". "    - Sentence boundaries
    4. " "     - Word boundaries
    5. ""      - Character level (last resort)

It tries each separator in order, only falling back to the next if
the current separator doesn't produce small enough chunks.

IMPORTANT: Still undershoots!
=============================
Our experiments found:
    - Requested: 1024 chars → Actual: ~667 chars (0.65x)
    - Undershoots by ~35% to avoid breaking sentences

This is BETTER than sentence chunking (3.6x overshoot) but still
doesn't give you precise control. For exact sizes, use token chunking.

Trade-offs:
    - UNDERSHOOTS target size (0.65x) to preserve sentence boundaries
    - Good balance of size control and semantic coherence
    - More predictable than sentence, less precise than token
    - Won't break mid-sentence if possible

Key characteristics:
    - Uses hierarchical separators
    - Prefers natural text boundaries
    - Industry standard approach
    - Good default for production RAG (but know the size variance)
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LlamaDocument
from llama_index.core.schema import TextNode

from .base import ChunkingStrategy


class RecursiveChunker(ChunkingStrategy):
    """
    Chunk documents using hierarchical recursive splitting.

    This strategy uses LangChain's RecursiveCharacterTextSplitter,
    which attempts to split text using a hierarchy of separators:

    1. Double newline (\\n\\n) - paragraph boundaries
    2. Single newline (\\n) - line breaks
    3. Period followed by space (. ) - sentence boundaries
    4. Space ( ) - word boundaries
    5. Empty string - character level (last resort)

    The splitter tries each separator in order, only moving to the
    next if the current one doesn't produce chunks small enough.

    WARNING: Undershoots chunk_size!
    ================================
    Our experiments found: 1024 requested → ~667 actual (0.65x)

    This is because it won't break mid-sentence, so it often returns
    chunks smaller than requested. Still better than sentence chunking
    (which overshoots by 3.6x), but not as precise as token chunking.

    Attributes:
        chunk_size: Target chunk size in characters (actual ~0.65x this)
        chunk_overlap: Overlap between chunks in characters

    Example:
        >>> chunker = RecursiveChunker(chunk_size=1024, chunk_overlap=128)
        >>> nodes = chunker.chunk(documents)
        >>> # Actual chunks will be ~667 chars, not 1024!
        >>> avg_size = sum(len(n.text) for n in nodes) / len(nodes)
        >>> print(f"Requested 1024, got {avg_size:.0f}")

    Note:
        Industry standard, but know the size variance.
        For precise control, use TokenChunker instead.
    """

    name = "recursive"

    # Default separator hierarchy (LangChain standard)
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separators: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separator hierarchy. If None, uses default:
                ["\\n\\n", "\\n", ". ", " ", ""]
            **kwargs: Additional arguments (unused, for interface compatibility)
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        self._separators = separators or self.DEFAULT_SEPARATORS

        # Initialize the underlying LangChain splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._separators,
        )

    def chunk(self, documents: list[LlamaDocument]) -> list[TextNode]:
        """
        Chunk documents using recursive hierarchical splitting.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects, split at natural boundaries
            where possible

        Example:
            >>> chunker = RecursiveChunker(chunk_size=512)
            >>> docs = [Document(text="Paragraph one.\\n\\nParagraph two.")]
            >>> nodes = chunker.chunk(docs)
            >>> # Will prefer to split at paragraph boundary (\\n\\n)
        """
        nodes = []

        for doc_idx, doc in enumerate(documents):
            # Use LangChain splitter to split text
            chunks = self._splitter.split_text(doc.text)

            # Convert to LlamaIndex TextNode format
            for chunk_idx, chunk in enumerate(chunks):
                node = TextNode(
                    text=chunk,
                    id_=f"doc_{doc_idx}_chunk_{chunk_idx}",
                    metadata=doc.metadata.copy() if doc.metadata else {},
                )
                nodes.append(node)

        return nodes

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"separators={self._separators})"
        )
