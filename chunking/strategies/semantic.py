"""
Semantic embedding-based chunking strategy.

This module implements chunking based on semantic similarity between
sentences, using embedding models to detect topic boundaries. This is
the most sophisticated chunking approach but has a critical limitation.

IMPORTANT: Chunk size is NOT controllable!
==========================================
Unlike other strategies, semantic chunking ignores chunk_size entirely.
Chunks are determined by where the embedding model detects topic shifts.

Our experiments found:
    - Regardless of chunk_size setting, produces ~1100 char chunks
    - This is NOT a bug - it's inherent to semantic chunking
    - You cannot force larger or smaller chunks

When to use:
============
1. Documents with clear topic shifts (news articles, textbooks)
2. When semantic coherence matters more than size control
3. Exploratory work where you want to see natural boundaries

When NOT to use:
================
1. Controlled experiments (chunk size confounds results)
2. When you need specific chunk sizes for LLM context
3. Production systems with strict size requirements

How it works:
=============
1. Splits document into sentences
2. Computes embedding for each sentence
3. Calculates similarity between adjacent sentence groups
4. Splits at points where similarity drops below threshold
5. Groups similar adjacent sentences into chunks

Cost consideration:
    - Requires embedding API calls during chunking (not just indexing)
    - More expensive than other strategies
    - Consider caching for repeated experiments

Key characteristics:
    - Uses embeddings to detect semantic boundaries
    - Chunks are semantically coherent
    - Chunk size is NOT controllable (~1100 chars typically)
    - Higher computational cost (requires embeddings)
    - Pre-splits very long documents to avoid context limits
"""

from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

from .base import ChunkingStrategy

# Maximum document length (chars) before pre-splitting
# ~20K chars ≈ 5K tokens, leaving headroom for 8192 token limit
MAX_DOC_LENGTH_FOR_SEMANTIC = 20000


class SemanticChunker(ChunkingStrategy):
    """
    Chunk documents based on semantic similarity.

    This strategy uses LlamaIndex's SemanticSplitterNodeParser to create
    chunks based on semantic boundaries detected using embedding similarity.
    Adjacent sentences with similar embeddings are grouped together, while
    splits occur at points of semantic shift.

    IMPORTANT: Unlike other strategies, chunk size is NOT directly
    controllable. The chunk_size parameter is ignored - chunks are
    determined entirely by semantic boundaries in the text.

    Attributes:
        embed_model: Embedding model for computing sentence similarities
        buffer_size: Number of sentences to compare (default: 3)
        breakpoint_percentile: Percentile threshold for splitting (default: 90)

    Example:
        >>> from llama_index.embeddings.openai import OpenAIEmbedding
        >>> embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        >>> chunker = SemanticChunker(embed_model=embed_model)
        >>> nodes = chunker.chunk(documents)

    Note:
        - Requires an embedding model (higher cost than other strategies)
        - Produces ~1100 char chunks on average (not controllable)
        - Best for documents with clear topic shifts
    """

    name = "semantic"

    def __init__(
        self,
        chunk_size: int = 1024,  # Ignored - kept for interface compatibility
        chunk_overlap: int = 128,  # Ignored - kept for interface compatibility
        embed_model: Any = None,
        buffer_size: int = 3,
        breakpoint_percentile: int = 90,
        **kwargs: Any,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Ignored. Kept for interface compatibility.
            chunk_overlap: Ignored. Kept for interface compatibility.
            embed_model: LlamaIndex embedding model for computing similarities.
                Required - will raise error if not provided.
            buffer_size: Number of sentences to compare for similarity.
                Higher values = more context but slower. Default: 3
            breakpoint_percentile: Percentile threshold for detecting breaks.
                Higher values = fewer, larger chunks. Default: 90
            **kwargs: Additional arguments

        Raises:
            ValueError: If embed_model is not provided

        Note:
            The chunk_size and chunk_overlap parameters are accepted for
            interface compatibility but are NOT used. Semantic chunking
            determines boundaries based on content, not target sizes.
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        if embed_model is None:
            raise ValueError(
                "SemanticChunker requires an embedding model. "
                "Pass embed_model parameter, e.g.:\n"
                "  from llama_index.embeddings.openai import OpenAIEmbedding\n"
                "  embed_model = OpenAIEmbedding(model='text-embedding-3-small')\n"
                "  chunker = SemanticChunker(embed_model=embed_model)"
            )

        self._embed_model = embed_model
        self._buffer_size = buffer_size
        self._breakpoint_percentile = breakpoint_percentile

        # Initialize the underlying splitter
        # buffer_size=3: compare triplets (avoids over-fragmentation)
        # breakpoint_percentile_threshold=90: only split at major topic shifts
        self._splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile,
        )

    def chunk(self, documents: list[LlamaDocument]) -> list[TextNode]:
        """
        Chunk documents based on semantic boundaries.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects, split at semantic boundaries

        Note:
            Chunk sizes will vary based on document content. Average
            chunk size is typically ~1100 characters, but individual
            chunks may range from a few hundred to several thousand
            characters depending on the semantic structure.

            For very long documents (>20K chars), we pre-split using
            sentence-based chunking to avoid exceeding embedding model
            context limits (8192 tokens for text-embedding-3-small).

        Example:
            >>> from llama_index.embeddings.openai import OpenAIEmbedding
            >>> embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            >>> chunker = SemanticChunker(embed_model=embed_model)
            >>> nodes = chunker.chunk(documents)
            >>> # Chunk sizes determined by semantic breaks, not target size
        """
        # Pre-split very long documents to avoid context length errors
        # The embedding model (text-embedding-3-small) has 8192 token limit
        processed_docs = []
        for doc in documents:
            if len(doc.text) > MAX_DOC_LENGTH_FOR_SEMANTIC:
                # Use SentenceSplitter to break into manageable sections
                pre_splitter = SentenceSplitter(
                    chunk_size=MAX_DOC_LENGTH_FOR_SEMANTIC // 4,  # ~5K chars
                    chunk_overlap=200,
                )
                # Create sub-documents from the pre-split nodes
                pre_nodes = pre_splitter.get_nodes_from_documents([doc])
                for i, node in enumerate(pre_nodes):
                    # Create new document preserving metadata
                    sub_doc = LlamaDocument(
                        text=node.text,
                        metadata={
                            **doc.metadata,
                            "pre_split_index": i,
                            "original_doc_id": doc.doc_id,
                        },
                    )
                    processed_docs.append(sub_doc)
            else:
                processed_docs.append(doc)

        return self._splitter.get_nodes_from_documents(processed_docs)

    def __repr__(self) -> str:
        return (
            f"SemanticChunker(buffer_size={self._buffer_size}, "
            f"breakpoint_percentile={self._breakpoint_percentile})"
        )
