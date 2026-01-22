"""
Base class for dataset loaders.

This module defines the abstract interface for dataset loaders and the
Query dataclass used throughout the chunking experiments.

Why Datasets Matter for RAG Experiments:
========================================
The choice of evaluation dataset significantly affects conclusions:

1. Document length determines chunking difficulty:
   - HotpotQA: 3-9K chars (1-3 chunks per doc at 3000 char chunks)
   - Natural Questions: 25-85K chars (8-28 chunks per doc)

2. Query type affects what "good retrieval" means:
   - Factual: Answer is a short span in one location
   - Multi-hop: Answer requires combining info from multiple places

3. Our experiments show chunk SIZE dominates regardless of dataset,
   but different datasets have different optimal chunk sizes.

The Dataset Interface:
======================
All datasets provide:
- corpus: Dict[doc_id, {title, text}] - the documents to chunk
- queries: List[Query] - questions with ground truth answers

Each Query has:
- question: What to ask
- answer: Ground truth for evaluation
- doc_id: Which document contains the answer
- complexity: "factual" or "multi_hop"

The corpus and queries are ALIGNED - every query references a document
that's in the corpus. This is critical for evaluating retrieval.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document as LlamaDocument


@dataclass
class Query:
    """
    Represents a query for RAG evaluation.

    This dataclass holds all information needed to evaluate a RAG system's
    retrieval and generation quality for a single query.

    Attributes:
        id: Unique identifier for the query
        question: The question text
        answer: Ground truth answer (used for RAGAS evaluation)
        doc_id: ID of the document containing the answer
        complexity: Query complexity level (e.g., "factual", "multi_hop")
        metadata: Additional metadata (dataset source, etc.)

    Example:
        >>> query = Query(
        ...     id="q_001",
        ...     question="Who wrote Romeo and Juliet?",
        ...     answer="William Shakespeare",
        ...     doc_id="doc_shakespeare_01",
        ...     complexity="factual",
        ... )
    """

    id: str
    question: str
    answer: str
    doc_id: str
    complexity: str = "factual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "doc_id": self.doc_id,
            "complexity": self.complexity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Query":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            doc_id=data["doc_id"],
            complexity=data.get("complexity", "factual"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Corpus:
    """
    Represents a document corpus with metadata.

    Attributes:
        documents: Dict mapping doc_id -> {"title": str, "text": str}
        name: Corpus name/identifier
        description: Human-readable description
        metadata: Additional corpus-level metadata
    """

    documents: dict[str, dict[str, str]]
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)

    def to_llamaindex_docs(self) -> list[LlamaDocument]:
        """
        Convert corpus to LlamaIndex Document objects.

        Returns:
            List of LlamaIndex Document objects with metadata

        Example:
            >>> corpus = Corpus(documents={"doc1": {"title": "...", "text": "..."}})
            >>> docs = corpus.to_llamaindex_docs()
        """
        documents = []

        for doc_id, doc_data in self.documents.items():
            title = doc_data.get("title", "")
            text = doc_data.get("text", "")

            # Combine title and text
            content = f"{title}\n\n{text}" if title else text

            doc = LlamaDocument(
                text=content,
                metadata={
                    "doc_id": doc_id,
                    "title": title,
                    "source": self.name,
                },
            )
            documents.append(doc)

        return documents


class Dataset(ABC):
    """
    Abstract base class for dataset loaders.

    All dataset loaders must inherit from this class and implement
    the `load` method. This ensures a consistent interface across
    different datasets.

    Attributes:
        name: Dataset identifier (e.g., "hotpotqa", "natural_questions")

    Example:
        >>> class MyDataset(Dataset):
        ...     name = "my_dataset"
        ...
        ...     def load(self, num_examples=40, **kwargs):
        ...         corpus = {}
        ...         queries = []
        ...         # Load your data here
        ...         return corpus, queries
        >>>
        >>> dataset = MyDataset()
        >>> corpus, queries = dataset.load(num_examples=100)
    """

    # Class attribute - override in subclasses
    name: str = "base"

    @abstractmethod
    def load(
        self,
        num_examples: int = 40,
        min_length: int | None = None,
        max_length: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> tuple[dict[str, dict[str, str]], list[Query]]:
        """
        Load corpus and aligned queries from the dataset.

        This method must be implemented by all subclasses. It should
        return a corpus dictionary and a list of Query objects that
        are aligned (i.e., each query references a document in the corpus).

        Args:
            num_examples: Number of document-query pairs to load
            min_length: Minimum document length in characters
            max_length: Maximum document length in characters
            seed: Random seed for reproducibility
            **kwargs: Dataset-specific additional arguments

        Returns:
            Tuple of:
                - corpus: Dict mapping doc_id -> {"title": str, "text": str}
                - queries: List of Query objects

        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If no documents match the length criteria
        """
        raise NotImplementedError("Subclasses must implement load()")

    def load_by_length(
        self,
        length_category: str,
        num_examples: int = 20,
        seed: int = 42,
    ) -> tuple[dict[str, dict[str, str]], list[Query]]:
        """
        Load corpus filtered by document length category.

        This is a convenience method that maps length categories
        to specific min/max length values. Subclasses can override
        this to define dataset-specific length thresholds.

        Args:
            length_category: One of "short", "medium", "long"
            num_examples: Number of examples to load
            seed: Random seed

        Returns:
            Tuple of (corpus dict, queries list)

        Raises:
            ValueError: If length_category is not recognized
        """
        # Default thresholds - subclasses can override
        length_ranges = {
            "short": (0, 5000),
            "medium": (5000, 10000),
            "long": (10000, None),
        }

        if length_category not in length_ranges:
            raise ValueError(
                f"Unknown length category '{length_category}'. "
                f"Choose from: {list(length_ranges.keys())}"
            )

        min_len, max_len = length_ranges[length_category]
        return self.load(
            num_examples=num_examples,
            min_length=min_len,
            max_length=max_len,
            seed=seed,
        )

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(name='{self.name}')"
