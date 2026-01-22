"""
HotpotQA dataset loader.

HotpotQA is a multi-hop question answering dataset that requires reasoning
over multiple supporting documents. Documents are typically 3K-9K characters,
making it suitable for testing chunking strategies with moderate document sizes.

Why HotpotQA for chunking experiments?
======================================
1. MODERATE LENGTH: Documents are 3-9K chars - small enough to test chunking
   impact without excessive compute, large enough to require multiple chunks.

2. MULTI-HOP: Questions require synthesizing info from different parts of
   the context. This tests whether chunking preserves cross-references.

3. CURATED CONTEXTS: Each question comes with exactly the paragraphs needed
   to answer it. No noise - clean signal for evaluating retrieval.

Our experiments showed:
    - At 3000 char chunks: 1-3 chunks per document
    - Sentence chunking (3600+ chars) often puts whole doc in one chunk!
    - This is why sentence chunking "wins" - not strategy, just bigger chunks

Key characteristics:
    - Multi-hop reasoning questions
    - Documents: 3K-9K characters (moderate length)
    - Wikipedia-based contexts
    - Good for testing chunking impact on reasoning tasks

Document length distribution (from validation split):
    - 2000-4000 chars: ~9% of docs (short)
    - 4000-7000 chars: ~60% of docs (medium)
    - 7000+ chars: ~31% of docs (long)
"""

import random
from typing import Any

from datasets import load_dataset

from .base import Dataset, Query


class HotpotQADataset(Dataset):
    """
    Load HotpotQA dataset with aligned corpus and queries.

    HotpotQA provides multi-hop reasoning questions with curated
    Wikipedia contexts. Each example contains multiple supporting
    paragraphs that must be combined to answer the question.

    Document lengths range from 2K-9K characters, making this dataset
    suitable for testing chunking strategies on moderate-length documents.

    Attributes:
        name: Dataset identifier ("hotpotqa")

    Example:
        >>> dataset = HotpotQADataset()
        >>> corpus, queries = dataset.load(num_examples=40)
        >>> print(f"Loaded {len(corpus)} documents, {len(queries)} queries")

        # Filter by length
        >>> corpus, queries = dataset.load(num_examples=20, min_length=7000)
        >>> print("Long documents only")

    Note:
        The dataset is loaded from HuggingFace Hub on first use.
        Internet connection required for initial download.
    """

    name = "hotpotqa"

    # Length thresholds specific to HotpotQA
    LENGTH_RANGES = {
        "short": (2000, 4000),    # ~9% of docs
        "medium": (4000, 7000),   # ~60% of docs
        "long": (7000, None),     # ~31% of docs
    }

    def load(
        self,
        num_examples: int = 40,
        min_length: int | None = None,
        max_length: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> tuple[dict[str, dict[str, str]], list[Query]]:
        """
        Load corpus and aligned queries from HotpotQA.

        Args:
            num_examples: Number of document-query pairs to load
            min_length: Minimum document length in characters (default: 0)
            max_length: Maximum document length in characters (default: None)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of:
                - corpus: Dict mapping doc_id -> {"title": str, "text": str}
                - queries: List of Query objects aligned to corpus

        Raises:
            ValueError: If no documents match the length criteria

        Example:
            >>> dataset = HotpotQADataset()
            >>> corpus, queries = dataset.load(
            ...     num_examples=20,
            ...     min_length=7000,  # Long documents only
            ... )
        """
        # Set defaults
        min_length = min_length or 0

        # Load from HuggingFace
        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
        )

        # Pre-compute document lengths and filter
        valid_indices = []
        doc_texts = {}

        for idx in range(len(dataset)):
            row = dataset[idx]

            # Combine all context paragraphs into one document
            context_parts = []
            for title, sentences in zip(
                row["context"]["title"],
                row["context"]["sentences"]
            ):
                para_text = " ".join(sentences)
                context_parts.append(f"## {title}\n\n{para_text}")

            full_context = "\n\n".join(context_parts)
            doc_len = len(full_context)

            # Filter by length
            if doc_len >= min_length:
                if max_length is None or doc_len <= max_length:
                    valid_indices.append(idx)
                    doc_texts[idx] = full_context

        if not valid_indices:
            raise ValueError(
                f"No HotpotQA documents meet length requirements "
                f"(min={min_length}, max={max_length}). "
                f"Try relaxing the length filters."
            )

        # Sample documents
        random.seed(seed)
        selected = random.sample(
            valid_indices,
            min(num_examples, len(valid_indices))
        )

        corpus = {}
        queries = []

        for idx in selected:
            row = dataset[idx]
            doc_id = f"hotpotqa_{idx}"
            full_context = doc_texts[idx]

            # Build corpus entry
            corpus[doc_id] = {
                "title": row["question"][:50] + "...",
                "text": full_context,
            }

            # Build query
            query = Query(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                doc_id=doc_id,
                complexity="multi_hop",
                metadata={
                    "dataset": "hotpotqa",
                    "doc_length": len(full_context),
                },
            )
            queries.append(query)

        return corpus, queries

    def load_by_length(
        self,
        length_category: str,
        num_examples: int = 20,
        seed: int = 42,
    ) -> tuple[dict[str, dict[str, str]], list[Query]]:
        """
        Load corpus filtered by document length category.

        Uses HotpotQA-specific length thresholds:
            - short: 2000-4000 chars (~9% of docs)
            - medium: 4000-7000 chars (~60% of docs)
            - long: 7000+ chars (~31% of docs)

        Args:
            length_category: One of "short", "medium", "long"
            num_examples: Number of examples to load
            seed: Random seed

        Returns:
            Tuple of (corpus dict, queries list)

        Example:
            >>> dataset = HotpotQADataset()
            >>> corpus, queries = dataset.load_by_length("long", num_examples=20)
            >>> # Documents are 7000+ characters
        """
        if length_category not in self.LENGTH_RANGES:
            raise ValueError(
                f"Unknown length category '{length_category}'. "
                f"Choose from: {list(self.LENGTH_RANGES.keys())}"
            )

        min_len, max_len = self.LENGTH_RANGES[length_category]
        return self.load(
            num_examples=num_examples,
            min_length=min_len,
            max_length=max_len,
            seed=seed,
        )

    def load_all_lengths(
        self,
        num_per_category: int = 20,
        seed: int = 42,
    ) -> dict[str, tuple[dict[str, dict[str, str]], list[Query]]]:
        """
        Load documents from all length categories.

        Convenience method for experiments that compare across
        document lengths.

        Args:
            num_per_category: Number of documents per length category
            seed: Base random seed (incremented per category)

        Returns:
            Dict mapping length category -> (corpus, queries)

        Example:
            >>> dataset = HotpotQADataset()
            >>> by_length = dataset.load_all_lengths(num_per_category=20)
            >>> for category, (corpus, queries) in by_length.items():
            ...     print(f"{category}: {len(corpus)} docs")
        """
        return {
            "short": self.load_by_length("short", num_per_category, seed),
            "medium": self.load_by_length("medium", num_per_category, seed + 1),
            "long": self.load_by_length("long", num_per_category, seed + 2),
        }
