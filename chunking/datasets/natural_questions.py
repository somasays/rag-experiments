"""
Natural Questions dataset loader.

Natural Questions contains real Google search queries with full Wikipedia
articles as context. Documents are significantly longer than HotpotQA
(25K-85K characters typical), making this dataset ideal for testing
chunking strategies on long documents.

Why Natural Questions for chunking experiments?
===============================================
1. LONG DOCUMENTS: 25-85K chars vs HotpotQA's 3-9K. This stress-tests
   chunking strategies on realistic long-form content.

2. REAL QUERIES: These are actual Google searches, not synthetic questions.
   Reflects how users actually ask questions.

3. CROSS-DATASET VALIDATION: Our key finding (chunk SIZE > strategy)
   holds on both HotpotQA and NQ - the finding generalizes!

Our cross_dataset experiment found:
    - Correlation r=0.98 between chunk size and recall on NQ
    - Same pattern as HotpotQA despite 10x longer documents
    - Chunk SIZE dominates strategy regardless of doc length

Key characteristics:
    - Single-hop factual questions (real Google queries)
    - Documents: 25K-85K characters (full Wikipedia articles)
    - Real user queries (not crowdsourced)
    - Good for testing chunking impact on long document retrieval

Document length distribution (from validation split):
    - 10K-30K chars: ~25% of docs (short for NQ)
    - 30K-60K chars: ~50% of docs (medium)
    - 60K+ chars: ~25% of docs (long)

Note: Initial download is ~45GB compressed. Plan accordingly.
"""

import random
from typing import Any

from datasets import load_dataset

from .base import Dataset, Query


def _extract_nq_text(document: dict) -> str:
    """
    Extract plain text from Natural Questions document structure.

    NQ stores tokens with is_html flags. We filter out HTML tags
    and join the remaining tokens to get clean text.

    Args:
        document: NQ document dict with tokens structure

    Returns:
        Clean text string with HTML tags removed
    """
    tokens = document["tokens"]["token"]
    is_html = document["tokens"]["is_html"]
    text_tokens = [t for t, h in zip(tokens, is_html) if not h]
    return " ".join(text_tokens)


class NaturalQuestionsDataset(Dataset):
    """
    Load Natural Questions dataset with aligned corpus and queries.

    Natural Questions contains real Google search queries paired with
    full Wikipedia articles. Documents are much longer than HotpotQA
    (25K-85K characters), making this dataset ideal for experiments
    on chunking long documents.

    Key differences from HotpotQA:
        - Single-hop factual questions (vs multi-hop reasoning)
        - Full Wikipedia articles (vs curated passages)
        - Real user queries from Google (vs crowdsourced)

    Attributes:
        name: Dataset identifier ("natural_questions")

    Example:
        >>> dataset = NaturalQuestionsDataset()
        >>> corpus, queries = dataset.load(num_examples=40)
        >>> print(f"Loaded {len(corpus)} documents, {len(queries)} queries")

        # Filter by length (long documents)
        >>> corpus, queries = dataset.load(num_examples=40, min_length=60000)
        >>> print("Long documents only")

    Note:
        The dataset is loaded from HuggingFace Hub on first use.
        Internet connection required for initial download.
        Download is ~45GB compressed.
    """

    name = "natural_questions"

    # Length thresholds specific to Natural Questions (longer than HotpotQA)
    LENGTH_RANGES = {
        "short": (10000, 30000),   # ~25% of docs
        "medium": (30000, 60000),  # ~50% of docs
        "long": (60000, None),     # ~25% of docs
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
        Load corpus and aligned queries from Natural Questions.

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
            >>> dataset = NaturalQuestionsDataset()
            >>> corpus, queries = dataset.load(
            ...     num_examples=40,
            ...     min_length=60000,  # Long documents only
            ... )
        """
        # Set defaults
        min_length = min_length or 0

        # Load from HuggingFace
        dataset = load_dataset(
            "google-research-datasets/natural_questions",
            "default",
            split="validation",
            trust_remote_code=True,
        )

        # Pre-compute and filter examples
        valid_indices = []
        doc_texts = {}

        for idx in range(len(dataset)):
            row = dataset[idx]

            # Skip examples without short answers (unanswerable)
            short_answers = row["annotations"]["short_answers"][0]["text"]
            if not short_answers:
                continue

            # Extract plain text from document
            full_text = _extract_nq_text(row["document"])
            doc_len = len(full_text)

            # Filter by length
            if doc_len >= min_length:
                if max_length is None or doc_len <= max_length:
                    valid_indices.append(idx)
                    doc_texts[idx] = full_text

        if not valid_indices:
            raise ValueError(
                f"No Natural Questions documents meet length requirements "
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
            doc_id = f"nq_{idx}"
            full_text = doc_texts[idx]
            title = row["document"]["title"]

            # Build corpus entry
            corpus[doc_id] = {
                "title": title,
                "text": full_text,
            }

            # Join multiple short answer spans
            short_answers = row["annotations"]["short_answers"][0]["text"]
            answer = (
                "; ".join(short_answers)
                if isinstance(short_answers, list)
                else short_answers
            )

            # Build query
            query = Query(
                id=str(row["id"]),
                question=row["question"]["text"],
                answer=answer,
                doc_id=doc_id,
                complexity="factual",  # NQ is primarily factual lookup
                metadata={
                    "dataset": "natural_questions",
                    "doc_length": len(full_text),
                    "title": title,
                },
            )
            queries.append(query)

        return corpus, queries

    def load_by_length(
        self,
        length_category: str,
        num_examples: int = 40,
        seed: int = 42,
    ) -> tuple[dict[str, dict[str, str]], list[Query]]:
        """
        Load corpus filtered by document length category.

        Uses Natural Questions-specific length thresholds (longer than HotpotQA):
            - short: 10K-30K chars (~25% of docs)
            - medium: 30K-60K chars (~50% of docs)
            - long: 60K+ chars (~25% of docs)

        Args:
            length_category: One of "short", "medium", "long"
            num_examples: Number of examples to load
            seed: Random seed

        Returns:
            Tuple of (corpus dict, queries list)

        Example:
            >>> dataset = NaturalQuestionsDataset()
            >>> corpus, queries = dataset.load_by_length("long", num_examples=40)
            >>> # Documents are 60K+ characters
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
        num_per_category: int = 40,
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
            >>> dataset = NaturalQuestionsDataset()
            >>> by_length = dataset.load_all_lengths(num_per_category=40)
            >>> for category, (corpus, queries) in by_length.items():
            ...     print(f"{category}: {len(corpus)} docs")
        """
        return {
            "short": self.load_by_length("short", num_per_category, seed),
            "medium": self.load_by_length("medium", num_per_category, seed + 1),
            "long": self.load_by_length("long", num_per_category, seed + 2),
        }
