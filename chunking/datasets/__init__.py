"""
Dataset Registry

This module provides a registry of all available datasets for chunking experiments.
Each dataset implements the Dataset interface and provides aligned corpus-query pairs.

Available Datasets:
    - hotpotqa: Multi-hop reasoning questions (3K-9K char documents)
    - natural_questions: Factual lookup questions (25K-85K char documents)

Usage:
    >>> from chunking.datasets import DATASETS, get_dataset
    >>>
    >>> # Get dataset by name
    >>> dataset = get_dataset("hotpotqa")
    >>> corpus, queries = dataset.load(num_examples=40)
    >>>
    >>> # Or instantiate directly
    >>> from chunking.datasets import HotpotQADataset
    >>> dataset = HotpotQADataset()
    >>> corpus, queries = dataset.load(num_examples=40, min_length=5000)

Extending:
    To add a custom dataset:
    1. Create a new file in chunking/datasets/
    2. Implement the Dataset interface
    3. Register it in this __init__.py:

    >>> from .my_dataset import MyDataset
    >>> DATASETS["my_dataset"] = MyDataset
"""

from typing import Any

from .base import Dataset, Query, Corpus
from .hotpotqa import HotpotQADataset
from .natural_questions import NaturalQuestionsDataset

# Dataset Registry
# Maps dataset name -> dataset class
DATASETS: dict[str, type[Dataset]] = {
    "hotpotqa": HotpotQADataset,
    "natural_questions": NaturalQuestionsDataset,
}


def get_dataset(name: str) -> Dataset:
    """
    Factory function to get a dataset loader by name.

    Args:
        name: Dataset name (hotpotqa, natural_questions)

    Returns:
        Instantiated Dataset

    Raises:
        ValueError: If dataset name is not recognized

    Example:
        >>> dataset = get_dataset("hotpotqa")
        >>> corpus, queries = dataset.load(num_examples=40)
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )

    dataset_class = DATASETS[name]
    return dataset_class()


def list_datasets() -> list[dict[str, str]]:
    """
    List all available datasets with descriptions.

    Returns:
        List of dicts with 'name' and 'description' keys

    Example:
        >>> for d in list_datasets():
        ...     print(f"{d['name']}: {d['description']}")
    """
    return [
        {
            "name": name,
            "description": cls.__doc__.split("\n")[0] if cls.__doc__ else "No description",
        }
        for name, cls in DATASETS.items()
    ]


__all__ = [
    "Dataset",
    "Query",
    "Corpus",
    "HotpotQADataset",
    "NaturalQuestionsDataset",
    "DATASETS",
    "get_dataset",
    "list_datasets",
]
