"""
Chunking Experiments Module

A self-contained module for running chunking strategy experiments on RAG pipelines.
This module investigates how different text chunking strategies affect retrieval
quality in RAG (Retrieval-Augmented Generation) systems.

Key Research Questions:
    1. Document Length: How does document length affect optimal chunking strategy?
    2. Chunk Size: Does larger chunk size consistently improve recall?
    3. Cross-Dataset: Do findings generalize across different datasets?
    4. Strategy Comparison: Which chunking strategy performs best?

Quick Start:
    # Run from command line
    ./run.sh chunking --list              # List experiments
    ./run.sh chunking -e document_length  # Run specific experiment
    ./run.sh chunking -a document_length  # Run analysis

    # Use programmatically
    from chunking.strategies import STRATEGIES
    from chunking.datasets import DATASETS
    from chunking.runner import ExperimentRunner

Example:
    >>> from chunking.strategies import TokenChunker
    >>> chunker = TokenChunker(chunk_size=256, chunk_overlap=50)
    >>> nodes = chunker.chunk(documents)
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
# Users should import from submodules directly:
#   from chunking.strategies import STRATEGIES, TokenChunker
#   from chunking.datasets import DATASETS, HotpotQADataset
#   from chunking.runner import ExperimentRunner
