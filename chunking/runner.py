"""
Unified experiment runner for chunking experiments.

This module provides the ExperimentRunner class that orchestrates the entire
experiment pipeline: loading data, chunking, indexing, retrieval, generation,
and evaluation.

The Experiment Pipeline:
========================

    YAML Config → Dataset → Chunking → Indexing → Retrieval → RAGAS
        ↓           ↓          ↓          ↓           ↓         ↓
    Settings    Corpus +   TextNodes   ChromaDB   Top-K     Metrics
                Queries                 Index     Results

Step-by-step:
1. Load YAML config defining experiment (strategies, sizes, filters)
2. Load dataset (corpus of documents + test queries with answers)
3. For each configuration:
   a. Chunk documents using the specified strategy
   b. Build vector index from chunks (ChromaDB + embeddings)
   c. For each query: retrieve top-k chunks, generate answer
   d. Evaluate with RAGAS (context_recall, precision, faithfulness)
4. Save results to JSON (detailed per-query + summary)

Key Features:
=============
- Checkpointing: Saves after each config, can resume interrupted runs
- Memory management: Calls gc.collect() between configs
- Detailed logging: Progress bars, timing, per-query results
- Flexible filtering: Support for doc_length filters, query subsets

YAML Configuration:
===================
Experiments are defined in YAML files under configs/:

    name: chunk_size_controlled
    description: Compare strategies at controlled chunk sizes
    dataset: hotpotqa
    dataset_args:
      num_examples: 40
      seed: 42
    configurations:
      - name: token_1024
        strategy: token
        chunk_size: 1024
      - name: sentence_1024
        strategy: sentence
        chunk_size: 1024  # WARNING: Will be ignored!
    evaluation:
      metrics: [context_recall, context_precision]
      top_k: 3
      num_queries: 40

Output Structure:
=================
    results/<experiment_name>/
    ├── summary.json           # Aggregated metrics for all configs
    └── detailed/
        ├── token_1024.json    # Per-query results for each config
        └── sentence_1024.json

Example Usage:
==============
    # Run from CLI
    ./run.sh chunking -e document_length

    # Or programmatically
    config = ExperimentConfig.from_yaml("configs/document_length.yaml")
    runner = ExperimentRunner(config)
    results = runner.run()

Dependencies:
    - llama_index: Indexing and retrieval
    - ChromaDB: Vector storage (via llama_index)
    - OpenAI: Embeddings
    - Ollama: Local LLM for generation
    - RAGAS: Evaluation metrics
"""

import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from tqdm import tqdm

from .datasets import DATASETS, Query, get_dataset
from .datasets.base import Corpus
from .evaluation import EvaluationResult, RAGASEvaluator
from .strategies import STRATEGIES, get_strategy


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment run.

    Attributes:
        name: Experiment name (e.g., "document_length")
        description: Human-readable description
        dataset: Dataset name (e.g., "hotpotqa")
        dataset_args: Arguments for dataset loading
        configurations: List of chunking configurations to test
        evaluation: Evaluation settings
        output: Output settings

    Example:
        >>> config = ExperimentConfig(
        ...     name="test_experiment",
        ...     dataset="hotpotqa",
        ...     configurations=[
        ...         {"name": "token_1024", "strategy": "token", "chunk_size": 1024}
        ...     ],
        ... )
    """

    name: str
    description: str = ""
    dataset: str = "hotpotqa"
    dataset_args: dict[str, Any] = field(default_factory=dict)
    configurations: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=lambda: {
        "metrics": ["context_recall", "context_precision", "faithfulness", "answer_relevancy"],
        "top_k": 3,
        "num_queries": 40,
    })
    output: dict[str, Any] = field(default_factory=lambda: {
        "detailed": True,
        "summary": True,
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance

        Example:
            >>> config = ExperimentConfig.from_yaml("configs/document_length.yaml")
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class ConfigurationResult:
    """
    Results from a single configuration run.

    Attributes:
        config_name: Name of the configuration
        strategy: Chunking strategy used
        chunk_size: Target chunk size
        actual_chunk_size: Actual average chunk size
        evaluation: RAGAS evaluation results
        chunking_stats: Statistics about chunking
        duration_seconds: Time taken for this configuration
        detailed_results: Per-query detailed results
    """

    config_name: str
    strategy: str
    chunk_size: int
    actual_chunk_size: float
    evaluation: EvaluationResult
    chunking_stats: dict[str, Any]
    duration_seconds: float
    detailed_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config_name,
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "actual_chunk_size": self.actual_chunk_size,
            **self.evaluation.to_dict(),
            "duration_seconds": self.duration_seconds,
            "num_chunks": self.chunking_stats.get("chunk_count", 0),
            "chunks_per_doc": self.chunking_stats.get("chunks_per_doc", 0),
            "avg_doc_length": self.chunking_stats.get("avg_doc_length", 0),
            "avg_chunk_length": self.chunking_stats.get("avg_chunk_length", 0),
        }


class ExperimentRunner:
    """
    Unified runner for chunking experiments.

    This class orchestrates the entire experiment pipeline:
    1. Load dataset (corpus + queries)
    2. For each configuration:
       a. Chunk documents using specified strategy
       b. Build vector index
       c. Retrieve and generate for each query
       d. Evaluate with RAGAS
    3. Save results

    Attributes:
        config: Experiment configuration
        results_dir: Directory for output files
        embed_model: Embedding model for indexing
        llm: LLM for answer generation

    Example:
        >>> config = ExperimentConfig.from_yaml("configs/document_length.yaml")
        >>> runner = ExperimentRunner(config)
        >>> results = runner.run()
        >>> print(f"Completed {len(results)} configurations")
    """

    def __init__(
        self,
        config: ExperimentConfig,
        results_dir: str | Path | None = None,
        embed_model_name: str = "text-embedding-3-small",
        llm_model: str = "mistral:7b",
        skip_completed: bool = True,
    ):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            results_dir: Directory for results (default: chunking/results/<experiment_name>)
            embed_model_name: Embedding model for indexing
            llm_model: LLM model for answer generation (Ollama)
            skip_completed: If True, skip configs that already have results saved
        """
        self.config = config
        self.embed_model_name = embed_model_name
        self.llm_model = llm_model
        self.skip_completed = skip_completed

        # Set up results directory
        if results_dir is None:
            results_dir = Path(__file__).parent / "results" / config.name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluator
        self.evaluator = RAGASEvaluator()

        # Results storage
        self.results: list[ConfigurationResult] = []

    def run(self) -> list[ConfigurationResult]:
        """
        Run the complete experiment.

        Returns:
            List of ConfigurationResult objects, one per configuration

        Example:
            >>> runner = ExperimentRunner(config)
            >>> results = runner.run()
            >>> for r in results:
            ...     print(f"{r.config_name}: {r.evaluation.context_recall:.2%}")
        """
        print("=" * 70)
        print(f"Experiment: {self.config.name}")
        print(f"Description: {self.config.description}")
        print("=" * 70)

        # Check if any configurations use doc_length filters
        configs_by_length = self._group_configs_by_doc_length()

        if configs_by_length:
            # Run configurations grouped by document length
            self._run_with_length_filters(configs_by_length)
        else:
            # Run all configurations against a single corpus
            self._run_without_length_filters()

        # Save summary
        if self.config.output.get("summary", True):
            self._save_summary()

        print(f"\n{'=' * 70}")
        print(f"Experiment '{self.config.name}' complete!")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 70)

        return self.results

    def _group_configs_by_doc_length(self) -> dict[str, list[dict[str, Any]]]:
        """
        Group configurations by their doc_length filter value.

        Returns:
            Dict mapping doc_length category -> list of configs.
            Empty dict if no configs have doc_length filters.
        """
        grouped: dict[str, list[dict[str, Any]]] = {}

        for config in self.config.configurations:
            filters = config.get("filters", {})
            doc_length = filters.get("doc_length")
            if doc_length:
                if doc_length not in grouped:
                    grouped[doc_length] = []
                grouped[doc_length].append(config)

        return grouped

    def _run_with_length_filters(
        self,
        configs_by_length: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        Run configurations that have doc_length filters.

        Each length category gets its own corpus loaded via dataset.load_by_length().
        """
        dataset = get_dataset(self.config.dataset)
        total_configs = sum(len(configs) for configs in configs_by_length.values())
        config_num = 0

        for length_category, configs in configs_by_length.items():
            print(f"\n{'=' * 70}")
            print(f"Loading {length_category} documents...")
            print("=" * 70)

            # Load corpus for this length category
            num_examples = self.config.dataset_args.get("num_examples", 20)
            seed = self.config.dataset_args.get("seed", 42)

            # Adjust seed per category to get different documents
            seed_offset = {"short": 0, "medium": 1, "long": 2}.get(length_category, 0)

            corpus, queries = dataset.load_by_length(
                length_category=length_category,
                num_examples=num_examples,
                seed=seed + seed_offset,
            )

            doc_lengths = [len(d["text"]) for d in corpus.values()]
            avg_len = mean(doc_lengths) if doc_lengths else 0
            print(f"  Loaded {len(corpus)} {length_category} documents "
                  f"(avg {avg_len/1000:.1f}K chars)")
            print(f"  Loaded {len(queries)} queries")

            # Convert corpus to LlamaIndex docs
            corpus_obj = Corpus(documents=corpus, name=self.config.dataset)
            corpus_docs = corpus_obj.to_llamaindex_docs()

            # Run each configuration for this length category
            for config in configs:
                config_num += 1
                config_name = config["name"]

                # Check if already completed (checkpointing)
                if self.skip_completed and self._is_config_completed(config_name):
                    print(f"\n[{config_num}/{total_configs}] "
                          f"Skipping (already completed): {config_name}")
                    result = self._load_completed_result(config_name)
                    self.results.append(result)
                    continue

                print(f"\n{'=' * 70}")
                print(f"[{config_num}/{total_configs}] Running: {config_name}")
                print("=" * 70)

                result = self._run_single_config(
                    config=config,
                    corpus_docs=corpus_docs,
                    queries=queries,
                )
                self.results.append(result)

                # Save detailed results immediately
                if self.config.output.get("detailed", True):
                    self._save_detailed_result(result)

                # Clean up memory
                gc.collect()

    def _run_without_length_filters(self) -> None:
        """Run all configurations against a single corpus (no length filters)."""
        print(f"\nLoading dataset: {self.config.dataset}")
        dataset = get_dataset(self.config.dataset)
        corpus, queries = dataset.load(**self.config.dataset_args)

        doc_lengths = [len(d["text"]) for d in corpus.values()]
        avg_len = mean(doc_lengths) if doc_lengths else 0
        print(f"  Loaded {len(corpus)} documents (avg {avg_len/1000:.1f}K chars)")
        print(f"  Loaded {len(queries)} queries")

        # Convert corpus to LlamaIndex docs
        corpus_obj = Corpus(documents=corpus, name=self.config.dataset)
        corpus_docs = corpus_obj.to_llamaindex_docs()

        # Run each configuration
        for i, config in enumerate(self.config.configurations, 1):
            config_name = config["name"]

            # Check if already completed (checkpointing)
            if self.skip_completed and self._is_config_completed(config_name):
                print(f"\n[{i}/{len(self.config.configurations)}] "
                      f"Skipping (already completed): {config_name}")
                result = self._load_completed_result(config_name)
                self.results.append(result)
                continue

            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(self.config.configurations)}] Running: {config_name}")
            print("=" * 70)

            result = self._run_single_config(
                config=config,
                corpus_docs=corpus_docs,
                queries=queries,
            )
            self.results.append(result)

            # Save detailed results immediately
            if self.config.output.get("detailed", True):
                self._save_detailed_result(result)

            # Clean up memory
            gc.collect()

    def _run_single_config(
        self,
        config: dict[str, Any],
        corpus_docs: list,
        queries: list[Query],
    ) -> ConfigurationResult:
        """Run a single configuration."""
        start_time = time.time()
        config_name = config["name"]
        strategy_name = config["strategy"]
        chunk_size = config.get("chunk_size", 1024)
        chunk_overlap = config.get("chunk_overlap", 128)

        # Set up embedding model
        embed_model = OpenAIEmbedding(model=self.embed_model_name)
        Settings.embed_model = embed_model

        # Set up LLM for generation
        Settings.llm = Ollama(model=self.llm_model, request_timeout=120.0)

        # Filter queries if needed
        filtered_queries = queries
        if "filters" in config:
            filtered_queries = self._filter_queries(queries, config["filters"])

        # Limit queries if specified
        num_queries = self.config.evaluation.get("num_queries", len(filtered_queries))
        filtered_queries = filtered_queries[:num_queries]

        # Get chunker with appropriate settings
        strategy_kwargs = {}
        if strategy_name == "semantic":
            strategy_kwargs["embed_model"] = embed_model
        if "token_chunk_size" in config:
            strategy_kwargs["token_chunk_size"] = config["token_chunk_size"]

        chunker = get_strategy(
            strategy_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **strategy_kwargs,
        )

        # Chunk documents
        print(f"  Chunking with {strategy_name} (target: {chunk_size} chars)...")
        nodes = chunker.chunk(corpus_docs)
        chunking_stats = chunker.verify_chunking(corpus_docs, nodes)

        # Calculate actual chunk size
        sizes = [len(n.text) for n in nodes]
        actual_chunk_size = mean(sizes) if sizes else 0

        print(f"  Created {len(nodes)} chunks "
              f"({chunking_stats['chunks_per_doc']:.1f} chunks/doc, "
              f"avg {actual_chunk_size:.0f} chars)")

        # Build index
        print("  Building vector index...")
        index = VectorStoreIndex(nodes, show_progress=False)

        # Set up retriever
        top_k = self.config.evaluation.get("top_k", 3)
        retriever = index.as_retriever(similarity_top_k=top_k)

        # Run queries
        print(f"  Running {len(filtered_queries)} queries...")
        detailed_results = self._run_queries(
            retriever=retriever,
            queries=filtered_queries,
        )

        # Evaluate with RAGAS
        print("  Running RAGAS evaluation...")
        ragas_results = [
            {
                "question": r["question"],
                "answer": r["generated_answer"],
                "ground_truth": r["ground_truth"],
                "contexts": [c["full_text"] for c in r["retrieved_contexts"]],
            }
            for r in detailed_results
        ]
        metrics_list = self.config.evaluation.get("metrics")
        evaluation = self.evaluator.evaluate(ragas_results, metrics=metrics_list)

        duration = time.time() - start_time

        print(f"\n  Results:")
        print(f"    context_recall: {evaluation.context_recall:.3f}")
        print(f"    retrieval_success_rate: {evaluation.retrieval_success_rate:.3f}")
        print(f"    duration: {duration:.1f}s")

        return ConfigurationResult(
            config_name=config_name,
            strategy=strategy_name,
            chunk_size=chunk_size,
            actual_chunk_size=actual_chunk_size,
            evaluation=evaluation,
            chunking_stats=chunking_stats,
            duration_seconds=round(duration, 2),
            detailed_results=detailed_results,
        )

    def _run_queries(
        self,
        retriever: BaseRetriever,
        queries: list[Query],
    ) -> list[dict[str, Any]]:
        """Run retrieval and generation for all queries."""
        detailed_results = []

        for query in tqdm(queries, desc="Queries"):
            # Retrieve
            ret_start = time.time()
            ret_results = retriever.retrieve(query.question)
            retrieval_latency = (time.time() - ret_start) * 1000

            chunks = [r.node.text for r in ret_results]
            scores = [r.score for r in ret_results]

            # Check if answer is in any retrieved context
            answer_lower = query.answer.lower()
            retrieved_contexts = []
            answer_found = False

            for rank, (chunk, score) in enumerate(zip(chunks, scores)):
                contains_answer = answer_lower in chunk.lower()
                if contains_answer:
                    answer_found = True
                retrieved_contexts.append({
                    "rank": rank + 1,
                    "score": round(score, 4),
                    "full_text": chunk,
                    "text_preview": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                    "contains_answer": contains_answer,
                })

            # Generate answer
            context = "\n\n---\n\n".join(chunks)
            prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query.question}

Answer:"""

            gen_start = time.time()
            response = Settings.llm.complete(prompt)
            generation_latency = (time.time() - gen_start) * 1000

            detailed_results.append({
                "query_id": query.id,
                "question": query.question,
                "ground_truth": query.answer,
                "generated_answer": str(response),
                "answer_found_in_context": answer_found,
                "retrieved_contexts": retrieved_contexts,
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "generation_latency_ms": round(generation_latency, 2),
            })

        return detailed_results

    def _filter_queries(
        self,
        queries: list[Query],
        filters: dict[str, Any],
    ) -> list[Query]:
        """
        Filter queries based on filter criteria.

        Note: doc_length filtering is handled at the corpus loading level
        (see _run_with_length_filters), not here. This method handles
        other query-level filters that may be added in the future.
        """
        filtered = queries

        # doc_length is handled at corpus level - skip it here
        # Future filters can be added below

        return filtered

    def _save_detailed_result(self, result: ConfigurationResult) -> None:
        """Save detailed results for a single configuration."""
        detailed_dir = self.results_dir / "detailed"
        detailed_dir.mkdir(exist_ok=True)

        path = detailed_dir / f"{result.config_name}.json"
        with open(path, "w") as f:
            json.dump({
                "config": result.config_name,
                "strategy": result.strategy,
                "chunk_size": result.chunk_size,
                "actual_chunk_size": result.actual_chunk_size,
                "timestamp": datetime.now().isoformat(),
                "chunking_stats": result.chunking_stats,
                "evaluation": result.evaluation.to_dict(),
                "query_results": result.detailed_results,
            }, f, indent=2)

    def _save_summary(self) -> None:
        """Save summary of all configurations."""
        summary = {
            "experiment": self.config.name,
            "description": self.config.description,
            "dataset": self.config.dataset,
            "timestamp": datetime.now().isoformat(),
            "num_configurations": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

        path = self.results_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def _is_config_completed(self, config_name: str) -> bool:
        """Check if a configuration has already been completed."""
        result_path = self.results_dir / "detailed" / f"{config_name}.json"
        return result_path.exists()

    def _load_completed_result(self, config_name: str) -> ConfigurationResult:
        """Load a previously completed result from disk."""
        result_path = self.results_dir / "detailed" / f"{config_name}.json"
        with open(result_path) as f:
            data = json.load(f)

        return ConfigurationResult(
            config_name=data["config"],
            strategy=data["strategy"],
            chunk_size=data["chunk_size"],
            actual_chunk_size=data["actual_chunk_size"],
            evaluation=EvaluationResult.from_dict(data["evaluation"]),
            chunking_stats=data["chunking_stats"],
            duration_seconds=data.get("duration_seconds", 0),
            detailed_results=data.get("query_results", []),
        )


def run_from_yaml(config_path: str | Path) -> list[ConfigurationResult]:
    """
    Convenience function to run an experiment from a YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        List of ConfigurationResult objects

    Example:
        >>> results = run_from_yaml("configs/document_length.yaml")
    """
    config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(config)
    return runner.run()
