"""
RAGAS evaluation wrapper for chunking experiments.

This module provides a unified interface for evaluating RAG retrieval quality
using RAGAS (Retrieval-Augmented Generation Assessment) metrics.

What is RAGAS?
==============
RAGAS uses an LLM-as-judge approach to evaluate RAG quality. Instead of
simple string matching, it uses GPT-4 (or similar) to semantically assess
whether retrieval and generation are working correctly.

Why LLM-as-judge?
- Handles paraphrasing: "Shakespeare wrote Hamlet" == "Hamlet by Shakespeare"
- Understands context: Can judge if context actually supports the answer
- Scalable: Evaluates thousands of queries automatically

The Four Metrics:
=================
1. context_recall (PRIMARY) - Did we retrieve the answer?
   "Of all the information needed, how much did we retrieve?"
   - Score 0: Answer not in retrieved context
   - Score 1: All necessary info retrieved
   - THIS IS THE METRIC THAT MATTERS FOR RETRIEVAL

2. context_precision - Is retrieved context relevant?
   "Of what we retrieved, how much was useful?"
   - High precision + low recall = missing info
   - Low precision + high recall = noisy context

3. faithfulness - Is the answer grounded in context?
   "Does the LLM make stuff up, or stick to retrieved facts?"
   - Tests hallucination
   - Not relevant if you're only evaluating retrieval

4. answer_relevancy - Does the answer address the question?
   "Is the generated response actually answering what was asked?"
   - Tests generation quality, not retrieval
   - Orthogonal to chunking strategy choice

For chunking experiments, context_recall is primary because:
- If retrieval fails, nothing else matters
- Larger chunks → higher recall (our key finding)
- Strategy differences vanish when chunk size is controlled

Cost Consideration:
==================
Each RAGAS evaluation makes multiple LLM calls (~3-5 per query per metric).
For 40 queries x 4 metrics = ~160-200 API calls per configuration.
Use gpt-4o-mini to minimize cost (~$0.10 per configuration).

Dependencies:
    - ragas: RAGAS evaluation library
    - langchain_openai: For evaluation LLM and embeddings
    - datasets: HuggingFace datasets library
"""

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


@dataclass
class EvaluationResult:
    """
    Container for evaluation metrics.

    Attributes:
        context_recall: Fraction of relevant context retrieved (0-1)
        context_precision: Fraction of retrieved context that is relevant (0-1)
        faithfulness: Is the answer faithful to context? (0-1)
        answer_relevancy: Does the answer address the question? (0-1)
        retrieval_success_rate: Fraction of queries where answer was in context
        per_query_results: Optional list of per-query detailed results

    Example:
        >>> result = EvaluationResult(
        ...     context_recall=0.85,
        ...     context_precision=0.75,
        ...     faithfulness=0.90,
        ...     answer_relevancy=0.80,
        ...     retrieval_success_rate=0.70,
        ... )
        >>> print(f"Recall: {result.context_recall:.2%}")
    """

    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    retrieval_success_rate: float = 0.0
    per_query_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary (excludes per_query_results)."""
        return {
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "retrieval_success_rate": self.retrieval_success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary (inverse of to_dict)."""
        return cls(
            context_recall=data.get("context_recall", 0.0),
            context_precision=data.get("context_precision", 0.0),
            faithfulness=data.get("faithfulness", 0.0),
            answer_relevancy=data.get("answer_relevancy", 0.0),
            retrieval_success_rate=data.get("retrieval_success_rate", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"recall={self.context_recall:.3f}, "
            f"precision={self.context_precision:.3f}, "
            f"faithfulness={self.faithfulness:.3f}, "
            f"relevancy={self.answer_relevancy:.3f})"
        )


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG retrieval quality.

    This class provides a unified interface for running RAGAS evaluations
    on retrieval results. It handles the conversion between different
    data formats and computes all standard RAGAS metrics.

    Attributes:
        llm_model: Model name for RAGAS evaluation LLM
        embedding_model: Model name for RAGAS embeddings

    Example:
        >>> evaluator = RAGASEvaluator()
        >>> results = [
        ...     {
        ...         "question": "Who wrote Hamlet?",
        ...         "answer": "Shakespeare",
        ...         "ground_truth": "William Shakespeare",
        ...         "contexts": ["Hamlet was written by William Shakespeare..."],
        ...     }
        ... ]
        >>> metrics = evaluator.evaluate(results)
        >>> print(f"Context Recall: {metrics.context_recall:.2%}")

    Note:
        Requires OPENAI_API_KEY environment variable for evaluation.
    """

    def __init__(
        self,
        llm_model: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the RAGAS evaluator.

        Args:
            llm_model: OpenAI model for RAGAS evaluation (default: gpt-5-mini)
            embedding_model: OpenAI embedding model (default: text-embedding-3-small)

        Note:
            Using gpt-5-mini for fast, reliable evaluation.
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # Initialize wrappers (lazy - actual models created on evaluate)
        self._ragas_llm = None
        self._ragas_embeddings = None

    def _init_models(self) -> None:
        """Initialize RAGAS models if not already done."""
        # Create OpenAI client (uses OPENAI_API_KEY from env)
        client = OpenAI()
        if self._ragas_llm is None:
            # llm_factory auto-handles temperature constraints for gpt-5-mini
            # Use low reasoning_effort to minimize token usage on reasoning
            self._ragas_llm = llm_factory(
                self.llm_model, client=client, reasoning_effort="low"
            )
        if self._ragas_embeddings is None:
            # embedding_factory needs provider first, then model
            self._ragas_embeddings = embedding_factory(
                "openai", model=self.embedding_model, client=client
            )

    def evaluate(
        self,
        results: list[dict[str, Any]],
        metrics: list[str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate retrieval results using RAGAS metrics.

        Args:
            results: List of result dicts, each containing:
                - question: The query text
                - answer: Generated answer
                - ground_truth: Expected answer
                - contexts: List of retrieved context strings
            metrics: Optional list of metrics to compute. If None, computes all.
                Available: context_recall, context_precision, faithfulness, answer_relevancy

        Returns:
            EvaluationResult with computed metrics

        Example:
            >>> results = [
            ...     {
            ...         "question": "What is Python?",
            ...         "answer": "A programming language",
            ...         "ground_truth": "Python is a high-level programming language",
            ...         "contexts": ["Python is a programming language created by..."],
            ...     }
            ... ]
            >>> metrics = evaluator.evaluate(results)
        """
        self._init_models()

        # Convert to RAGAS format
        eval_data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }

        dataset = Dataset.from_dict(eval_data)

        # Select metrics
        metric_objects = []
        if metrics is None or "context_recall" in metrics:
            metric_objects.append(context_recall)
        if metrics is None or "context_precision" in metrics:
            metric_objects.append(context_precision)
        if metrics is None or "faithfulness" in metrics:
            metric_objects.append(faithfulness)
        if metrics is None or "answer_relevancy" in metrics:
            metric_objects.append(answer_relevancy)

        # Run RAGAS evaluation
        ragas_scores = evaluate(
            dataset,
            metrics=metric_objects,
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
        )

        # Extract metrics safely
        def get_metric(key: str) -> float:
            try:
                val = ragas_scores[key]
                if isinstance(val, list):
                    valid = [v for v in val if v is not None and not math.isnan(v)]
                    return mean(valid) if valid else 0.0
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0.0
                return float(val)
            except (KeyError, TypeError, ValueError):
                return 0.0

        # Calculate retrieval success rate (answer found in context)
        success_count = sum(
            1 for r in results
            if self._answer_in_context(r["ground_truth"], r["contexts"])
        )
        retrieval_success_rate = success_count / len(results) if results else 0.0

        return EvaluationResult(
            context_recall=get_metric("context_recall"),
            context_precision=get_metric("context_precision"),
            faithfulness=get_metric("faithfulness"),
            answer_relevancy=get_metric("answer_relevancy"),
            retrieval_success_rate=retrieval_success_rate,
        )

    @staticmethod
    def _answer_in_context(answer: str, contexts: list[str]) -> bool:
        """Check if the answer appears in any of the contexts."""
        answer_lower = answer.lower()
        return any(answer_lower in ctx.lower() for ctx in contexts)

    def evaluate_batch(
        self,
        results_list: list[list[dict[str, Any]]],
        batch_names: list[str] | None = None,
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate multiple result sets.

        Args:
            results_list: List of result lists, one per configuration
            batch_names: Optional names for each batch

        Returns:
            Dict mapping batch name -> EvaluationResult

        Example:
            >>> batch_results = evaluator.evaluate_batch(
            ...     [results_token, results_sentence],
            ...     batch_names=["token", "sentence"],
            ... )
            >>> for name, metrics in batch_results.items():
            ...     print(f"{name}: {metrics.context_recall:.2%}")
        """
        if batch_names is None:
            batch_names = [f"batch_{i}" for i in range(len(results_list))]

        return {
            name: self.evaluate(results)
            for name, results in zip(batch_names, results_list)
        }
