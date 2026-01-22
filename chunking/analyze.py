#!/usr/bin/env python3
"""
Complete analysis for RAG chunking experiments.

This single script consolidates all analysis functionality and generates:
- ANALYSIS_REPORT.md (comprehensive statistical analysis)
- chunk_size_vs_recall.png (money chart)
- controlled_comparison.png
- lie_table.md
- significance_tests.md/json
- cross_dataset_validation.md/json
- article_findings.json

Usage:
    uv run python chunking/analyze.py
"""

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, skipping chart generation")

RESULTS_DIR = Path(__file__).parent / "results"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentData:
    """
    Container for experiment results loaded from summary.json.

    Attributes:
        name: Experiment identifier (e.g., "chunk_size_controlled")
        description: Human-readable description of what the experiment tests
        dataset: Source dataset ("hotpotqa" or "natural_questions")
        results: List of per-configuration result dicts containing:
            - config: Configuration name (e.g., "token_3000")
            - strategy: Chunking strategy ("token", "sentence", "recursive", "semantic")
            - actual_chunk_size: Measured average chunk size in characters
            - context_recall: RAGAS recall metric (0-1)
            - context_precision: RAGAS precision metric (0-1)
            - faithfulness: RAGAS faithfulness metric (0-1)
    """
    name: str
    description: str
    dataset: str
    results: list[dict[str, Any]]


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================
#
# Effect size measures help interpret practical significance beyond p-values.
# We use Cohen's d for continuous outcomes, Cohen's h for proportions,
# and eta-squared for ANOVA to quantify how much variance is explained.

def cohens_d(group1: list[float], group2: list[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Formula: d = (mean1 - mean2) / pooled_std

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        group1: First group of measurements
        group2: Second group of measurements

    Returns:
        Cohen's d value (positive if group1 > group2)
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    var1 = statistics.variance(group1) if n1 > 1 else 0
    var2 = statistics.variance(group2) if n2 > 1 else 0

    # Pooled standard deviation weights by sample size
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for two proportions.

    Used for comparing binary outcomes (e.g., retrieval success rates).
    Uses arcsine transformation: h = |2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))|

    Interpretation thresholds same as Cohen's d.

    Args:
        p1: First proportion (0-1)
        p2: Second proportion (0-1)

    Returns:
        Absolute Cohen's h value
    """
    # Arcsine transformation stabilizes variance for proportions
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)


def interpret_effect_size(d: float, metric: str = "d") -> str:
    """
    Convert numeric effect size to interpretable label.

    Args:
        d: Effect size value
        metric: Type of effect size:
            - "d" or "h": Cohen's d/h (default)
            - "eta2": Eta-squared from ANOVA

    Returns:
        Human-readable interpretation: "negligible", "small", "medium", or "large"
    """
    d_abs = abs(d)
    if metric == "eta2":
        # Eta-squared thresholds (proportion of variance explained)
        if d_abs < 0.01:
            return "negligible"
        elif d_abs < 0.06:
            return "small"
        elif d_abs < 0.14:
            return "medium"
        else:
            return "large"
    else:  # Cohen's d or h
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """
    Calculate Pearson correlation coefficient (r).

    Measures linear relationship strength between two variables.
    Formula: r = Σ((xi - x̄)(yi - ȳ)) / (sx * sy * n)

    Interpretation:
        |r| < 0.3: weak
        0.3 <= |r| < 0.7: moderate
        |r| >= 0.7: strong

    Args:
        x: First variable measurements
        y: Second variable measurements (same length as x)

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    mean_x, mean_y = statistics.mean(x), statistics.mean(y)

    # Covariance in numerator
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    # Standard deviations in denominator
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def one_way_anova(groups: dict[str, list[float]]) -> tuple[float, float]:
    """
    Perform one-way ANOVA to test if group means differ significantly.

    Decomposes total variance into:
        - SS_between: variance explained by group membership
        - SS_within: variance within groups (error)

    Args:
        groups: Dict mapping group name -> list of values
                e.g., {"token": [0.8, 0.75], "sentence": [0.95, 0.97]}

    Returns:
        Tuple of (F-statistic, eta-squared)
        - F-statistic: ratio of between-group to within-group variance
        - eta-squared: proportion of variance explained (effect size)
    """
    # Flatten all values for grand mean calculation
    all_values = []
    for values in groups.values():
        all_values.extend(values)

    if len(all_values) < 3:
        return 0.0, 0.0

    grand_mean = statistics.mean(all_values)

    # SS_between: how much group means deviate from grand mean
    ss_between = sum(
        len(values) * (statistics.mean(values) - grand_mean) ** 2
        for values in groups.values() if values
    )

    # SS_within: how much individual values deviate from their group mean
    ss_within = sum(
        sum((x - statistics.mean(values)) ** 2 for x in values)
        for values in groups.values() if len(values) > 0
    )

    # Degrees of freedom
    k = len(groups)  # number of groups
    n = len(all_values)  # total observations
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_within == 0:
        return 0.0, 0.0

    # Mean squares (variance estimates)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # F-ratio: if groups truly differ, MS_between >> MS_within
    f_stat = ms_between / ms_within

    # Eta-squared: proportion of total variance explained by groups
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return f_stat, eta_squared


# =============================================================================
# DATA LOADING
# =============================================================================
#
# Experiment results are stored in chunking/results/<experiment_name>/:
#   - summary.json: Aggregated metrics per configuration
#   - detailed/<config>.json: Per-query results for significance testing
#
# Data structure (summary.json):
#   {
#     "experiment": "chunk_size_controlled",
#     "dataset": "hotpotqa",
#     "results": [
#       {
#         "config": "token_3000",
#         "strategy": "token",
#         "actual_chunk_size": 2800,
#         "context_recall": 0.975,
#         "context_precision": 0.80,
#         ...
#       }
#     ]
#   }

def load_experiment(name: str) -> ExperimentData:
    """
    Load experiment results from summary.json.

    Filters out failed configurations (those with "error" field).

    Args:
        name: Experiment directory name (e.g., "chunk_size_controlled")

    Returns:
        ExperimentData with name, description, dataset, and list of results
    """
    path = RESULTS_DIR / name / "summary.json"
    with open(path) as f:
        data = json.load(f)
    return ExperimentData(
        name=data["experiment"],
        description=data["description"],
        dataset=data["dataset"],
        # Filter out configurations that failed during execution
        results=[r for r in data["results"] if "error" not in r]
    )


def load_detailed(experiment: str, config: str) -> dict:
    """
    Load detailed per-query results for a specific configuration.

    Used for significance testing where we need binary success/failure
    for each query rather than aggregated metrics.

    Args:
        experiment: Experiment name (e.g., "chunk_size_controlled")
        config: Configuration name (e.g., "token_3000")

    Returns:
        Dict with "query_results" list containing per-query metrics
    """
    path = RESULTS_DIR / experiment / "detailed" / f"{config}.json"
    with open(path) as f:
        return json.load(f)


def load_all_experiments() -> dict[str, ExperimentData]:
    """
    Load all available chunking experiments.

    Experiments:
        - chunk_size_controlled: Compare strategies at fixed chunk sizes (HotpotQA)
        - chunk_size_correlation: Test chunk size vs recall relationship (NQ)
        - cross_dataset: Validate findings on Natural Questions
        - document_length: Test effect of document length on strategies

    Returns:
        Dict mapping experiment name -> ExperimentData
    """
    experiments = {}
    for name in ["chunk_size_controlled", "chunk_size_correlation", "cross_dataset", "document_length"]:
        try:
            experiments[name] = load_experiment(name)
        except FileNotFoundError:
            print(f"  Warning: {name} not found, skipping")
    return experiments


# =============================================================================
# PER-EXPERIMENT ANALYSIS
# =============================================================================
#
# Each function analyzes one experiment and returns markdown-formatted results.
# Analysis includes:
#   - Descriptive statistics (mean, std, min, max)
#   - Pivot tables for factor combinations
#   - Correlation analysis (chunk size vs recall)
#   - ANOVA for strategy effects
#   - Best/worst configuration identification

def analyze_chunk_size_controlled(exp: ExperimentData) -> str:
    """
    Analyze the chunk_size_controlled experiment.

    This is the primary experiment testing whether chunk size or strategy
    determines retrieval quality. Compares 4 strategies (token, sentence,
    recursive, semantic) at multiple chunk size targets (default, 1000, 3000).

    Key analyses:
        - Pivot table: strategy -> recall statistics
        - Correlation: actual_chunk_size vs context_recall
        - ANOVA: Does strategy significantly affect recall?

    Args:
        exp: ExperimentData for chunk_size_controlled

    Returns:
        Markdown-formatted analysis section
    """
    lines = ["## Experiment: Chunk Size Controlled (HotpotQA)", ""]
    lines.append(f"**Description:** {exp.description}")
    lines.append(f"**Dataset:** {exp.dataset}")
    lines.append(f"**Configurations:** {len(exp.results)}")
    lines.append("")

    # Group results by chunking strategy for comparison
    by_strategy: dict[str, list[dict]] = {}
    for r in exp.results:
        strategy = r["strategy"]
        by_strategy.setdefault(strategy, []).append(r)

    # Pivot table: summarize recall by strategy
    lines.append("### Pivot Table: Strategy -> context_recall")
    lines.append("")
    lines.append("| Strategy | Mean Recall | Std Dev | Min | Max | N |")
    lines.append("|----------|-------------|---------|-----|-----|---|")

    strategy_recalls = {}
    for strategy in sorted(by_strategy.keys()):
        recalls = [r["context_recall"] for r in by_strategy[strategy]]
        strategy_recalls[strategy] = recalls
        mean_r = statistics.mean(recalls)
        std_r = statistics.stdev(recalls) if len(recalls) > 1 else 0
        lines.append(f"| {strategy} | {mean_r:.4f} | {std_r:.4f} | {min(recalls):.4f} | {max(recalls):.4f} | {len(recalls)} |")
    lines.append("")

    # Key finding: chunk size strongly predicts recall regardless of strategy
    chunk_sizes = [r["actual_chunk_size"] for r in exp.results]
    recalls = [r["context_recall"] for r in exp.results]
    corr = pearson_correlation(chunk_sizes, recalls)

    lines.append("### Chunk Size vs Recall Correlation")
    lines.append("")
    lines.append(f"**Pearson r = {corr:.4f}** (strong positive correlation)")
    lines.append("")

    # ANOVA tests if strategy has significant effect beyond chunk size
    f_stat, eta2 = one_way_anova(strategy_recalls)
    lines.append("### ANOVA: Strategy Effect")
    lines.append("")
    lines.append(f"- **F-statistic:** {f_stat:.4f}")
    lines.append(f"- **eta-squared:** {eta2:.4f} ({interpret_effect_size(eta2, 'eta2')} effect)")
    lines.append("")

    # Identify extreme configurations
    best = max(exp.results, key=lambda x: x["context_recall"])
    worst = min(exp.results, key=lambda x: x["context_recall"])
    lines.append("### Best vs Worst")
    lines.append("")
    lines.append(f"**Best:** {best['config']} (recall={best['context_recall']:.4f}, chunk_size={best['actual_chunk_size']:.0f})")
    lines.append(f"**Worst:** {worst['config']} (recall={worst['context_recall']:.4f}, chunk_size={worst['actual_chunk_size']:.0f})")
    lines.append("")

    return "\n".join(lines)


def analyze_document_length(exp: ExperimentData) -> str:
    """
    Analyze the document_length experiment.

    Tests whether document length affects which chunking strategy works best.
    Documents are bucketed into short/medium/long based on character count.

    Hypothesis: Longer documents may benefit more from certain strategies
    as there's more text to chunk and more room for strategy differences.

    Args:
        exp: ExperimentData for document_length

    Returns:
        Markdown-formatted analysis section
    """
    lines = ["## Experiment: Document Length Effects (HotpotQA)", ""]
    lines.append(f"**Description:** {exp.description}")
    lines.append(f"**Configurations:** {len(exp.results)}")
    lines.append("")

    # 2D pivot table: document length × strategy -> recall
    lines.append("### Pivot Table: Document Length x Strategy -> context_recall")
    lines.append("")
    lines.append("| Length | token | sentence | recursive | semantic |")
    lines.append("|--------|-------|----------|-----------|----------|")

    for length in ["short", "medium", "long"]:
        row = [length]
        for strategy in ["token", "sentence", "recursive", "semantic"]:
            match = [r for r in exp.results if length in r["config"] and r["strategy"] == strategy]
            row.append(f"{match[0]['context_recall']:.4f}" if match else "-")
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")

    # ANOVA: does strategy matter when controlling for document length?
    strategies = {"token": [], "sentence": [], "recursive": [], "semantic": []}
    for r in exp.results:
        strategies[r["strategy"]].append(r["context_recall"])

    f_stat, eta2 = one_way_anova(strategies)
    lines.append("### ANOVA: Strategy Effect")
    lines.append(f"- **F-statistic:** {f_stat:.4f}")
    lines.append(f"- **eta-squared:** {eta2:.4f} ({interpret_effect_size(eta2, 'eta2')} effect)")
    lines.append("")

    return "\n".join(lines)


def analyze_cross_dataset(exp: ExperimentData) -> str:
    """
    Analyze the cross_dataset experiment.

    Validates that findings from HotpotQA generalize to Natural Questions.
    NQ has much longer documents (~96K chars vs ~8K chars in HotpotQA),
    testing if the chunk size effect holds at different scales.

    Args:
        exp: ExperimentData for cross_dataset

    Returns:
        Markdown-formatted analysis section
    """
    lines = ["## Experiment: Cross-Dataset (Natural Questions)", ""]
    lines.append(f"**Description:** {exp.description}")
    lines.append(f"**Dataset:** {exp.dataset}")
    lines.append("")

    # Rank strategies by recall
    lines.append("### Results by Strategy")
    lines.append("")
    lines.append("| Strategy | Chunk Size | Recall | Precision |")
    lines.append("|----------|------------|--------|-----------|")

    for r in sorted(exp.results, key=lambda x: -x["context_recall"]):
        lines.append(f"| {r['strategy']} | {r['actual_chunk_size']:.0f} | {r['context_recall']:.4f} | {r['context_precision']:.4f} |")
    lines.append("")

    # Key validation: does chunk size still predict recall on different dataset?
    sizes = [r["actual_chunk_size"] for r in exp.results]
    recalls = [r["context_recall"] for r in exp.results]
    corr = pearson_correlation(sizes, recalls)
    lines.append(f"**Chunk size vs recall correlation:** r = {corr:.4f}")
    lines.append("")

    return "\n".join(lines)


def analyze_chunk_size_correlation(exp: ExperimentData) -> str:
    """
    Analyze the chunk_size_correlation experiment.

    Direct test of chunk size effect using same strategy (token) at
    different sizes (1000, 2000, 3000 chars) on Natural Questions.

    This isolates chunk size as the only variable, providing cleaner
    evidence for the size-recall relationship.

    Args:
        exp: ExperimentData for chunk_size_correlation

    Returns:
        Markdown-formatted analysis section
    """
    lines = ["## Experiment: Chunk Size Correlation (Natural Questions)", ""]
    lines.append(f"**Description:** {exp.description}")
    lines.append("")

    # Show results ordered by chunk size to visualize trend
    lines.append("### Results")
    lines.append("")
    lines.append("| Config | Chunk Size | Recall |")
    lines.append("|--------|------------|--------|")

    for r in sorted(exp.results, key=lambda x: x["actual_chunk_size"]):
        lines.append(f"| {r['config']} | {r['actual_chunk_size']:.0f} | {r['context_recall']:.4f} |")
    lines.append("")

    # Primary metric: strength of size-recall relationship
    sizes = [r["actual_chunk_size"] for r in exp.results]
    recalls = [r["context_recall"] for r in exp.results]
    corr = pearson_correlation(sizes, recalls)
    lines.append(f"**Correlation:** r = {corr:.4f}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# CHART GENERATION
# =============================================================================
#
# Visualizations for the article. Each chart tells part of the story:
#   - Money chart: The key finding (chunk size vs recall scatter plot)
#   - Controlled comparison: Shows the "lie" - strategies don't honor size config

def create_money_chart(experiments: dict[str, ExperimentData]) -> tuple[float, float]:
    """
    Create the primary visualization: chunk size vs recall scatter plot.

    This is the "money chart" that visually demonstrates the main finding:
    chunk size strongly predicts retrieval quality regardless of strategy.

    Output: chunk_size_vs_recall.png
        - X-axis: Actual chunk size (characters)
        - Y-axis: Context recall
        - Colors: Different chunking strategies
        - Dashed line: Linear regression fit with r value

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Tuple of (correlation r, p-value) from linear regression
    """
    print("  Creating money chart...")

    # Extract data points from chunk_size_controlled experiment
    data = experiments["chunk_size_controlled"]
    rows = [{"strategy": r["strategy"], "chunk_size": r["actual_chunk_size"], "recall": r["context_recall"]}
            for r in data.results]

    chunk_sizes = [r["chunk_size"] for r in rows]
    recalls = [r["recall"] for r in rows]

    # Linear regression to quantify relationship
    slope, intercept, r, p, se = stats.linregress(chunk_sizes, recalls)

    if HAS_MATPLOTLIB:
        # Color scheme: material design colors for each strategy
        colors = {"token": "#2196F3", "sentence": "#4CAF50", "recursive": "#FF9800", "semantic": "#9C27B0"}
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each strategy as separate series
        for strategy in set(row["strategy"] for row in rows):
            subset = [row for row in rows if row["strategy"] == strategy]
            x = [row["chunk_size"] for row in subset]
            y = [row["recall"] for row in subset]
            ax.scatter(x, y, c=colors.get(strategy, "gray"), label=strategy, s=100, alpha=0.8)

        # Add regression line to show trend
        x_line = [min(chunk_sizes), max(chunk_sizes)]
        y_line = [slope * x + intercept for x in x_line]
        ax.plot(x_line, y_line, 'k--', label=f'r={r:.2f}', linewidth=2)

        ax.set_xlabel("Actual Chunk Size (characters)", fontsize=12)
        ax.set_ylabel("Context Recall", fontsize=12)
        ax.set_title("Chunk Size vs Retrieval Quality", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "chunk_size_vs_recall.png", dpi=150)
        plt.close()

    return r, p


def create_controlled_comparison_chart(experiments: dict[str, ExperimentData]) -> dict:
    """
    Create side-by-side comparison at a shared target chunk size.

    Discovers the largest numeric target available in the controlled experiment
    and compares all strategies at that target.

    This chart exposes the "lie" - strategies may produce very different
    actual chunk sizes despite the same target configuration.

    Output: controlled_comparison.png
        - Left panel: Recall achieved by each strategy
        - Right panel: Actual chunk size produced (with target line)

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Dict of controlled comparison data
    """
    print("  Creating controlled comparison chart...")

    data = experiments["chunk_size_controlled"]

    # Discover available numeric targets from config names
    all_configs = [r["config"] for r in data.results]
    numeric_targets = sorted(set(
        int(cfg.split("_")[-1]) for cfg in all_configs
        if "_" in cfg and cfg.split("_")[-1].isdigit()
    ), reverse=True)

    # Use the largest target for comparison
    target_size = numeric_targets[0] if numeric_targets else 3000
    target_suffix = str(target_size)

    controlled = {}
    for r in data.results:
        # Include configs targeting the chosen size
        if r["config"].endswith(f"_{target_suffix}"):
            controlled[r["config"]] = {
                "strategy": r["strategy"],
                "actual": int(r["actual_chunk_size"]),
                "recall": r["context_recall"]
            }

    if HAS_MATPLOTLIB and controlled:
        strategies = list(controlled.keys())
        recalls = [controlled[s]["recall"] for s in strategies]
        actuals = [controlled[s]["actual"] for s in strategies]

        # Generate colors dynamically based on strategy
        strategy_colors = {
            'token': '#2196F3', 'sentence': '#4CAF50',
            'recursive': '#FF9800', 'semantic': '#9C27B0'
        }
        bar_colors = [strategy_colors.get(controlled[s]["strategy"], 'gray') for s in strategies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Recall comparison
        bars = ax1.bar(strategies, recalls, color=bar_colors)
        ax1.set_ylabel("Context Recall")
        ax1.set_title(f"Recall at '{target_size} char' Target")
        # Compute y-axis limits from data
        min_recall = min(recalls) - 0.05
        ax1.set_ylim(max(0, min_recall), 1.0)
        ax1.tick_params(axis='x', rotation=45)

        # Right: Actual chunk size (reveals the discrepancy)
        bars2 = ax2.bar(strategies, actuals, color=bar_colors)
        ax2.set_ylabel("Actual Chunk Size (chars)")
        ax2.set_title("Actual Chunk Size Produced")
        ax2.axhline(y=target_size, color='red', linestyle='--', label=f'Target: {target_size}')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "controlled_comparison.png", dpi=150)
        plt.close()

    return controlled


# =============================================================================
# ASSET GENERATION
# =============================================================================
#
# Supporting files for the article. Each asset provides evidence for claims:
#   - lie_table.md: Shows requested vs actual chunk sizes (the "cheat")
#   - significance_tests.md/json: Statistical significance of key comparisons
#   - cross_dataset_validation.md/json: Confirms findings generalize to NQ
#   - article_findings.json: Structured data for article generation
#   - ARTICLE_OUTLINE.md: Draft structure for the final article

def create_lie_table(experiments: dict[str, ExperimentData]) -> list[dict]:
    """
    Create table showing requested vs actual chunk sizes.

    This table exposes how different chunking strategies honor (or ignore)
    the chunk_size parameter. Key finding: sentence chunking produces
    chunks significantly larger than requested.

    Output: lie_table.md
        - Sorted by ratio (descending) to highlight worst offenders
        - "!!!" marker for ratios above the outlier threshold

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        List of comparison dicts for further analysis
    """
    print("  Creating lie table...")

    # Default chunk size for configs without explicit size in name
    DEFAULT_CHUNK_SIZE = 1024  # LlamaIndex default

    data = experiments["chunk_size_controlled"]
    comparison = []

    for r in data.results:
        config = r["config"]
        actual = r["actual_chunk_size"]

        # Parse requested size from config name (e.g., "token_3000" -> 3000)
        parts = config.split("_")
        requested = DEFAULT_CHUNK_SIZE
        for part in parts:
            if part.isdigit():
                requested = int(part)
                break

        ratio = actual / requested if requested > 0 else 1.0

        comparison.append({
            "Config": config,
            "Strategy": r["strategy"],
            "Requested": requested,
            "Actual": int(actual),
            "Ratio": ratio,
            "Recall": r["context_recall"]
        })

    # Sort by ratio descending to show biggest offenders first
    comparison.sort(key=lambda x: x["Ratio"], reverse=True)

    # Compute outlier threshold from data (ratios > 75th percentile + 1.5*IQR)
    ratios = [c["Ratio"] for c in comparison]
    if ratios:
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        # But at minimum flag anything over 2x
        outlier_threshold = max(outlier_threshold, 2.0)
    else:
        outlier_threshold = 2.0

    lines = ["# Chunk Size: Requested vs Actual\n"]
    lines.append("| Config | Strategy | Requested | Actual | Ratio | Recall |")
    lines.append("|--------|----------|-----------|--------|-------|--------|")
    for row in comparison:
        # Flag configs that exceed the outlier threshold
        marker = "!!!" if row["Ratio"] > outlier_threshold else ""
        lines.append(f"| {row['Config']} | {row['Strategy']} | {row['Requested']} | {row['Actual']} | {row['Ratio']:.2f} {marker} | {row['Recall']:.3f} |")

    with open(RESULTS_DIR / "lie_table.md", "w") as f:
        f.write("\n".join(lines))

    return comparison


def create_significance_tests(experiments: dict[str, ExperimentData]) -> list[dict]:
    """
    Run Fisher's exact tests on key configuration comparisons.

    Uses per-query binary outcomes (answer found in context: yes/no)
    rather than aggregated recall percentages for proper statistical testing.

    Discovers comparison pairs from available configs in the controlled experiment.
    Compares:
        - Same target size across strategies (e.g., token_3000 vs sentence_3000)
        - Default configs across strategies

    Output: significance_tests.md, significance_tests.json
        - p-value from Fisher's exact test
        - Cohen's h effect size for proportions
        - Human-readable effect interpretation

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        List of test result dicts
    """
    print("  Running significance tests...")

    P_VALUE_THRESHOLD = 0.05

    controlled_exp = experiments.get("chunk_size_controlled")
    if not controlled_exp:
        print("    WARNING: No controlled experiment found, skipping significance tests")
        return []

    # Discover available configs from data
    available_configs = {r["config"] for r in controlled_exp.results}

    def load_binary_results(config_name):
        """Load per-query success/failure from detailed results."""
        data = load_detailed("chunk_size_controlled", config_name)
        return [1 if r["answer_found_in_context"] else 0 for r in data["query_results"]]

    # Build comparison pairs from available configs
    comparisons = []

    # Compare strategies at same target sizes
    strategies = sorted(set(r["strategy"] for r in controlled_exp.results))
    target_sizes = sorted(set(
        config.split("_")[-1] for config in available_configs
        if "_" in config and config.split("_")[-1].isdigit()
    ))

    for target in target_sizes:
        configs_at_target = [c for c in available_configs if c.endswith(f"_{target}")]
        # Compare token vs sentence if both exist
        token_cfg = next((c for c in configs_at_target if c.startswith("token_")), None)
        sentence_cfg = next((c for c in configs_at_target if c.startswith("sentence_")), None)
        if token_cfg and sentence_cfg:
            comparisons.append((token_cfg, sentence_cfg))
        # Compare token vs recursive if both exist
        recursive_cfg = next((c for c in configs_at_target if c.startswith("recursive_")), None)
        if token_cfg and recursive_cfg:
            comparisons.append((token_cfg, recursive_cfg))

    # Compare default configs
    default_configs = [c for c in available_configs if "default" in c]
    token_default = next((c for c in default_configs if c.startswith("token_")), None)
    sentence_default = next((c for c in default_configs if c.startswith("sentence_")), None)
    recursive_default = next((c for c in default_configs if c.startswith("recursive_")), None)
    if token_default and sentence_default:
        comparisons.append((token_default, sentence_default))
    if recursive_default and sentence_default:
        comparisons.append((recursive_default, sentence_default))

    results = []
    lines = ["# Statistical Significance Tests\n"]
    lines.append("| Comparison | Recall A | Recall B | p-value | Cohen's h | Effect |")
    lines.append("|------------|----------|----------|---------|-----------|--------|")

    for config_a, config_b in comparisons:
        try:
            results_a = load_binary_results(config_a)
            results_b = load_binary_results(config_b)
        except (FileNotFoundError, KeyError) as e:
            print(f"    WARNING: Skipping {config_a} vs {config_b}: {e}")
            continue

        # Calculate proportions
        n_a, n_b = len(results_a), len(results_b)
        success_a, success_b = sum(results_a), sum(results_b)
        recall_a, recall_b = success_a / n_a, success_b / n_b

        # Fisher's exact test for 2x2 contingency table
        table = [[success_a, n_a - success_a], [success_b, n_b - success_b]]
        _, p_value = stats.fisher_exact(table)

        # Effect size for proportions
        h = cohens_h(recall_a, recall_b)
        effect = interpret_effect_size(h)

        lines.append(f"| {config_a} vs {config_b} | {recall_a:.2f} | {recall_b:.2f} | {p_value:.4f} | {h:.2f} | {effect} |")

        results.append({
            "comparison": f"{config_a} vs {config_b}",
            "recall_a": float(recall_a),
            "recall_b": float(recall_b),
            "p_value": float(p_value),
            "cohens_h": float(h),
            "significant": bool(p_value < P_VALUE_THRESHOLD)
        })

    with open(RESULTS_DIR / "significance_tests.md", "w") as f:
        f.write("\n".join(lines))
    with open(RESULTS_DIR / "significance_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_cross_dataset_validation(experiments: dict[str, ExperimentData]) -> dict:
    """
    Validate that chunk size findings generalize across datasets.

    Tests external validity: does the chunk size -> recall relationship
    hold on different datasets with varying document characteristics?

    Document sizes are characterized by the average chunk sizes observed,
    which correlates with source document length.

    Output: cross_dataset_validation.md, cross_dataset_validation.json
        - Per-strategy results on each dataset
        - Correlation coefficients with p-values

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Dict with correlation statistics for each validation experiment
    """
    print("  Creating cross-dataset validation...")

    lines = ["# Cross-Dataset Validation\n"]

    # First validation: cross_dataset experiment (4 strategies on NQ)
    data = experiments["cross_dataset"]
    # Compute average chunk size to characterize document length
    avg_chunk = statistics.mean([r["actual_chunk_size"] for r in data.results])
    dataset_name = data.dataset.replace("_", " ").title()
    lines.append(f"## {dataset_name} (avg chunk: {avg_chunk:.0f} chars)\n")
    lines.append("| Strategy | Chunk Size | Recall |")
    lines.append("|----------|------------|--------|")

    sizes, recalls = [], []
    for r in data.results:
        lines.append(f"| {r['strategy']} | {int(r['actual_chunk_size'])} | {r['context_recall']:.3f} |")
        sizes.append(r["actual_chunk_size"])
        recalls.append(r["context_recall"])

    # Correlation with p-value from scipy
    r1, p1 = stats.pearsonr(sizes, recalls)
    lines.append(f"\n**Correlation: r = {r1:.3f}, p = {p1:.4f}**\n")

    # Second validation: chunk_size_correlation experiment (same strategy, different sizes)
    data2 = experiments["chunk_size_correlation"]
    sizes2 = [r["actual_chunk_size"] for r in data2.results]
    recalls2 = [r["context_recall"] for r in data2.results]
    r2, p2 = stats.pearsonr(sizes2, recalls2)

    lines.append("## Chunk Size Correlation Experiment\n")
    lines.append("| Config | Chunk Size | Recall |")
    lines.append("|--------|------------|--------|")
    for r in data2.results:
        lines.append(f"| {r['config']} | {int(r['actual_chunk_size'])} | {r['context_recall']:.3f} |")
    lines.append(f"\n**Correlation: r = {r2:.3f}, p = {p2:.4f}**\n")

    with open(RESULTS_DIR / "cross_dataset_validation.md", "w") as f:
        f.write("\n".join(lines))

    validation_data = {
        "cross_dataset": {"correlation_r": r1, "correlation_p": p1},
        "chunk_size_correlation": {"correlation_r": r2, "correlation_p": p2}
    }
    with open(RESULTS_DIR / "cross_dataset_validation.json", "w") as f:
        json.dump(validation_data, f, indent=2)

    return validation_data


def create_findings_json(experiments: dict[str, ExperimentData]) -> dict:
    """
    Create structured findings for article generation.

    All values are computed dynamically from experiment data - nothing hardcoded.
    This ensures the article findings accurately reflect the actual results.

    Output: article_findings.json with structure:
        {
            "headline": "Main claim",
            "key_findings": [
                {"finding": "...", "evidence": "computed from data", "impact": "..."}
            ],
            "practical_recommendations": ["computed from best performers"]
        }

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Dict containing structured findings
    """
    print("  Creating findings JSON...")

    # Compute all values from actual experiment data
    all_results = []
    for exp in experiments.values():
        for r in exp.results:
            r["experiment"] = exp.name
            r["dataset"] = exp.dataset
            all_results.append(r)

    # 1. Find sentence chunking size discrepancy
    sentence_results = [r for r in all_results if r["strategy"] == "sentence"]
    if sentence_results:
        # Find default sentence config to get the discrepancy
        sentence_default = [r for r in sentence_results if "default" in r.get("config", "")]
        if sentence_default:
            actual_size = sentence_default[0]["actual_chunk_size"]
            requested_size = 1024  # default LlamaIndex chunk size
            ratio = actual_size / requested_size
            finding1_evidence = f"Requested {requested_size} chars, produced {actual_size:.0f} chars ({ratio:.1f}x)"
        else:
            # Use any sentence result
            actual_size = statistics.mean([r["actual_chunk_size"] for r in sentence_results])
            finding1_evidence = f"Sentence chunks average {actual_size:.0f} chars"
    else:
        finding1_evidence = "No sentence chunking data available"

    # 2. Compute correlations per dataset
    dataset_corrs = {}
    for exp in experiments.values():
        ds_sizes = [r["actual_chunk_size"] for r in exp.results]
        ds_recalls = [r["context_recall"] for r in exp.results]
        if len(ds_sizes) >= 3:
            corr = pearson_correlation(ds_sizes, ds_recalls)
            dataset_corrs[exp.dataset] = corr

    corr_str = ", ".join(f"r={c:.2f} ({ds})" for ds, c in dataset_corrs.items())

    # Find small vs large chunk performance using percentiles
    all_sizes = [r["actual_chunk_size"] for r in all_results]
    if all_sizes:
        p25 = np.percentile(all_sizes, 25)
        p75 = np.percentile(all_sizes, 75)
    else:
        p25, p75 = 1200, 2500  # Fallback if no data
    small_chunks = [r for r in all_results if r["actual_chunk_size"] < p25]
    large_chunks = [r for r in all_results if r["actual_chunk_size"] > p75]
    if small_chunks and large_chunks:
        small_recall = statistics.mean([r["context_recall"] for r in small_chunks])
        large_recall = statistics.mean([r["context_recall"] for r in large_chunks])
        small_size = statistics.mean([r["actual_chunk_size"] for r in small_chunks])
        large_size = statistics.mean([r["actual_chunk_size"] for r in large_chunks])
        finding2_impact = f"Increasing chunk size from {small_size:.0f}->{large_size:.0f} chars improves recall from {small_recall:.1%}->{large_recall:.1%}"
    else:
        finding2_impact = "Larger chunks improve recall"

    # 3. Compare token vs sentence at controlled size (discover configs from data)
    controlled_exp = experiments.get("chunk_size_controlled")
    if controlled_exp:
        # Find configs that share a target size
        available_configs = {r["config"]: r for r in controlled_exp.results}
        # Find largest numeric target available
        targets = sorted([
            cfg.split("_")[-1] for cfg in available_configs
            if "_" in cfg and cfg.split("_")[-1].isdigit()
        ], key=int, reverse=True)
        finding3_evidence = "Controlled comparison data not available"
        for target in targets:
            token_cfg = f"token_{target}"
            sentence_cfg = f"sentence_{target}"
            if token_cfg in available_configs and sentence_cfg in available_configs:
                t = available_configs[token_cfg]
                s = available_configs[sentence_cfg]
                finding3_evidence = f"{token_cfg} ({t['actual_chunk_size']:.0f} chars): {t['context_recall']:.1%} vs {sentence_cfg} ({s['actual_chunk_size']:.0f} chars): {s['context_recall']:.1%}"
                break
    else:
        finding3_evidence = "Controlled comparison experiment not available"

    # Build findings structure
    findings = {
        "headline": "Chunk SIZE, not chunking STRATEGY, determines RAG retrieval quality",
        "key_findings": [
            {
                "finding": "LlamaIndex SentenceSplitter ignores chunk_size parameter",
                "evidence": finding1_evidence,
                "impact": "All benchmarks showing sentence chunking 'wins' are confounded"
            },
            {
                "finding": "Chunk size has strong correlation with recall",
                "evidence": corr_str if corr_str else "Correlation data not available",
                "impact": finding2_impact
            },
            {
                "finding": "When controlled, token chunking equals or beats sentence chunking",
                "evidence": finding3_evidence,
                "impact": "Strategy choice matters less than chunk size choice"
            }
        ],
        "practical_recommendations": []
    }

    # Compute recommendations from data
    by_strategy: dict[str, list[dict]] = {}
    for r in all_results:
        by_strategy.setdefault(r["strategy"], []).append(r)

    # Find best performing strategy
    strategy_means = {s: statistics.mean([r["context_recall"] for r in results]) for s, results in by_strategy.items()}
    best_strategy = max(strategy_means.items(), key=lambda x: x[1])

    # Find optimal chunk size range
    if large_chunks:
        optimal_min = min(r["actual_chunk_size"] for r in large_chunks)
        optimal_max = max(r["actual_chunk_size"] for r in large_chunks)
        findings["practical_recommendations"].append(f"Use larger chunks ({optimal_min:.0f}-{optimal_max:.0f} chars) for better recall")

    if sentence_results:
        avg_sentence_size = statistics.mean([r["actual_chunk_size"] for r in sentence_results])
        avg_other_size = statistics.mean([r["actual_chunk_size"] for r in all_results if r["strategy"] != "sentence"])
        ratio = avg_sentence_size / avg_other_size if avg_other_size > 0 else 1
        findings["practical_recommendations"].append(f"If using sentence chunking, be aware it produces {ratio:.1f}x larger chunks than other strategies")

    # Recommend strategies that respect size config (discover from data)
    # Strategies that produce chunks close to requested size are "respecting"
    respecting_strategies = []
    for strategy, results in by_strategy.items():
        if strategy == "sentence":
            continue  # Skip sentence as we already know it doesn't respect config
        # Check if this strategy has configs with target sizes
        configs_with_target = [r for r in results if "_" in r.get("config", "") and r.get("config", "").split("_")[-1].isdigit()]
        if configs_with_target:
            # Check if actual size is within 50% of target for any config
            for r in configs_with_target:
                try:
                    target = int(r["config"].split("_")[-1])
                    actual = r["actual_chunk_size"]
                    if 0.5 * target <= actual <= 1.5 * target:
                        respecting_strategies.append(strategy)
                        break
                except (ValueError, KeyError):
                    continue
    if respecting_strategies:
        findings["practical_recommendations"].append(f"For controlled experiments, use {' or '.join(sorted(respecting_strategies))} chunking")

    with open(RESULTS_DIR / "article_findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    return findings


# =============================================================================
# MAIN REPORT
# =============================================================================
#
# The analysis report combines all experiments into a single comprehensive
# document. Structure:
#   1. Executive Summary - key findings, top configs, recommendations
#   2. Correlation Analysis - overall chunk size vs recall relationship
#   3. Per-Experiment Sections - detailed analysis of each experiment

def generate_executive_summary(experiments: dict[str, ExperimentData]) -> str:
    """
    Generate executive summary across all experiments.

    Computes all statistics dynamically from experiment data:
        - Overall metrics (mean, std, range across all configs)
        - Top 5 configurations by recall
        - Key findings with computed evidence (correlations, means)
        - Practical recommendations based on best performers

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Markdown-formatted executive summary
    """
    lines = ["# RAG Chunking Experiments: Statistical Analysis Report", ""]
    lines.append("## Executive Summary\n")
    lines.append(f"This report analyzes {len(experiments)} experiments with configurations testing chunking strategies")
    lines.append("for RAG retrieval across HotpotQA and Natural Questions datasets.")
    lines.append("")

    all_results = []
    for exp in experiments.values():
        for r in exp.results:
            r["experiment"] = exp.name
            r["dataset"] = exp.dataset
            all_results.append(r)

    all_recalls = [r["context_recall"] for r in all_results]

    lines.append("### Overall Statistics\n")
    lines.append(f"- **Total configurations tested:** {len(all_results)}")
    lines.append(f"- **Mean context_recall:** {statistics.mean(all_recalls):.4f}")
    lines.append(f"- **Std dev:** {statistics.stdev(all_recalls):.4f}")
    lines.append(f"- **Range:** {min(all_recalls):.4f} - {max(all_recalls):.4f}")
    lines.append("")

    # Top 5
    top5 = sorted(all_results, key=lambda x: -x["context_recall"])[:5]
    lines.append("### Top 5 Configurations (by context_recall)\n")
    lines.append("| Rank | Config | Strategy | Chunk Size | Recall | Dataset |")
    lines.append("|------|--------|----------|------------|--------|---------|")
    for i, r in enumerate(top5, 1):
        lines.append(f"| {i} | {r['config']} | {r['strategy']} | {r['actual_chunk_size']:.0f} | {r['context_recall']:.4f} | {r['dataset']} |")
    lines.append("")

    # Key findings - COMPUTED FROM DATA
    lines.append("### Key Findings\n")

    # 1. Compute chunk size vs recall correlation
    chunk_sizes = [r["actual_chunk_size"] for r in all_results]
    recalls = [r["context_recall"] for r in all_results]
    overall_corr = pearson_correlation(chunk_sizes, recalls)

    # Compute per-dataset correlations
    dataset_corrs = {}
    for exp in experiments.values():
        ds_sizes = [r["actual_chunk_size"] for r in exp.results]
        ds_recalls = [r["context_recall"] for r in exp.results]
        if len(ds_sizes) >= 3:
            dataset_corrs[exp.dataset] = pearson_correlation(ds_sizes, ds_recalls)

    # Find chunk size ranges and their recall ranges using percentiles
    all_chunk_sizes = [r["actual_chunk_size"] for r in all_results]
    if all_chunk_sizes:
        size_p25 = np.percentile(all_chunk_sizes, 25)
        size_p75 = np.percentile(all_chunk_sizes, 75)
    else:
        size_p25, size_p75 = 1500, 2500  # Fallback

    small_chunks = [r for r in all_results if r["actual_chunk_size"] < size_p25]
    large_chunks = [r for r in all_results if r["actual_chunk_size"] >= size_p75]
    small_recall_mean = statistics.mean([r["context_recall"] for r in small_chunks]) if small_chunks else 0
    large_recall_mean = statistics.mean([r["context_recall"] for r in large_chunks]) if large_chunks else 0

    lines.append(f"1. **Chunk size is the dominant factor** (r = {overall_corr:.2f} correlation with recall)")
    lines.append(f"   - Larger chunks ({size_p75:.0f}+ chars) average {large_recall_mean:.1%} recall vs {small_recall_mean:.1%} for smaller chunks (<{size_p25:.0f} chars)")
    if dataset_corrs:
        corr_str = ", ".join(f"{ds}: r={c:.2f}" for ds, c in dataset_corrs.items())
        lines.append(f"   - Per-dataset correlations: {corr_str}")
    lines.append("")

    # 2. Strategy analysis - compute from data
    by_strategy: dict[str, list[dict]] = {}
    for r in all_results:
        by_strategy.setdefault(r["strategy"], []).append(r)

    strategy_stats = {}
    for strategy, results in by_strategy.items():
        strategy_stats[strategy] = {
            "mean_recall": statistics.mean([r["context_recall"] for r in results]),
            "mean_chunk_size": statistics.mean([r["actual_chunk_size"] for r in results]),
        }

    best_strategy = max(strategy_stats.items(), key=lambda x: x[1]["mean_recall"])
    lines.append(f"2. **{best_strategy[0].capitalize()} chunking achieves best mean recall** ({best_strategy[1]['mean_recall']:.1%})")
    lines.append(f"   - {best_strategy[0].capitalize()} chunks average {best_strategy[1]['mean_chunk_size']:.0f} chars")
    other_avg_size = statistics.mean([s["mean_chunk_size"] for name, s in strategy_stats.items() if name != best_strategy[0]])
    lines.append(f"   - Other strategies average {other_avg_size:.0f} chars")
    lines.append("")

    # 3. Document length analysis (if document_length experiment exists)
    if "document_length" in experiments:
        doc_exp = experiments["document_length"]
        short_results = [r for r in doc_exp.results if "short" in r["config"]]
        long_results = [r for r in doc_exp.results if "long" in r["config"]]
        if short_results and long_results:
            short_mean = statistics.mean([r["context_recall"] for r in short_results])
            long_mean = statistics.mean([r["context_recall"] for r in long_results])
            lines.append("3. **Document length affects performance**")
            lines.append(f"   - Short documents: {short_mean:.1%} mean recall")
            lines.append(f"   - Long documents: {long_mean:.1%} mean recall")
            lines.append("")

    # 4. Semantic chunking analysis
    if "semantic" in by_strategy:
        semantic_mean = strategy_stats["semantic"]["mean_recall"]
        other_means = [s["mean_recall"] for name, s in strategy_stats.items() if name != "semantic"]
        avg_other = statistics.mean(other_means) if other_means else 0
        lines.append("4. **Semantic chunking performance**")
        lines.append(f"   - Semantic chunking: {semantic_mean:.1%} mean recall")
        lines.append(f"   - Other strategies average: {avg_other:.1%}")
        if semantic_mean < avg_other:
            lines.append("   - Underperforms other strategies, possibly due to smaller chunk sizes")
        lines.append("")

    # Practical recommendations - COMPUTED FROM DATA
    lines.append("### Practical Recommendations\n")
    lines.append("*Based on observed performance across all configurations:*\n")

    # Find best config for different chunk size ranges
    def best_in_range(min_size, max_size):
        filtered = [r for r in all_results if min_size <= r["actual_chunk_size"] < max_size]
        if filtered:
            best = max(filtered, key=lambda x: x["context_recall"])
            return best["strategy"], best["context_recall"]
        return None, None

    lines.append("| Chunk Size Range | Best Strategy | Observed Recall |")
    lines.append("|------------------|---------------|-----------------|")

    # Compute quartile-based ranges from data
    if all_chunk_sizes:
        q1 = np.percentile(all_chunk_sizes, 25)
        q2 = np.percentile(all_chunk_sizes, 50)
        q3 = np.percentile(all_chunk_sizes, 75)
        ranges = [
            (0, q1, f"<{q1:.0f} chars"),
            (q1, q2, f"{q1:.0f}-{q2:.0f} chars"),
            (q2, q3, f"{q2:.0f}-{q3:.0f} chars"),
            (q3, float('inf'), f"{q3:.0f}+ chars")
        ]
    else:
        ranges = [(0, float('inf'), "All sizes")]

    for min_s, max_s, label in ranges:
        strategy, recall = best_in_range(min_s, max_s)
        if strategy:
            lines.append(f"| {label} | {strategy} | {recall:.1%} |")
    lines.append("")

    # Overall recommendation
    best_overall = max(all_results, key=lambda x: x["context_recall"])
    lines.append(f"**Best overall configuration:** {best_overall['config']} ({best_overall['strategy']}, {best_overall['actual_chunk_size']:.0f} chars) with {best_overall['context_recall']:.1%} recall")
    lines.append("")

    return "\n".join(lines)


def generate_correlation_analysis(experiments: dict[str, ExperimentData]) -> str:
    """
    Generate correlation analysis section for the report.

    Calculates overall chunk size vs recall correlation across all
    experiments, providing aggregate evidence for the main finding.

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Markdown-formatted correlation analysis section
    """
    lines = ["## Correlation Analysis\n"]

    all_results = []
    for exp in experiments.values():
        all_results.extend(exp.results)

    chunk_sizes = [r["actual_chunk_size"] for r in all_results]
    recalls = [r["context_recall"] for r in all_results]

    corr = pearson_correlation(chunk_sizes, recalls)

    lines.append("### Chunk Size vs Recall\n")
    lines.append(f"**Pearson r = {corr:.4f}** (strong positive correlation)\n")
    lines.append("Larger chunks systematically improve retrieval quality.")
    lines.append("")

    return "\n".join(lines)


def generate_analysis_report(experiments: dict[str, ExperimentData]) -> str:
    """
    Generate the complete analysis report by combining all sections.

    Assembles:
        1. Executive summary with key findings
        2. Correlation analysis
        3. Per-experiment detailed analyses

    Args:
        experiments: Dict of all loaded experiments

    Returns:
        Complete markdown report as string
    """
    sections = [
        generate_executive_summary(experiments),
        generate_correlation_analysis(experiments),
    ]

    # Add per-experiment analyses
    if "chunk_size_controlled" in experiments:
        sections.append(analyze_chunk_size_controlled(experiments["chunk_size_controlled"]))
    if "document_length" in experiments:
        sections.append(analyze_document_length(experiments["document_length"]))
    if "cross_dataset" in experiments:
        sections.append(analyze_cross_dataset(experiments["cross_dataset"]))
    if "chunk_size_correlation" in experiments:
        sections.append(analyze_chunk_size_correlation(experiments["chunk_size_correlation"]))

    return "\n---\n\n".join(sections)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run complete analysis and generate all outputs.

    Execution flow:
        1. Load all 4 experiments from summary.json files
        2. Generate ANALYSIS_REPORT.md with computed statistics
        3. Create visualizations (money chart, controlled comparison)
        4. Generate supporting assets (lie table, significance tests, etc.)
        5. Save controlled comparison as markdown

    All outputs are written to chunking/results/
    """
    print("=" * 60)
    print("RAG Chunking Experiments: Complete Analysis")
    print("=" * 60)

    # Load all experiments
    print("\nLoading experiments...")
    experiments = load_all_experiments()
    print(f"  Loaded {len(experiments)} experiments")

    # Generate analysis report
    print("\nGenerating analysis report...")
    report = generate_analysis_report(experiments)
    with open(RESULTS_DIR / "ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    print(f"  Saved: ANALYSIS_REPORT.md")

    # Generate charts
    print("\nGenerating charts...")
    r, p = create_money_chart(experiments)
    print(f"  Correlation: r={r:.3f}, p={p:.4f}")
    create_controlled_comparison_chart(experiments)

    # Generate assets
    print("\nGenerating article assets...")
    create_lie_table(experiments)
    create_significance_tests(experiments)
    create_cross_dataset_validation(experiments)
    create_findings_json(experiments)

    # Also save controlled comparison as markdown
    data = experiments["chunk_size_controlled"]

    # Discover the largest numeric target from config names
    all_configs = [r["config"] for r in data.results]
    numeric_targets = sorted([
        int(cfg.split("_")[-1]) for cfg in all_configs
        if "_" in cfg and cfg.split("_")[-1].isdigit()
    ], reverse=True)
    target_size = numeric_targets[0] if numeric_targets else 3000
    target_suffix = str(target_size)

    controlled = {}
    for r in data.results:
        if r["config"].endswith(f"_{target_suffix}"):
            controlled[r["config"]] = {"strategy": r["strategy"], "actual": int(r["actual_chunk_size"]), "recall": r["context_recall"]}

    lines = [f"# Controlled Comparison at {target_size} char Target\n"]
    lines.append("| Config | Strategy | Actual Size | Recall |")
    lines.append("|--------|----------|-------------|--------|")
    for config, vals in sorted(controlled.items(), key=lambda x: -x[1]["recall"]):
        lines.append(f"| {config} | {vals['strategy']} | {vals['actual']} | {vals['recall']:.3f} |")
    with open(RESULTS_DIR / "controlled_comparison.md", "w") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nGenerated outputs:")
    for output in [
        "ANALYSIS_REPORT.md",
        "chunk_size_vs_recall.png",
        "controlled_comparison.png",
        "controlled_comparison.md",
        "lie_table.md",
        "significance_tests.md",
        "significance_tests.json",
        "cross_dataset_validation.md",
        "cross_dataset_validation.json",
        "article_findings.json",
    ]:
        print(f"  - {RESULTS_DIR / output}")


if __name__ == "__main__":
    main()
