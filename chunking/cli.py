"""
Command-line interface for chunking experiments.

This module provides a Click-based CLI for running chunking experiments,
analyzing results, and generating reports.

Usage:
    # From project root
    uv run python -m chunking.cli --help

    # Or via run.sh wrapper
    ./run.sh chunking --list
    ./run.sh chunking --experiment document_length
    ./run.sh chunking --analyze document_length

Commands:
    --list, -l              List available experiments
    --experiment, -e NAME   Run specific experiment
    --analyze, -a NAME      Run analysis for experiment
    --all                   Run all experiments
    --report NAME           Generate report for experiment
"""

import sys
from pathlib import Path

import click

from .runner import ExperimentConfig, ExperimentRunner


# Available experiments (YAML configs in configs/ directory)
EXPERIMENTS = {
    "document_length": "How document length affects chunking strategies",
    "chunk_size_controlled": "Controlled chunk size comparison across strategies",
    "cross_dataset": "Cross-dataset generalization (HotpotQA → Natural Questions)",
    "chunk_size_correlation": "Chunk size vs recall correlation analysis",
    "smoke_test": "Quick verification of chunking pipeline (5 queries)",
}


def get_config_path(experiment: str) -> Path:
    """
    Get the path to an experiment config file.

    Args:
        experiment: Experiment name

    Returns:
        Path to the YAML config file

    Raises:
        click.ClickException: If experiment not found
    """
    config_dir = Path(__file__).parent / "configs"
    config_path = config_dir / f"{experiment}.yaml"

    if not config_path.exists():
        available = ", ".join(EXPERIMENTS.keys())
        raise click.ClickException(
            f"Experiment '{experiment}' not found.\n"
            f"Available experiments: {available}"
        )

    return config_path


def get_results_dir(experiment: str) -> Path:
    """
    Get the results directory for an experiment.

    Args:
        experiment: Experiment name

    Returns:
        Path to results directory
    """
    return Path(__file__).parent / "results" / experiment


@click.command()
@click.option(
    "--experiment", "-e",
    help="Run specific experiment by name",
    metavar="NAME",
)
@click.option(
    "--analyze", "-a",
    is_flag=True,
    help="Run comprehensive analysis on all completed experiments",
)
@click.option(
    "--all", "run_all",
    is_flag=True,
    help="Run all experiments sequentially",
)
@click.option(
    "--list", "-l", "list_experiments",
    is_flag=True,
    help="List available experiments",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(
    experiment: str | None,
    analyze: bool,
    run_all: bool,
    list_experiments: bool,
    verbose: bool,
) -> None:
    """
    Chunking experiments CLI.

    Run RAG chunking experiments, analyze results, and generate reports.

    \b
    Examples:
        # List available experiments
        python -m chunking.cli --list

        # Run a specific experiment
        python -m chunking.cli --experiment document_length

        # Run comprehensive analysis on all experiments
        python -m chunking.cli --analyze

        # Run all experiments
        python -m chunking.cli --all
    """
    # Handle mutually exclusive options
    options_count = sum([
        experiment is not None,
        analyze,
        run_all,
        list_experiments,
    ])

    if options_count == 0:
        # No options provided - show help
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    if options_count > 1:
        raise click.ClickException(
            "Only one of --experiment, --analyze, --all, or --list can be specified at a time."
        )

    # Execute requested action
    if list_experiments:
        _list_experiments()
    elif experiment:
        _run_experiment(experiment, verbose)
    elif analyze:
        _run_analysis()
    elif run_all:
        _run_all_experiments(verbose)


def _list_experiments() -> None:
    """List all available experiments."""
    click.echo("\nAvailable Chunking Experiments")
    click.echo("=" * 50)

    for name, description in EXPERIMENTS.items():
        # Check if results exist
        results_dir = get_results_dir(name)
        status = "✓ completed" if (results_dir / "summary.json").exists() else "○ pending"

        click.echo(f"\n{name}")
        click.echo(f"  {description}")
        click.echo(f"  Status: {status}")

    click.echo("\n" + "=" * 50)
    click.echo("Run with: python -m chunking.cli --experiment <name>")


def _run_experiment(experiment: str, verbose: bool) -> None:
    """Run a specific experiment."""
    config_path = get_config_path(experiment)

    click.echo(f"\n{'=' * 50}")
    click.echo(f"Running experiment: {experiment}")
    click.echo(f"Config: {config_path}")
    click.echo(f"{'=' * 50}\n")

    try:
        config = ExperimentConfig.from_yaml(config_path)
        runner = ExperimentRunner(config)
        results = runner.run()

        click.echo(f"\n{'=' * 50}")
        click.echo("Experiment completed successfully!")
        click.echo(f"Results saved to: {runner.results_dir}")

        # Print summary
        click.echo("\nSummary:")
        for config_result in results:
            click.echo(
                f"  {config_result.config_name}: "
                f"recall={config_result.evaluation.context_recall:.3f}, "
                f"precision={config_result.evaluation.context_precision:.3f}"
            )

    except Exception as e:
        raise click.ClickException(f"Experiment failed: {e}")


def _run_analysis() -> None:
    """Run comprehensive analysis on all experiments."""
    from .analysis import run_comprehensive_analysis

    click.echo(f"\n{'=' * 50}")
    click.echo("Running comprehensive analysis")
    click.echo(f"{'=' * 50}\n")

    try:
        results_dir = run_comprehensive_analysis()
        click.echo(f"\n{'=' * 50}")
        click.echo("Analysis complete!")
        click.echo(f"Results saved to: {results_dir}")
        click.echo(f"{'=' * 50}")

    except Exception as e:
        raise click.ClickException(f"Analysis failed: {e}")


def _run_all_experiments(verbose: bool) -> None:
    """Run all experiments sequentially."""
    click.echo(f"\n{'=' * 50}")
    click.echo("Running ALL chunking experiments")
    click.echo(f"{'=' * 50}\n")

    failed = []
    succeeded = []

    for experiment in EXPERIMENTS:
        click.echo(f"\n--- {experiment} ---")
        try:
            _run_experiment(experiment, verbose)
            succeeded.append(experiment)
        except click.ClickException as e:
            click.echo(f"FAILED: {e.message}", err=True)
            failed.append(experiment)

    # Summary
    click.echo(f"\n{'=' * 50}")
    click.echo("All experiments completed")
    click.echo(f"Succeeded: {len(succeeded)}")
    click.echo(f"Failed: {len(failed)}")

    if failed:
        click.echo(f"\nFailed experiments: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
