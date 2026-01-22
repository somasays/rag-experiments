"""CLI API - wraps analyze.py for CLI usage."""

from pathlib import Path

from .analyze import main as run_analysis


def run_comprehensive_analysis() -> Path:
    """
    Run all analysis and return results directory.

    This function calls the analyze.py main() which:
    - Loads all experiment results
    - Generates statistical analysis
    - Creates charts, tables, and reports
    - Saves all output files to chunking/results/
    """
    run_analysis()
    return Path(__file__).parent / "results"
