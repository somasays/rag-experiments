#!/bin/bash
# run.sh - Unified entry point for RAG experiments
#
# Usage:
#   ./run.sh chunking [OPTIONS]        Run chunking experiments
#   ./run.sh chunking --list           List available experiments
#   ./run.sh chunking -e NAME          Run specific experiment
#   ./run.sh chunking --analyze        Run comprehensive analysis
#   ./run.sh chunking --all            Run all experiments
#
# Examples:
#   ./run.sh chunking --list
#   ./run.sh chunking --experiment document_length
#   ./run.sh chunking --analyze
#   ./run.sh chunking --all

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo -e "${BLUE}RAG Experiments Runner${NC}"
    echo ""
    echo "Usage: $0 <module> [options]"
    echo ""
    echo "Modules:"
    echo "  chunking    Run chunking strategy experiments"
    echo ""
    echo "Chunking Options:"
    echo "  --list, -l           List available experiments"
    echo "  --experiment, -e     Run specific experiment"
    echo "  --analyze, -a        Run comprehensive analysis"
    echo "  --all                Run all experiments"
    echo "  --verbose, -v        Enable verbose output"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 chunking --list"
    echo "  $0 chunking --experiment document_length"
    echo "  $0 chunking --analyze"
    echo "  $0 chunking --all"
    exit 0
}

# Check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: uv is not installed${NC}"
        echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Run chunking module
run_chunking() {
    echo -e "${GREEN}Running chunking experiments...${NC}"
    uv run python -m chunking.cli "$@"
}

# Main entry point
main() {
    # Check for help flag
    if [[ "$1" == "--help" || "$1" == "-h" || -z "$1" ]]; then
        usage
    fi

    # Check prerequisites
    check_uv

    # Route to appropriate module
    case "$1" in
        chunking)
            shift
            run_chunking "$@"
            ;;
        *)
            echo -e "${RED}Unknown module: $1${NC}"
            echo ""
            usage
            ;;
    esac
}

main "$@"
