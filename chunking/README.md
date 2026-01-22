# Chunking Experiments

Controlled experiments comparing RAG chunking strategies.

**Finding**: Chunk SIZE matters more than chunk STRATEGY.

## Structure

```
chunking/
├── strategies/         # Chunking implementations
│   ├── token.py        # Fixed token count (most precise)
│   ├── recursive.py    # Hierarchical splits (LangChain default)
│   ├── sentence.py     # Sentence boundaries (ignores chunk_size!)
│   └── semantic.py     # Embedding-based boundaries
├── datasets/           # Data loaders
│   ├── hotpotqa.py     # Multi-hop QA (3-9K char docs)
│   └── natural_questions.py  # Wikipedia (96K char docs)
├── configs/            # Experiment configurations
├── results/            # Raw experiment data
├── runner.py           # Experiment orchestration
├── evaluation.py       # RAGAS metrics wrapper
├── analyze.py          # Statistical analysis
└── cli.py              # Command-line interface
```

## Usage

```bash
# List experiments
./run.sh chunking --list

# Run an experiment
./run.sh chunking --experiment smoke_test

# Run analysis (generates all reports)
./run.sh chunking --analyze
```

## Experiments

| Experiment | Question |
|------------|----------|
| `document_length` | Does doc length affect optimal strategy? |
| `chunk_size_controlled` | When sizes are equal, which strategy wins? |
| `cross_dataset` | Do findings generalize across datasets? |
| `chunk_size_correlation` | Is size vs recall relationship linear? |

## Results

See the full analysis and findings: [Substack Article](SUBSTACK_URL)

Raw data available in `results/` directory.
