# RAG Experiments

Controlled experiments answering: **When do RAG optimizations actually help?**

## Experiments

| # | Question | Status |
|---|----------|--------|
| 1 | [Does chunk SIZE matter more than chunk STRATEGY?](chunking/) | Done |
| 2 | Do complex queries need complex retrieval? | Planned |
| 3 | Can embedding + chunking be optimized independently? | Planned |
| 4 | Does scale change optimal strategy? | Planned |

## Experiment 1: Chunking

**Finding**: Chunk SIZE matters more than chunk STRATEGY.

| Strategy | Requested | Actual | Recall |
|----------|-----------|--------|--------|
| token | 1024 | 934 | 75% |
| recursive | 1024 | 667 | 68% |
| **sentence** | 1024 | **3677** | 98% |

Sentence chunking ignores your config and produces 3.6x larger chunks.

See [`chunking/`](chunking/) for code and raw data.

Full analysis: [Substack Article](SUBSTACK_URL)

## Quick Start

```bash
uv sync
export OPENAI_API_KEY=sk-...

./run.sh chunking --list        # See experiments
./run.sh chunking --analyze     # Run analysis
```

## License

MIT
