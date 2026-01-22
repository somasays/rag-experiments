# Cross-Dataset Validation

## Natural Questions (avg chunk: 1587 chars)

| Strategy | Chunk Size | Recall |
|----------|------------|--------|
| token | 941 | 0.658 |
| sentence | 3652 | 0.856 |
| recursive | 901 | 0.681 |
| semantic | 853 | 0.700 |

**Correlation: r = 0.976, p = 0.0235**

## Chunk Size Correlation Experiment

| Config | Chunk Size | Recall |
|--------|------------|--------|
| token_1000_nq | 918 | 0.567 |
| token_2000_nq | 1914 | 0.822 |
| token_3000_nq | 2912 | 0.864 |

**Correlation: r = 0.923, p = 0.2509**
