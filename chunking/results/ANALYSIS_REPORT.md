# RAG Chunking Experiments: Statistical Analysis Report

## Executive Summary

This report analyzes 4 experiments with configurations testing chunking strategies
for RAG retrieval across HotpotQA and Natural Questions datasets.

### Overall Statistics

- **Total configurations tested:** 30
- **Mean context_recall:** 0.8105
- **Std dev:** 0.1212
- **Range:** 0.5667 - 1.0000

### Top 5 Configurations (by context_recall)

| Rank | Config | Strategy | Chunk Size | Recall | Dataset |
|------|--------|----------|------------|--------|---------|
| 1 | sentence_short | sentence | 3226 | 1.0000 | hotpotqa |
| 2 | token_3000 | token | 2800 | 0.9750 | hotpotqa |
| 3 | sentence_default | sentence | 3677 | 0.9750 | hotpotqa |
| 4 | sentence_3000 | sentence | 8150 | 0.9500 | hotpotqa |
| 5 | recursive_short | recursive | 774 | 0.9500 | hotpotqa |

### Key Findings

1. **Chunk size is the dominant factor** (r = 0.57 correlation with recall)
   - Larger chunks (2674+ chars) average 94.0% recall vs 80.6% for smaller chunks (<870 chars)
   - Per-dataset correlations: hotpotqa: r=0.53, natural_questions: r=0.98

2. **Sentence chunking achieves best mean recall** (94.7%)
   - Sentence chunks average 4244 chars
   - Other strategies average 1129 chars

3. **Document length affects performance**
   - Short documents: 95.0% mean recall
   - Long documents: 77.5% mean recall

4. **Semantic chunking performance**
   - Semantic chunking: 75.6% mean recall
   - Other strategies average: 83.9%
   - Underperforms other strategies, possibly due to smaller chunk sizes

### Practical Recommendations

*Based on observed performance across all configurations:*

| Chunk Size Range | Best Strategy | Observed Recall |
|------------------|---------------|-----------------|
| <870 chars | recursive | 95.0% |
| 870-979 chars | token | 80.0% |
| 979-2674 chars | recursive | 90.0% |
| 2674+ chars | sentence | 100.0% |

**Best overall configuration:** sentence_short (sentence, 3226 chars) with 100.0% recall

---

## Correlation Analysis

### Chunk Size vs Recall

**Pearson r = 0.5747** (strong positive correlation)

Larger chunks systematically improve retrieval quality.

---

## Experiment: Chunk Size Controlled (HotpotQA)

**Description:** Controlled chunk size comparison to isolate strategy effects
**Dataset:** hotpotqa
**Configurations:** 11

### Pivot Table: Strategy -> context_recall

| Strategy | Mean Recall | Std Dev | Min | Max | N |
|----------|-------------|---------|-----|-----|---|
| recursive | 0.7667 | 0.1181 | 0.6750 | 0.9000 | 3 |
| semantic | 0.7310 | 0.0392 | 0.7000 | 0.7750 | 3 |
| sentence | 0.9625 | 0.0177 | 0.9500 | 0.9750 | 2 |
| token | 0.8333 | 0.1233 | 0.7500 | 0.9750 | 3 |

### Chunk Size vs Recall Correlation

**Pearson r = 0.7382** (strong positive correlation)

### ANOVA: Strategy Effect

- **F-statistic:** 2.7413
- **eta-squared:** 0.5402 (large effect)

### Best vs Worst

**Best:** token_3000 (recall=0.9750, chunk_size=2800)
**Worst:** recursive_default (recall=0.6750, chunk_size=667)

---

## Experiment: Document Length Effects (HotpotQA)

**Description:** How document length affects chunking strategies
**Configurations:** 12

### Pivot Table: Document Length x Strategy -> context_recall

| Length | token | sentence | recursive | semantic |
|--------|-------|----------|-----------|----------|
| short | 0.9000 | 1.0000 | 0.9500 | 0.9500 |
| medium | 0.7500 | 0.9500 | 0.8000 | 0.8500 |
| long | 0.8000 | 0.9500 | 0.7500 | 0.6000 |

### ANOVA: Strategy Effect
- **F-statistic:** 1.3944
- **eta-squared:** 0.3434 (large effect)

---

## Experiment: Cross-Dataset (Natural Questions)

**Description:** Cross-dataset generalization from HotpotQA to Natural Questions
**Dataset:** natural_questions

### Results by Strategy

| Strategy | Chunk Size | Recall | Precision |
|----------|------------|--------|-----------|
| sentence | 3653 | 0.8556 | 0.7917 |
| semantic | 853 | 0.7000 | 0.6958 |
| recursive | 902 | 0.6806 | 0.6083 |
| token | 942 | 0.6583 | 0.7021 |

**Chunk size vs recall correlation:** r = 0.9765

---

## Experiment: Chunk Size Correlation (Natural Questions)

**Description:** Chunk size vs recall correlation on Natural Questions

### Results

| Config | Chunk Size | Recall |
|--------|------------|--------|
| token_1000_nq | 918 | 0.5667 |
| token_2000_nq | 1915 | 0.8222 |
| token_3000_nq | 2912 | 0.8639 |

**Correlation:** r = 0.9234
