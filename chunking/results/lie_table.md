# Chunk Size: Requested vs Actual

| Config | Strategy | Requested | Actual | Ratio | Recall |
|--------|----------|-----------|--------|-------|--------|
| sentence_default | sentence | 1024 | 3676 | 3.59 !!! | 0.975 |
| sentence_3000 | sentence | 3000 | 8150 | 2.72 !!! | 0.950 |
| semantic_1000 | semantic | 1000 | 1117 | 1.12  | 0.700 |
| semantic_default | semantic | 1024 | 1117 | 1.09  | 0.775 |
| token_3000 | token | 3000 | 2799 | 0.93  | 0.975 |
| token_1000 | token | 1000 | 912 | 0.91  | 0.775 |
| token_default | token | 1024 | 933 | 0.91  | 0.750 |
| recursive_3000 | recursive | 3000 | 2295 | 0.77  | 0.900 |
| recursive_default | recursive | 1024 | 666 | 0.65  | 0.675 |
| recursive_1000 | recursive | 1000 | 643 | 0.64  | 0.725 |
| semantic_3000 | semantic | 3000 | 1117 | 0.37  | 0.718 |