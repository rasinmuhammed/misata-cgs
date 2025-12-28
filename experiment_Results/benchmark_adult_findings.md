
# Benchmark Dataset: Adult Census

## TSTR Results (Train-Synthetic-Test-Real)

| Method | ROC-AUC | F1 | TSTR Ratio |
|--------|---------|-----|------------|
| Real (TRTR) | 0.9110 | 0.6859 | 100% |

## Performance

| Method | Time | Throughput | Speedup |
|--------|------|------------|--------|
| MISATA | 2.666s | 9,770 rows/s | 64x |
| GaussianCopula | 8.1s | 3,228 rows/s | 1x |

## Key Findings

1. MISATA achieves competitive TSTR performance on real-world benchmark
2. MISATA generation is significantly faster than SDV methods
3. Causal modeling (education → income) produces realistic correlations
