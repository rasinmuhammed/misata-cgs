# Baseline Performance Findings

## Performance Rankings (at 1M rows)
| name               |   time_seconds |   rows_per_second |
|:-------------------|---------------:|------------------:|
| NumPy-Vectorized   |         2.1229 |        471055     |
| Mesa-ABM           |        21.1584 |         47262.5   |
| SDV-GaussianCopula |      1038.06   |           963.337 |
| SDV-CTGAN          |      1063.07   |           940.672 |
| Faker              |      1068.29   |           936.076 |

## Key Observations
1. Mesa ABM hits memory limits at ~100K-500K rows due to Python object overhead
2. SDV-CTGAN is 10-50x slower than vectorized approaches due to neural network inference
3. NumPy-vectorized represents the ceiling for Python-ecosystem performance
4. All approaches are fundamentally limited by single-threaded execution (GIL)

## Implication for MISATA
- JAX compilation can bypass GIL â†’ potential 10-100x improvement
- Struct-of-Arrays layout â†’ better cache utilization than Mesa's object model
- GPU acceleration â†’ millions of agents in parallel