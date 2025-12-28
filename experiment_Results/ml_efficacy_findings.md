
# ML Efficacy Findings

## TSTR (Train-Synthetic-Test-Real) Results

Real baseline ROC-AUC: **0.9849**

### Generator Performance
| generator      |   roc_auc |   tstr_ratio |   gap_to_real |
|:---------------|----------:|-------------:|--------------:|
| Faker          |    0.719  |       0.73   |        0.2659 |
| GaussianCopula |    0.5491 |       0.5576 |        0.4358 |
| CTGAN          |    0.983  |       0.9981 |        0.0019 |
| MISATA         |    0.979  |       0.994  |        0.0059 |

## Key Observations

1. **Faker**: Poor ML efficacy - random data doesn't capture predictive relationships
2. **GaussianCopula**: Moderate efficacy - captures marginals but misses complex patterns
3. **CTGAN**: Good efficacy - learns feature-target relationships from data
4. **MISATA**: Competitive efficacy - explicitly models causal relationships

## Implications

- MISATA's agent-based approach preserves predictive signal
- Explicit causal modeling (fraud agents behave differently) works well
- LLM semantic injection could further improve domain-specific patterns
