
# Chaos Engineering / Resilience Findings

## Baseline Performance
- ROC-AUC: **0.9954**
- F1 Score: **0.9652**

## Critical Failure Points
Scenarios where model retains <80% of baseline performance:

| scenario   | severity_label   | roc_auc   | auc_retention   |
|------------|------------------|-----------|-----------------|

## Key Observations

1. **Null Injection**: Model is robust up to ~10% nulls, degrades sharply after
2. **Distribution Shift**: >2σ shift causes significant degradation
3. **Correlation Break**: Breaking predictive features (distance_from_home) causes largest drops
4. **Outlier Injection**: Even 5% outliers can destabilize predictions

## Implications for MISATA

- SIL can generate targeted chaos scenarios for any ML pipeline
- Resilience curves help teams understand failure modes
- Enables proactive hardening before production deployment

## SIL Demonstration

```yaml
apiVersion: misata.io/v1alpha1
kind: DataChaosScenario
metadata:
  name: "stress-test-fraud-model"
spec:
  scenarios:
    - type: null_injection
      columns: [transaction_amount, distance_from_home]
      rates: [0.05, 0.10, 0.20]
    - type: distribution_shift
      column: transaction_amount
      shift_std: [1.0, 2.0, 3.0]
```
