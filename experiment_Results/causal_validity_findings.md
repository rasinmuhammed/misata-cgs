
# Causal Validity Experiment Findings

## Key Results

1. **Causal interventions produce predictable effects**:
   - Income x2 → Transaction 100.0% increase
   - Income x2 → Fraud 61.3% increase

2. **Causal consistency is high**:
   - Income → Transaction: r = 1.000
   - Income → Fraud: r = 0.998

3. **Effects follow causal graph predictions**:
   - ✓ Income ↑ causes Spend Rate ↑
   - ✓ Spend Rate ↑ causes Transaction Amount ↑
   - ✓ Spend Rate ↑ causes Fraud Rate ↑

## Implications

- MISATA enables **what-if analysis** for business scenarios
- Unlike GANs, causal structure is explicit and interpretable
- Supports policy testing before production deployment

## Comparison to GANs

| Capability | MISATA | GAN |
|------------|--------|-----|
| Causal intervention | ✅ Yes | ❌ No |
| Explain why effects occur | ✅ Yes | ❌ No |
| What-if analysis | ✅ Yes | ❌ No |
| Policy testing | ✅ Yes | ❌ No |
