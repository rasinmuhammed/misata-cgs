# GAN Limitations: The Fidelity Gap

## Core Problem
> GANs learn **correlations**, not **causes**
> 
> They **interpolate** observed data, cannot **extrapolate** to Black Swans

---

## 1. Temporal Incoherence

### Window-Based Myopia

```
TimeGAN Training:
  [t-24, t-23, ... t-1] → predict [t]
         ↑
   Rolling window (e.g., 24 steps)

Reality:
  Model captures LOCAL correlations
  Loses GLOBAL causal arc over 1000s of steps
```

### Failure Modes

| Issue | Manifestation |
|-------|--------------|
| **Drift** | Bank balance drifts below zero (no overdraft logic) |
| **Mode Collapse in Time** | Repetitive looping patterns, flat-lining |
| **Phase Shifts** | Requires DTW post-processing to align |

### Example: Synthetic Bank Balance
```
Real:    ████████████████▼████████████████
                         ↑
                    Overdraft event (causal)

GAN:     ████████████████░░░░░░░░░░░░░░░░░
                         ↑
                    Drifts aimlessly (no mechanism)
```

---

## 2. Causal Vacuity

### The Correlation Trap

```
Training Data:
  High Ice Cream Sales ←→ High Drowning Deaths
                    ↑
              Both caused by SUMMER (confound)

GAN Behavior:
  Generates high drowning whenever ice cream is high
  Cannot simulate: "What if swimming is banned?"

ABM Behavior:
  Models mechanism (people → swimming → risk)
  Intervention is trivial: remove swimming agents
```

### Counterfactual Impossibility

| Question | GAN | ABM |
|----------|-----|-----|
| "What if X never happened?" | ❌ Cannot reason | ✅ Remove agent action |
| "What caused Y?" | ❌ Only correlations | ✅ Trace causal chain |

---

## 3. The Black Swan Problem

### Interpolation vs Extrapolation

```
Training Distribution (2010-2019: Stable):

    ┌────────────────────────────────────┐
    │        ███████████████             │
    │       █████████████████            │
    │      ███████████████████           │
    │                                    │ ← GAN latent space
    └────────────────────────────────────┘
                    
    2008 Crash: ░░░░░ (NOT in training)
    COVID Shock: ░░░░░ (NOT in training)
```

### GAN Cannot Generate Unseen Extremes
- Training = stable decade
- Required = crash scenario
- Result: **GAN cannot extrapolate to unseen regimes**

### ABM Solution
> Push agent behaviors to limits → **Black Swans emerge endogenously**
> 
> "What if everyone withdraws at once?" = Bank Run simulation

---

## 4. Tabular Data Failures

### Imbalanced Data (Fraud Detection)

| Training | Fraud Rate | GAN Behavior |
|----------|------------|--------------|
| Real Data | 0.5% | |
| GAN Output | ~0% | Mode collapse to majority class |

**Result**: Synthetic data "cleanses" the anomalies needed for risk modeling

### Constraint Violations

| Constraint | GAN | ABM |
|------------|-----|-----|
| `start_date < end_date` | May violate → rejection needed | Enforced by design |
| `total = part_a + part_b` | Approximate | Exact |
| `balance >= 0 OR overdraft` | Ignores logic | Agent state machine |

---

## 5. Privacy-Utility Trade-off

### Differential Privacy (DP-SGD)
```
Noise injection → protects privacy
             BUT → destroys outlier fidelity

Outliers = Black Swans = What we need for chaos testing
```

---

## Summary: Why GANs Fail for Chaos Engineering

| Requirement | GAN Capability | Rating |
|-------------|----------------|--------|
| Temporal coherence (1000+ steps) | Window-limited | ❌ |
| Causal reasoning | Correlation only | ❌ |
| Black Swan generation | Interpolation only | ❌ |
| Imbalanced data | Mode collapse | ❌ |
| Hard constraints | Approximate | ⚠️ |
| Outlier fidelity (with DP) | Privacy trade-off | ❌ |

---

## Key Insight
> **GANs generate data that LOOKS real**
> 
> **ABMs simulate agents that BEHAVE consistently**
> 
> Resilience testing requires the latter.

---

## References
- TimeGAN: Temporal structure via rolling windows
- CTGAN: Tabular GAN struggles with imbalance
- DTW post-processing: Alignment artifacts
- Mode collapse literature
