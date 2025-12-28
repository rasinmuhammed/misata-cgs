# SIL: Scenario Injection Language

## Purpose
> **Data Chaos Engineering**: Inject statistical anomalies, distribution drifts, and quality failures to test downstream analytics/ML resilience.

Inspiration: OpenSCENARIO (automotive), Chaos Mesh (Kubernetes)

---

## Key Abstractions

| Abstraction | Definition | Example |
|-------------|------------|---------|
| **Selector** | Blast Radius (target subset) | "All transactions in Region-East" |
| **Trigger** | Activation condition | "Simulation Step > 1000" |
| **Action** | Statistical fault | "Shift Mean", "Inject Nulls", "Invert Correlation" |

---

## Example SIL Specification

```yaml
apiVersion: misata.io/v1alpha1
kind: DataChaosScenario
metadata:
  name: "2025-Liquidity-Flash-Crash"
spec:
  # 1. BLAST RADIUS
  selector:
    entity: "market_maker_agent"
    filter: "risk_profile == 'HIGH_LEVERAGE'"
  
  # 2. TRIGGER
  trigger:
    type: "conditional"
    expression: "simulation_step >= 500 && simulation_step <= 550"

  # 3. ACTIONS (Fault Injections)
  actions:
    # A. Distribution Shift
    - type: "parameter_override"
      target: "sell_probability"
      method: "affine_transform"
      params: { scale: 5.0, offset: 0.2 }
    
    # B. Data Quality Failure (sensor lag)
    - type: "latency_injection"
      target: "timestamp"
      distribution: "exponential"
      lambda: 0.5
    
    # C. Correlation Break
    - type: "correlation_break"
      pair: ["bid_price", "ask_price"]
      force_value: "uncorrelated"
```

---

## Scenario Compiler: YAML → XLA

```
┌──────────────┐
│  YAML Parse  │
└──────┬───────┘
       ▼
┌──────────────────────────────────────┐
│  GRAPH INJECTION                     │
│  Identify target tensors in JAX state│
│  Insert: jax.lax.cond / jax.lax.select│
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  CODE GENERATION                     │
│  next_state = jax.lax.select(        │
│    trigger_condition,                │
│    apply_fault(state),               │
│    normal_update(state)              │
│  )                                   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  XLA JIT COMPILATION                 │
│  Single kernel, GPU-efficient        │
│  No host-device sync overhead        │
└──────────────────────────────────────┘
```

---

## Inverted Validation: Red Team Workflow

### Concept
Ingest data quality rules (Great Expectations, Soda) and **invert them** to generate attacks.

### Example

| Input (Great Expectations) | Output (SIL Attack) |
|---------------------------|---------------------|
| `expect_column_values_to_be_between(min=0, max=100)` | `InjectOutliers { range: [-50, 200], probability: 0.05 }` |
| `expect_column_values_to_not_be_null` | `InjectNulls { probability: 0.03 }` |

### Result
**Closed-loop Red Teaming**: Data engine actively tries to break validation pipeline → ensures robustness to failures systems are programmed to detect but rarely see.

---

## References
- OpenSCENARIO: [ASAM](https://www.asam.net/standards/detail/openscenario-dsl/)
- Chaos Mesh: [Docs](https://chaos-mesh.org/docs/)
- Great Expectations: [Docs](https://docs.greatexpectations.io/)
- Soda: [Docs](https://docs.soda.io/soda-cl-overview)
