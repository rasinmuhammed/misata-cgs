# Data Chaos Engineering: Defining the Discipline

## From Infrastructure Chaos to Data Chaos

| Domain | Infrastructure Chaos | Data Chaos |
|--------|---------------------|------------|
| **Target** | Containers (pods, VMs) | Content (data pipelines) |
| **Tool** | Chaos Monkey, Gremlin | Savage, Misata |
| **Fault** | Kill server, network latency | Statistical anomaly, schema drift |
| **Question** | "Does the service stay up?" | "Does the model stay valid?" |

---

## 1. The Savage Framework

### Paper: "Stress-testing ML pipelines with Adversarial Data Corruption"

### Data Corruption Processes (DCPs)

| DCP Type | Description | Example |
|----------|-------------|---------|
| **MCAR** | Missing Completely At Random | 5% random nulls |
| **MAR** | Missing At Random (conditional) | Nulls if Age > 60 |
| **MNAR** | Missing Not At Random (self-referential) | Nulls if Income is high |
| **Label Errors** | Systematic label corruption | Flip fraud labels in Region-East |
| **Covariate Shift** | Input distribution change | Mean shift in Transaction_Value |

### Adversarial Optimization
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Clean Data  │────►│ DCP(θ)      │────►│ ML Pipeline │
└─────────────┘     │ (corruption)│     └──────┬──────┘
                    └─────────────┘            │
                           ▲                   ▼
                           │            ┌─────────────┐
                           └────────────│ Maximize    │
                              Optimize  │ Model Error │
                                        └─────────────┘
```
**Key Insight**: Random noise insufficient → **Adversarial generation required**

---

## 2. Resilience Testing Definition

> Evaluate model **stability** under:
> - Distributional shifts
> - Adversarial perturbations  
> - Noisy/corrupted inputs
> - Schema changes

### Why Hold-out Accuracy is Insufficient

| Metric | Measures | Misses |
|--------|----------|--------|
| Test Accuracy | Average performance | Tail failures |
| Resilience | Worst-case stability | - |

---

## 3. Misata's SIL: Operationalizing Data Chaos

### Design Philosophy
- **Declarative**: YAML specification (like Chaos Mesh)
- **Compiled**: Injected into JAX computation graph
- **Zero-overhead**: Conditional logic compiled by XLA

### SIL Fault Types

| Type | Description | Use Case |
|------|-------------|----------|
| `parameter_override` | Shift distribution parameters | Market stress |
| `null_injection` | Insert NULLs with pattern | Sensor failure |
| `latency_injection` | Delay timestamps | Feed lag |
| `correlation_break` | Decorrelate paired columns | Logic breakdown |
| `schema_drift` | Add/remove/rename columns | Provider change |
| `outlier_injection` | Insert extreme values | Black Swan |

### Graph Injection Architecture
```
Normal Simulation:
  state_{t+1} = normal_update(state_t)

With SIL Injection:
  state_{t+1} = jax.lax.select(
      trigger_condition(state_t, step),
      apply_fault(state_t, fault_params),
      normal_update(state_t)
  )
```
**Result**: Branching compiled into single XLA kernel

---

## 4. Inverted Validation (Red Team)

### Concept
> Ingest data quality rules → Invert → Generate violations

### Workflow
```
┌─────────────────────────────────────────────┐
│ Great Expectations / Soda Config            │
│                                             │
│ expect_column_values_to_be_between(0, 100) │
│ expect_column_values_to_not_be_null         │
└─────────────────────┬───────────────────────┘
                      │ INVERT
                      ▼
┌─────────────────────────────────────────────┐
│ Generated SIL Attacks                       │
│                                             │
│ - type: outlier_injection                   │
│   range: [-50, 200]                         │
│   probability: 0.05                         │
│                                             │
│ - type: null_injection                      │
│   probability: 0.03                         │
└─────────────────────────────────────────────┘
```

### Closed-Loop Red Teaming
> Data engine actively tries to break validation pipeline
> 
> → Ensures robustness to failures systems are programmed to detect but rarely see

---

## 5. Why ABM Enables Data Chaos (GAN Cannot)

| Capability | GAN | ABM (Misata) |
|------------|-----|--------------|
| Generate normal data | ✅ | ✅ |
| Generate targeted anomalies | ❌ | ✅ |
| Simulate Black Swans | ❌ | ✅ |
| Conditional fault injection | ❌ | ✅ |
| Counterfactual scenarios | ❌ | ✅ |
| Deterministic replay | ⚠️ | ✅ |

---

## Key Insight
> **Chaos Engineering for Data** = Adversarial stress-testing of ML pipelines
> 
> **Requires**: Mechanism-based generation (ABM) 
> 
> **Not**: Distribution-mimicking (GAN)

---

## References
- Savage Framework: Adversarial data corruption
- Chaos Mesh: YAML-based infrastructure chaos
- Great Expectations: Data quality rules
- Soda: Data quality monitoring
