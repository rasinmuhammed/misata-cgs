# JAX Ecosystem Survey: Differentiable Simulation Libraries

## Why JAX?
> **JIT compilation via XLA** + **vmap vectorization** = Python usability with C++ performance

---

## Landscape Overview

```
                    ┌─────────────────────────────────────┐
                    │         JAX SIMULATION STACK        │
                    ├─────────────────────────────────────┤
                    │                                     │
   Physics ────────►│  JAX-MD (Molecular Dynamics)        │
                    │                                     │
   Gaming ─────────►│  JaxMARL, PureJaxRL (RL/Games)      │
                    │                                     │
   Economics ──────►│  EconoJax, JAX-LOB (Finance)        │
                    │                                     │
   General ABM ────►│  ABMax, Foragax (Prototypes)        │
                    │                                     │
   Business ───────►│  ??? GAP → MISATA                   │
                    │                                     │
                    └─────────────────────────────────────┘
```

---

## 1. JAX-MD (Molecular Dynamics)

| Aspect | Details |
|--------|---------|
| **Domain** | Particle physics, proteins |
| **Scale** | 100,000s of particles on single GPU |
| **Key Innovation** | Differentiable neighbor lists, end-to-end gradients |
| **Limitation for Misata** | Continuous physics ≠ Discrete business logic |

### Relevance
- ✅ Proves JAX scales to massive interacting agents
- ✅ Neighbor list pattern applicable to spatial business queries
- ❌ Newtonian integrators incompatible with transaction logic

---

## 2. JaxMARL & PureJaxRL (Gaming/RL)

| Aspect | Details |
|--------|---------|
| **Domain** | StarCraft, Hanabi, Overcooked, Gridworlds |
| **Speedup** | Up to **12,500x** vs CPU baselines |
| **Key Innovation** | Environment + Policy training on same GPU |

### Architecture Insight
```
Traditional RL:
  CPU (Env) ←───PCIe───→ GPU (Model)  ← SLOW

JaxMARL:
  GPU (Env + Model)  ← ZERO-COPY, FAST
```

### Relevance
- ✅ Validates complex agent logic can be vectorized
- ✅ Demonstrates "Zero-Copy" philosophy power
- ❌ Fixed action spaces, simple state representations
- ❌ Focus on *training* agents, not *generating data*

---

## 3. EconoJax (AI Economist)

| Aspect | Details |
|--------|---------|
| **Domain** | Tax policy, resource gathering economy |
| **Training Time** | Days → Minutes |
| **State** | Simplified 1D representation |

### Limitations (Author-Acknowledged)
- Functional purity requirement restricts OOP agent behaviors
- Too simplified for enterprise schema complexity

### Relevance
- ✅ Proves JAX handles economic logic
- ❌ Highly specialized, not general-purpose

---

## 4. JAX-LOB (Limit Order Book)

| Aspect | Details |
|--------|---------|
| **Domain** | High-frequency trading, order matching |
| **Scale** | Thousands of order books in parallel |

### Relevance
- ✅ Financial matching logic works in JAX
- ❌ Cannot simulate healthcare, supply chain, etc.

---

## 5. ABMax & Foragax (General ABM)

### The Static Shape Problem
> JAX JIT requires **array sizes known at compile time**
> 
> But ABMs need **dynamic populations** (birth/death)

### ABMax Solution
```
Fixed Array with Masking:
┌───────────────────────────────────────────┐
│ [Agent1] [Agent2] [MASKED] [MASKED] ...   │
│     ↑        ↑         ↑         ↑        │
│   Active   Active   Inactive  Inactive    │
└───────────────────────────────────────────┘
```

**Algorithms**: Rank-Match, Sort-Count-Iterate

### Performance Reality (ABMax Authors)
| Platform | Best Choice |
|----------|-------------|
| CPU single run | Agents.jl (dynamic arrays native) |
| GPU batch (10,000 sims) | ABMax (padding overhead amortized) |

### Relevance
- ✅ Proves general ABM possible in JAX
- ✅ Padding/masking pattern adopted by Misata
- ⚠️ Academic prototype, not production-ready

---

## Strategic Gap Identified

| Domain | JAX Solution | Maturity |
|--------|--------------|----------|
| Physics | JAX-MD | ⭐⭐⭐⭐⭐ |
| Gaming/RL | JaxMARL | ⭐⭐⭐⭐⭐ |
| Economics | EconoJax | ⭐⭐⭐ |
| Finance | JAX-LOB | ⭐⭐⭐ |
| General ABM | ABMax/Foragax | ⭐⭐ |
| **Business Data Gen** | **??? → MISATA** | ⭐ (gap) |

---

## Misata Positioning
> **Combine Mesa usability + JAX performance**
> 
> For general-purpose, tabular synthetic data generation

---

## References
- JAX-MD: [GitHub](https://github.com/jax-md/jax-md)
- JaxMARL: [GitHub](https://github.com/FLAIROx/JaxMARL)
- PureJaxRL: [GitHub](https://github.com/luchris429/purejaxrl)
- EconoJax: [Paper](https://arxiv.org/abs/2xxx)
- ABMax: [arXiv](https://arxiv.org/html/2508.16508v3)
