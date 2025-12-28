# RADS: Reactive Agentic Data System Architecture

## Core Concept
> A "row" of data = **state of an agent at a point in time**
>
> System simulates interaction of millions of agents to *emerge* complex datasets (not sample from fixed distributions)

---

## 1. Compute Backbone: JAX + XLA

### Why JAX Over NumPy/PyTorch

| Feature | Benefit |
|---------|---------|
| **JIT Compilation (XLA)** | Eliminates Python interpreter overhead |
| **vmap (Auto-Vectorization)** | Write single-agent logic → auto-parallelize to millions |
| **PRNG Key Splitting** | Deterministic reproducibility for chaos engineering |
| **Hardware Agnostic** | Same code → CPU/GPU/TPU |

### Design Patterns: ABMax & Foragax
- **Struct of Arrays (SoA)**: Agent states as tensors (not object lists)
  - ❌ `List[Agent]` (cache-inefficient)
  - ✅ `Tensor['health', N]`, `Tensor['position', N]`
- Enables XLA kernel fusion and maximum memory bandwidth

---

## 2. Zero-Copy Data Transport

```
┌─────────────┐    DLPack    ┌─────────────┐
│  JAX Array  │ ────────────►│ Arrow Table │
│  (GPU/CPU)  │   __dlpack__ │ (zero-copy) │
└─────────────┘              └──────┬──────┘
                                    │
                                    ▼
                            ┌─────────────┐
                            │   Polars    │
                            │  DataFrame  │
                            └─────────────┘
```

**Key**: `pyarrow.from_dlpack(jax_array)` → same memory address, no copy, no serialization

### Control Plane (Rust)
- Manages simulation lifecycle
- Triggers JAX compute kernel
- Receives DLPack pointer
- Exposes data as Polars DataFrame

---

## 3. Categorical Data Handling

### Challenge: JAX/XLA lacks native string support

### Solutions

| Approach | Use Case | Trade-off |
|----------|----------|-----------|
| **Integer Mapping** | All categoricals | Metadata sidecar maintains bijection |
| **Deterministic Hashing** | High-cardinality (Transaction IDs) | Bounded collision rate |
| **Learnable Embeddings** | ML-integrated simulations | Dense vectors capture semantics |

### Arrow Dictionary Encoding
Apply string mapping only at export phase (Rust/Python metadata registry)

---

## 4. Differentiable Logic & Constraint Learning

### Soft Logic Relaxation
- Libraries: `difflogic`
- `if x > 0` → smooth sigmoid function
- Allows gradients to flow through decisions

### Optimization as a Layer
- `cvxpylayers`, `TorchOpt`
- Embed convex optimization in forward pass
- Generator "solves" for valid projection onto constraint manifold

### Neuro-Symbolic Integration
- **Neural (JAX)**: Learns complex high-dimensional distribution
- **Symbolic (DiffLogic)**: Enforces hard domain constraints (physics, business rules)

---

## References
- JAX: [GitHub](https://github.com/jax-ml/jax)
- ABMax: [arXiv](https://arxiv.org/html/2508.16508v3)
- Foragax: [ResearchGate](https://www.researchgate.net/publication/383917409)
- DLPack Arrow: [Apache Arrow Docs](https://arrow.apache.org/docs/python/dlpack.html)
- DiffLogic: [GitHub](https://github.com/Felix-Petersen/difflogic)
- cvxpylayers: [GitHub](https://github.com/cvxpy/cvxpylayers)
