# MISATA-CGS: Democratizing Causal Simulation via Copula-Guided Synthesis

**Moving Synthetic Tabular Data from Distribution Matching (Rung 1) to Causal Intervention (Rung 2)**

---

## Abstract

Current state-of-the-art synthetic tabular data methods, including 2025 foundation models like TabDiff (ICLR 2025) and TABSYN, excel at **distribution matching**—Pearl's Rung 1 (Association). However, they remain opaque regarding the underlying data-generating mechanism, making them unsuitable for **causal simulation** ("What happens if we intervene on X?"). 

We present **MISATA-CGS**, a framework that democratizes causal simulation by combining the statistical efficiency of Gaussian copulas with the causal validity of learned structural models.

**Key Contributions:**

1. **From Association to Intervention**: First framework enabling population-level `do(X=x)` interventions (Rung 2) for synthetic tabular data, with r=0.97 causal effect recovery against ground-truth SCMs.

2. **Democratized Access**: Achieves 26× speedup over CTGAN (0.59s vs 31.6s) on standard CPU hardware, removing the GPU barrier for causal analysis.

3. **Transparent Mechanism**: Unlike latent diffusion models, MISATA's causal graph is explicit and verifiable—critical for high-stakes domains like healthcare and policy.

4. **Competitive Fidelity**: 98.1% TSTR ratio on Adult Census benchmark, demonstrating that causal validity does not require sacrificing statistical quality.

> **Scope**: MISATA is designed for *causal simulation and decision support*, not privacy-preserving data release. Optimal for datasets with <50 features.

---

## 1. Introduction: The Causal Gap in Generative AI

As of late 2025, Generative AI has largely solved the "fidelity" problem for tabular data. Models like **TabDiff (ICLR 2025)** and **TABSYN** generate synthetic records statistically indistinguishable from real data. Yet for decision-making in healthcare, policy, and business, **fidelity alone is insufficient**.

Practitioners don't just want to know *what the world looks like* (correlation); they need to know *what happens if they act* (causation). Current deep generative models remain at **Rung 1 of Pearl's Ladder of Causation**: they model joint distributions P(X, Y) but cannot answer intervention queries P(Y | do(X)).

### The Research Gap

| Capability | TabDiff/TABSYN | CTGAN/TVAE | MISATA-CGS |
|------------|----------------|------------|------------|
| Distribution matching | ✅ SOTA | ✅ Good | ✅ Strong |
| do(X=x) interventions | ❌ | ❌ | ✅ |
| Interpretable mechanism | ❌ | ❌ | ✅ |
| CPU-only execution | ❌ | ⚠️ | ✅ |
| Treatment effect estimation | ❌ | ❌ | ✅ |

**MISATA-CGS** addresses this fundamental gap by explicitly modeling causal mechanisms via conditional distributions P(X_i | PA_i) while preserving complex correlations via Gaussian copulas.

---

## 2. Method: The Causal Simulation Engine

### 2.1 Architecture Overview

MISATA operationalizes the Structural Causal Model (SCM) framework in three phases:

```
Phase 1: Structure Learning (The Prior)
├── LLM-Assisted DAG Extraction: Domain text → Causal graph
└── Human Validation: Expert review of extracted edges

Phase 2: Mechanism Learning (The Conditionals)
├── For each node X_i given parents PA_i:
│   └── Learn P(X_i | PA_i) via GradientBoosting
└── Preserves functional causal mechanism f_i(PA_i, U_i)

Phase 3: Dependency Learning (The Copula)
├── Gaussian copula estimation on transformed residuals
└── Captures unmodeled confounders and complex correlations
```

### 2.2 LLM as Domain Knowledge Compiler

Causal discovery from data alone is notoriously hard. We leverage LLMs (Llama 3.3) not as "causal discoverers" but as **domain knowledge compilers**—translating standard domain descriptions (medical guidelines, economic theory) into verifiable DAG structures.

**Critical Framing**: The LLM parses causal *language* into formal structure; it does not discover novel causal relationships. Users must validate the extracted graph.

### 2.3 Intervention Mechanism

MISATA enables stepping up Pearl's Ladder from Rung 1 to Rung 2:

- **Observation** (Rung 1): P(Y | X=x) — Standard ML inference
- **Intervention** (Rung 2): P(Y | do(X=x)) — MISATA Simulation

By structurally modifying the graph (breaking edges to parents of intervention targets), MISATA generates correct interventional distributions.

---

## 3. Experiments

### 3.1 Experimental Setup

- **Datasets**: Adult Census (32K×15), California Housing (20K×9), Fraud Detection (50K×7), Cover Type (581K×54)
- **Baselines**: CTGAN, GaussianCopula, TVAE (validated), TabDDPM (literature values)
- **Metrics**: TSTR ratio, causal effect correlation, timing (fit + generate)
- **Hardware**: Standard CPU (no GPU)

### 3.2 Speed & Accessibility

| Method | Total Time | Speedup | Hardware |
|--------|------------|---------|----------|
| **MISATA-CGS** | **0.59s** | **1.0×** | CPU |
| GaussianCopula | 5.1s | 0.12× | CPU |
| CTGAN | 31.6s | 0.02× | CPU/GPU |
| TabDDPM† | 630s | 0.001× | GPU Required |

> †TabDDPM timing from ICML 2023 paper (literature values, not direct comparison)

### 3.3 Statistical Fidelity

| Dataset | TSTR Ratio | Correlation Recovery |
|---------|------------|---------------------|
| Adult Census | 98.1% | 99.2% |
| California Housing | 91.3% | 92.2% |
| Fraud Detection | 99.9% | 99.9% |
| Cover Type (54 dims) | 91.0% | 88.7% |

> **Limitation**: Performance degrades ~10% for high-dimensional data (>50 features) due to copula estimation instability.

### 3.4 Causal Validity

Tested against external ground-truth SCM (not fitted on):

| Metric | Result | Interpretation |
|--------|--------|----------------|
| Effect Correlation | **r = 0.97** | Model learned true causal mechanism |
| ATE Estimation | 34.6% ± 5.2% | Accurate population-level effects |
| LLM DAG F1 | 100% | On known domains (economics, medicine) |

---

## 4. Limitations & Failure Modes

### 4.1 Privacy

- **MIA AUC: 0.87-0.91** — Near-perfect membership detection
- **Implication**: MISATA is NOT suitable for privacy-preserving data release
- **Intended Use**: Internal simulation, policy testing, data augmentation

### 4.2 High Dimensionality

- Gaussian copula estimation requires O(d²) correlation matrix
- Performance degrades significantly for d > 50 features
- **Mitigation**: PCA preprocessing available but sacrifices interpretability

### 4.3 Causal Graph Dependency

- Results are only as good as the provided causal graph
- LLM extraction works for textbook domains; novel domains require expert validation
- Wrong graph → Wrong interventional estimates

---

## 5. Related Work

### Distribution Matching (Rung 1)
- **TabDiff (ICLR 2025)** & **TABSYN**: SOTA fidelity via diffusion/VAE, but opaque mechanisms
- **ProgSyn (ICML 2025)**: Adds constraints but lacks explicit intervention logic

### Causal Inference (Rung 2)  
- **STEAM (NeurIPS 2025)**: Excellent for *evaluating* causal estimators; MISATA generates the *simulation data*

**Positioning**: MISATA bridges statistical efficiency with causal capability—offering the speed of traditional methods with the Rung 2 capability that deep models lack.

---

## 6. Conclusion

As generative AI matures from pure generation to reasoning, synthetic data must follow. **MISATA-CGS** provides the necessary tooling to move tabular synthesis from static associations to dynamic causal simulations.

**The future of synthetic data is not just privacy—it's simulation.**

---

## Reproducibility

- **Code**: https://github.com/rasinmuhammed/misata-cgs
- **Hardware**: Runs on any modern CPU
- **Notebooks**: 24 experiments covering all claims

---

## Citation

```bibtex
@article{misata2025,
  title={MISATA-CGS: Democratizing Causal Simulation via Copula-Guided Synthesis},
  author={[Muhammed Rasin]},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```
