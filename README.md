# MISATA-CGS: Copula-Guided Causal Synthesis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-rasinmuhammed/misata--cgs-blue?logo=github)](https://github.com/rasinmuhammed/misata-cgs)

> **Moving synthetic data from distribution matching (Rung 1) to causal intervention (Rung 2)**

MISATA-CGS is a novel causal simulation framework that combines Gaussian copulas for statistical fidelity with learned causal models for intervention capability. Unlike black-box generative models, MISATA enables `do(X=x)` interventions for policy simulation and decision support.

## 🎯 Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| **Speed** | **26x faster** than CTGAN | [Fair benchmark](experiments/01B_fair_performance_benchmark.ipynb) |
| **TSTR Ratio** | **98.1%** on Adult Census | [Multi-dataset eval](experiments/13_multi_dataset_evaluation.ipynb) |
| **Causal Recovery** | **r = 0.97** | [External SCM validation](experiments/07B_external_causal_validation.ipynb) |
| **LLM DAG F1** | **100%** on known domains | [Ground-truth test](experiments/12B_groundtruth_llm_dag.ipynb) |

> ⚠️ **Scope**: Optimal for datasets with <50 features. High-dimensional data (50+) shows ~10% TSTR degradation.

## 🚀 Why MISATA?

| Capability | MISATA | CTGAN/TVAE | TabDDPM |
|------------|--------|------------|---------|
| Distribution matching | ✅ | ✅ | ✅ |
| `do(X=x)` interventions | ✅ | ❌ | ❌ |
| Interpretable mechanism | ✅ | ❌ | ❌ |
| CPU-only (no GPU) | ✅ | ⚠️ | ❌ |
| Sub-second generation | ✅ | ❌ | ❌ |

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
from misata import MISATASynthesizer

# Fit on your data
synth = MISATASynthesizer(target_col='income')
synth.fit(train_df)

# Generate synthetic samples
synthetic_df = synth.sample(n_samples=1000)

# Causal intervention: do(education=16)
synth.intervene('education', 16)
intervention_df = synth.sample(n_samples=1000)
# Now see how income distribution changes under the intervention!
```

## ✨ Features

- 🏃 **Fast**: 0.59s total time (vs 31.6s for CTGAN) - no neural network training
- 🎯 **Causally Valid**: Interventional distributions via learned SCM structure  
- 🧠 **LLM Integration**: Parse domain knowledge → DAG (human-in-the-loop)
- 📊 **What-If Analysis**: Population-level treatment effect estimation

## 📁 Repository Structure

```
misata-cgs/
├── src/misata/          # Core library
│   └── synthesizers/    # MISATA-CGS, ConditionalIntervention
├── experiments/         # 24 Jupyter notebooks
├── experiment_Results/  # All CSV + figures
└── paper/               # arXiv draft
```

## 📚 Experiments

See [`experiments/`](experiments/) for all 24 notebooks validating our claims:

| Notebook | Purpose |
|----------|---------|
| `01B_fair_performance_benchmark` | Speed comparison (CTGAN, GaussianCopula) |
| `07B_external_causal_validation` | Causal recovery on known SCM |
| `13_multi_dataset_evaluation` | Cross-dataset generalization |
| `12B_groundtruth_llm_dag` | LLM DAG extraction accuracy |

## 📝 Citation

```bibtex
@article{misata2025,
  title={MISATA-CGS: Democratizing Causal Simulation via Copula-Guided Synthesis},
  author={Rasin, Muhammed},
  journal={arXiv preprint},
  year={2025}
}
```

## ⚖️ License

MIT

---

**Note**: MISATA is designed for *causal simulation*, not privacy-preserving data release. For privacy applications, consider methods with formal differential privacy guarantees.
