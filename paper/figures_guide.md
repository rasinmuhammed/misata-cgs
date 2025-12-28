# MISATA Paper Figures

## Required Figures for Paper

Generate these from experiment results:

### Figure 1: Performance Comparison (Section 4.2)
**Source**: `baseline_benchmark_results.csv`, `misata_benchmark_results.csv`
**Visualization**: 
- Bar chart: Throughput (rows/sec) for each generator at 1M scale
- Log scale Y-axis to show 856× difference
- Include memory usage as secondary metric

### Figure 2: ML Efficacy Comparison (Section 4.3)
**Source**: `ml_efficacy_results.csv`
**Visualization**:
- Grouped bar chart: ROC-AUC, Precision, Recall, F1 for each method
- Highlight MISATA's precision advantage (0.917 vs 0.247)

### Figure 3: Chaos Resilience Curves (Section 4.4)
**Source**: `chaos_resilience_results.csv`
**Visualization**:
- Line charts showing AUC retention vs chaos severity
- Separate panel for each chaos type
- Critical failure threshold line at 80%

### Figure 4: Causal Intervention Effects (Section 4.5)
**Source**: `causal_intervention_results.csv` (from Experiment 7)
**Visualization**:
- Scatter plot: Income multiplier vs downstream effects
- Show linear relationship (r > 0.99)

### Figure 5: LLM Persona Distribution (Section 3.4)
**Source**: `llm_vs_random_comparison.png` (from Experiment 6)
**Visualization**:
- Distribution comparison: LLM-guided vs Random
- Fraud rate by persona type

---

## Existing Figures

Already generated (download from Kaggle):
- `baseline_performance_comparison.png`
- `misata_performance.png`
- `statistical_fidelity_comparison.png`
- `ml_efficacy_comparison.png`
- `chaos_resilience_curves.png`
- `chaos_heatmap.png`
- `correlation_comparison.png`
- `distribution_comparison.png`

---

## Figure Generation Script

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# Load data
baseline_df = pd.read_csv('experiment_Results/baseline_benchmark_results.csv')
misata_df = pd.read_csv('experiment_Results/misata_benchmark_results.csv')
ml_df = pd.read_csv('experiment_Results/ml_efficacy_results.csv')
chaos_df = pd.read_csv('experiment_Results/chaos_resilience_results.csv')

# Figure 1: Performance
fig, ax = plt.subplots(figsize=(10, 6))
# ... (plotting code)
plt.savefig('paper/figures/fig1_performance.png', dpi=300, bbox_inches='tight')

# Figure 2: ML Efficacy
fig, ax = plt.subplots(figsize=(10, 6))
# ... (plotting code)
plt.savefig('paper/figures/fig2_ml_efficacy.png', dpi=300, bbox_inches='tight')
```

---

## LaTeX Figure Includes

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1_performance.png}
\caption{Performance comparison at 1M rows. MISATA achieves 856× speedup over SDV/CTGAN while using 13,000× less memory.}
\label{fig:performance}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_ml_efficacy.png}
\caption{ML efficacy comparison (TSTR protocol). MISATA achieves 99.4\% of real-data ROC-AUC with 3.7× higher precision than CTGAN.}
\label{fig:ml_efficacy}
\end{figure}
```
