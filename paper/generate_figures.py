#!/usr/bin/env python3
"""
Generate publication-ready figures for MISATA paper.
Run this script after collecting all experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Paths
RESULTS_DIR = Path("experiment_Results")
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'MISATA': '#2ecc71',      # Green
    'CTGAN': '#e74c3c',       # Red
    'GaussianCopula': '#3498db',  # Blue
    'Faker': '#95a5a6',       # Gray
    'Mesa-ABM': '#f39c12',    # Orange
    'NumPy': '#9b59b6',       # Purple
    'Real': '#1a1a2e',        # Dark
}


def load_data():
    """Load all experiment results."""
    data = {}
    
    # Baseline performance
    baseline_path = RESULTS_DIR / "baseline_benchmark_results.csv"
    if baseline_path.exists():
        data['baseline'] = pd.read_csv(baseline_path)
    
    # MISATA performance
    misata_path = RESULTS_DIR / "misata_benchmark_results.csv"
    if misata_path.exists():
        data['misata'] = pd.read_csv(misata_path)
    
    # ML efficacy
    ml_path = RESULTS_DIR / "ml_efficacy_results.csv"
    if ml_path.exists():
        data['ml_efficacy'] = pd.read_csv(ml_path)
    
    # TSTR ratios
    tstr_path = RESULTS_DIR / "tstr_ratios.csv"
    if tstr_path.exists():
        data['tstr'] = pd.read_csv(tstr_path)
    
    # Chaos resilience
    chaos_path = RESULTS_DIR / "chaos_resilience_results.csv"
    if chaos_path.exists():
        data['chaos'] = pd.read_csv(chaos_path)
    
    # Statistical fidelity
    fidelity_path = RESULTS_DIR / "statistical_fidelity_results.csv"
    if fidelity_path.exists():
        data['fidelity'] = pd.read_csv(fidelity_path)
    
    return data


def fig1_performance_comparison(data):
    """Figure 1: Performance at 1M scale (log scale)."""
    if 'baseline' not in data:
        print("Skipping Fig 1: baseline data not found")
        return
    
    # Get 1M results
    df = data['baseline']
    df_1m = df[df['n_rows'] == 1_000_000].copy()
    
    # Add MISATA (use largest scale result)
    if 'misata' in data:
        misata = data['misata'].iloc[-1]
        df_1m = pd.concat([df_1m, pd.DataFrame([{
            'name': 'MISATA',
            'rows_per_second': misata['rows_per_second'],
            'peak_memory_mb': misata['peak_memory_mb']
        }])], ignore_index=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput
    colors = [COLORS.get(n, '#333') for n in df_1m['name']]
    bars1 = ax1.bar(df_1m['name'], df_1m['rows_per_second'], color=colors, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_ylabel('Throughput (rows/second)')
    ax1.set_title('(a) Generation Throughput at 1M Rows')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, df_1m['rows_per_second']):
        if val > 10000:
            label = f'{val/1000:.0f}K'
        else:
            label = f'{val:.0f}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 label, ha='center', va='bottom', fontsize=9)
    
    # Memory
    bars2 = ax2.bar(df_1m['name'], df_1m['peak_memory_mb'], color=colors, alpha=0.8)
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('(b) Memory Usage at 1M Rows')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_performance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_performance.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated fig1_performance.png")


def fig2_ml_efficacy(data):
    """Figure 2: ML Efficacy comparison (TSTR)."""
    if 'ml_efficacy' not in data:
        print("Skipping Fig 2: ml_efficacy data not found")
        return
    
    df = data['ml_efficacy'].copy()
    
    # Clean up names
    df['method'] = df['training_data'].str.replace(' (TSTR)', '').str.replace(' (TRTR)', '')
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metrics = ['roc_auc', 'precision', 'recall', 'f1']
    titles = ['ROC-AUC', 'Precision', 'Recall', 'F1 Score']
    
    for ax, metric, title in zip(axes, metrics, titles):
        colors = [COLORS.get(m.split()[0], '#333') for m in df['method']]
        bars = ax.bar(df['method'], df[metric], color=colors, alpha=0.8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1)
        
        # Highlight key values
        for bar, val in zip(bars, df[metric]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_ml_efficacy.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_ml_efficacy.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated fig2_ml_efficacy.png")


def fig3_chaos_resilience(data):
    """Figure 3: Chaos resilience curves."""
    if 'chaos' not in data:
        print("Skipping Fig 3: chaos data not found")
        return
    
    df = data['chaos']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scenarios = ['Null Injection', 'Distribution Shift', 'Correlation Break', 'Outlier Injection']
    
    for ax, scenario in zip(axes.flat, scenarios):
        scenario_df = df[df['scenario'] == scenario]
        
        if scenario == 'Correlation Break':
            # Bar chart for categorical
            ax.bar(scenario_df['severity_label'], scenario_df['auc_retention'] * 100, 
                   color='#3498db', alpha=0.8)
            ax.set_xlabel('Broken Feature')
        else:
            # Line chart for numerical
            ax.plot(scenario_df['severity'], scenario_df['auc_retention'] * 100, 
                   'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Severity')
        
        ax.set_ylabel('AUC Retention (%)')
        ax.set_title(scenario)
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_chaos_resilience.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_chaos_resilience.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated fig3_chaos_resilience.png")


def fig4_tstr_comparison(data):
    """Figure 4: TSTR ratio comparison."""
    if 'tstr' not in data:
        print("Skipping Fig 4: tstr data not found")
        return
    
    df = data['tstr']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = [COLORS.get(g, '#333') for g in df['generator']]
    bars = ax.barh(df['generator'], df['tstr_ratio'] * 100, color=colors, alpha=0.8)
    
    ax.set_xlabel('TSTR Ratio (%)')
    ax.set_title('Train-Synthetic-Test-Real Performance Ratio')
    ax.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='Real Data Baseline')
    ax.set_xlim(0, 110)
    
    # Add value labels
    for bar, val in zip(bars, df['tstr_ratio'] * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_tstr_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_tstr_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated fig4_tstr_comparison.png")


def main():
    """Generate all figures."""
    print("Loading experiment results...")
    data = load_data()
    print(f"Loaded: {list(data.keys())}")
    
    print("\nGenerating figures...")
    fig1_performance_comparison(data)
    fig2_ml_efficacy(data)
    fig3_chaos_resilience(data)
    fig4_tstr_comparison(data)
    
    print(f"\n✓ All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
