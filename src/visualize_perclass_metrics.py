#!/usr/bin/env python
"""
visualize_perclass_metrics.py
Visualize per-class precision, recall, F1 scores from FL results

Usage:
    python3 -m src.visualize_perclass_metrics
    OR
    python3 visualize_perclass_metrics.py  (if in project root)
"""

import sys
from pathlib import Path

# Handle imports whether run as module or script
try:
    from src.config import RESULTS_DIR, SUPERCLASSES, N_CLASSES
    from src.utils import ensure_dir
except ModuleNotFoundError:
    # Add parent directory to path if running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import RESULTS_DIR, SUPERCLASSES, N_CLASSES
    from src.utils import ensure_dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = RESULTS_DIR / "per_class_metrics"


def compute_metrics_from_cm(cm: np.ndarray) -> dict:
    """Compute precision, recall, F1 from confusion matrix."""
    metrics = {}
    prec_list, rec_list, f1_list = [], [], []
    
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        metrics[f"precision_{i}"] = prec
        metrics[f"recall_{i}"] = rec
        metrics[f"f1_{i}"] = f1
        
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
    
    metrics["precision_macro"] = float(np.mean(prec_list))
    metrics["recall_macro"] = float(np.mean(rec_list))
    metrics["f1_macro"] = float(np.mean(f1_list))
    
    return metrics


def load_cm_and_compute_metrics(cm_csv: Path) -> tuple:
    """Load confusion matrix CSV and compute metrics for last round."""
    if not cm_csv.exists():
        return None, None
    
    df = pd.read_csv(cm_csv)
    df.columns = df.columns.str.lower()
    
    if not {"round", "true", "pred", "count"}.issubset(df.columns):
        return None, None
    
    # Get last round
    last_round = int(df["round"].max())
    df_round = df[df["round"] == last_round]
    
    # Build confusion matrix
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for _, row in df_round.iterrows():
        i, j = int(row["true"]), int(row["pred"])
        cm[i, j] = int(row["count"])
    
    # Compute metrics
    metrics = compute_metrics_from_cm(cm)
    
    return metrics, last_round


def plot_metrics_table(metrics_dict: dict, title: str, save_path: Path):
    """Create a professional table visualization of metrics."""
    # Prepare data for table
    data = []
    for i, class_name in enumerate(SUPERCLASSES):
        data.append({
            "Class": class_name,
            "Precision": f"{metrics_dict[f'precision_{i}']:.4f}",
            "Recall": f"{metrics_dict[f'recall_{i}']:.4f}",
            "F1-Score": f"{metrics_dict[f'f1_{i}']:.4f}",
        })
    
    # Add macro row
    data.append({
        "Class": "Macro Avg",
        "Precision": f"{metrics_dict['precision_macro']:.4f}",
        "Recall": f"{metrics_dict['recall_macro']:.4f}",
        "F1-Score": f"{metrics_dict['f1_macro']:.4f}",
    })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Macro avg row styling
    for i in range(len(df.columns)):
        cell = table[(len(df), i)]
        cell.set_facecolor('#D9E1F2')
        cell.set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(df)):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    
    # Save
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_metrics_bars(metrics_dict: dict, title: str, save_path: Path):
    """Create grouped bar chart for precision, recall, F1."""
    # Prepare data
    classes = SUPERCLASSES
    precision = [metrics_dict[f"precision_{i}"] for i in range(N_CLASSES)]
    recall = [metrics_dict[f"recall_{i}"] for i in range(N_CLASSES)]
    f1 = [metrics_dict[f"f1_{i}"] for i in range(N_CLASSES)]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    ax.bar(x - width, precision, width, label='Precision', color='#4472C4')
    ax.bar(x, recall, width, label='Recall', color='#ED7D31')
    ax.bar(x + width, f1, width, label='F1-Score', color='#70AD47')
    
    # Styling
    ax.set_xlabel('Class', fontsize=12, weight='bold')
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_metrics_heatmap(all_metrics: dict, save_path: Path):
    """Create heatmap comparing metrics across configurations."""
    # Prepare data
    configs = list(all_metrics.keys())
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    
    data = []
    for config in configs:
        metrics = all_metrics[config]
        row = []
        for i in range(N_CLASSES):
            row.extend([
                metrics[f"precision_{i}"],
                metrics[f"recall_{i}"],
                metrics[f"f1_{i}"]
            ])
        data.append(row)
    
    # Create column labels
    col_labels = []
    for class_name in SUPERCLASSES:
        for metric in metrics_names:
            col_labels.append(f"{class_name}\n{metric}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(configs, fontsize=10)
    
    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title("Per-Class Metrics Comparison Across Configurations", 
                fontsize=14, weight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=90, fontsize=11)
    
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def save_metrics_to_csv(all_metrics: dict, save_path: Path):
    """Save all metrics to a comprehensive CSV."""
    rows = []
    
    for config, metrics in all_metrics.items():
        parts = config.split("_")
        model = parts[0]
        freeze = parts[1]
        phase = "_".join(parts[2:])  # Handle multi-word phases
        
        for i, class_name in enumerate(SUPERCLASSES):
            rows.append({
                "model": model,
                "freeze": freeze,
                "phase": phase,
                "class": class_name,
                "precision": f"{metrics[f'precision_{i}']:.4f}",
                "recall": f"{metrics[f'recall_{i}']:.4f}",
                "f1_score": f"{metrics[f'f1_{i}']:.4f}",
            })
        
        # Add macro row
        rows.append({
            "model": model,
            "freeze": freeze,
            "phase": phase,
            "class": "MACRO_AVG",
            "precision": f"{metrics['precision_macro']:.4f}",
            "recall": f"{metrics['recall_macro']:.4f}",
            "f1_score": f"{metrics['f1_macro']:.4f}",
        })
    
    df = pd.DataFrame(rows)
    ensure_dir(save_path.parent)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved CSV: {save_path}")


def main():
    """Generate all per-class metric visualizations."""
    ensure_dir(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("GENERATING PER-CLASS METRICS VISUALIZATIONS")
    print("="*70 + "\n")
    
    configs = [
        ("cnn", "frozen", "default"),
        ("cnn", "frozen", "tuned"),
        ("cnn", "non_frozen", "default"),
        ("cnn", "non_frozen", "tuned"),
        ("lstm", "frozen", "default"),
        ("lstm", "frozen", "tuned"),
        ("lstm", "non_frozen", "default"),
        ("lstm", "non_frozen", "tuned"),
    ]
    
    all_metrics = {}
    
    for model, freeze, phase in configs:
        cm_csv = RESULTS_DIR / f"{model}_{freeze}_run_cm_{phase}.csv"
        metrics, round_num = load_cm_and_compute_metrics(cm_csv)
        
        if metrics is None:
            print(f"✗ Skipping {model}_{freeze}_{phase} - no data")
            continue
        
        config_key = f"{model}_{freeze}_{phase}"
        all_metrics[config_key] = metrics
        
        # Generate table
        title = f"Per-Class Metrics: {model.upper()} - {freeze.title()} - {phase.title()} (Round {round_num})"
        plot_metrics_table(
            metrics,
            title,
            OUTPUT_DIR / f"table_{model}_{freeze}_{phase}.png"
        )
        
        # Generate bar chart
        plot_metrics_bars(
            metrics,
            title,
            OUTPUT_DIR / f"bars_{model}_{freeze}_{phase}.png"
        )
    
    # Generate comparison heatmap
    if all_metrics:
        plot_metrics_heatmap(
            all_metrics,
            OUTPUT_DIR / "comparison_heatmap.png"
        )
        
        # Save to CSV
        save_metrics_to_csv(
            all_metrics,
            OUTPUT_DIR / "all_metrics.csv"
        )
    
    print("\n" + "="*70)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()