#!/usr/bin/env python
"""
generate_evaluation_tables.py
Generate publication-ready comparison tables for evaluation section

Usage:
    python3 -m src.generate_evaluation_tables
"""

import sys
from pathlib import Path

try:
    from src.config import RESULTS_DIR, SUPERCLASSES, N_CLASSES
    from src.utils import ensure_dir
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import RESULTS_DIR, SUPERCLASSES, N_CLASSES
    from src.utils import ensure_dir

import numpy as np
import pandas as pd


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
    
    # Compute accuracy from confusion matrix
    accuracy = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0
    
    metrics["accuracy"] = accuracy
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
    
    last_round = int(df["round"].max())
    df_round = df[df["round"] == last_round]
    
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for _, row in df_round.iterrows():
        i, j = int(row["true"]), int(row["pred"])
        cm[i, j] = int(row["count"])
    
    metrics = compute_metrics_from_cm(cm)
    
    return metrics, last_round


def generate_summary_table():
    """Generate overall performance summary table."""
    print("\n" + "="*70)
    print("GENERATING EVALUATION TABLES")
    print("="*70 + "\n")

    MODELS = ["lstm", "gru", "cnn_lstm", "mlp"]
    FREEZES = ["frozen", "non_frozen"]
    PHASES = ["default", "tuned"]
    configs = [(m, f, p) for m in MODELS for f in FREEZES for p in PHASES]
    
    rows = []
    
    for model, freeze, phase in configs:
        cm_csv = RESULTS_DIR / f"{model}_{freeze}_run_cm_{phase}.csv"
        metrics, round_num = load_cm_and_compute_metrics(cm_csv)
        
        if metrics is None:
            continue
        
        rows.append({
            "Model": model.upper(),
            "Freeze": "Yes" if freeze == "frozen" else "No",
            "Tuning": phase.title(),
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision_macro']:.4f}",
            "Recall": f"{metrics['recall_macro']:.4f}",
            "F1-Score": f"{metrics['f1_macro']:.4f}",
            "Round": round_num,
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_dir = RESULTS_DIR / "evaluation_tables"
    ensure_dir(output_dir)
    
    csv_path = output_dir / "summary_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    return df


def generate_per_class_table():
    """Generate detailed per-class comparison table."""
    MODELS = ["lstm", "gru", "cnn_lstm", "mlp"]
    FREEZES = ["frozen", "non_frozen"]
    PHASES = ["default", "tuned"]
    configs = [(m, f, p) for m in MODELS for f in FREEZES for p in PHASES]
    
    all_rows = []
    
    for model, freeze, phase in configs:
        cm_csv = RESULTS_DIR / f"{model}_{freeze}_run_cm_{phase}.csv"
        metrics, round_num = load_cm_and_compute_metrics(cm_csv)
        
        if metrics is None:
            continue
        
        config_name = f"{model.upper()}-{freeze}-{phase}"
        
        # Per-class rows
        for i, class_name in enumerate(SUPERCLASSES):
            all_rows.append({
                "Configuration": config_name,
                "Class": class_name,
                "Precision": f"{metrics[f'precision_{i}']:.4f}",
                "Recall": f"{metrics[f'recall_{i}']:.4f}",
                "F1-Score": f"{metrics[f'f1_{i}']:.4f}",
            })
    
    df = pd.DataFrame(all_rows)
    
    # Save to CSV
    output_dir = RESULTS_DIR / "evaluation_tables"
    csv_path = output_dir / "per_class_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    return df


def generate_best_config_table():
    """Generate table showing best configuration per metric."""
    MODELS = ["lstm", "gru", "cnn_lstm", "mlp"]
    FREEZES = ["frozen", "non_frozen"]
    PHASES = ["default", "tuned"]
    configs = [(m, f, p) for m in MODELS for f in FREEZES for p in PHASES]
    
    results = {}
    
    for model, freeze, phase in configs:
        cm_csv = RESULTS_DIR / f"{model}_{freeze}_run_cm_{phase}.csv"
        metrics, round_num = load_cm_and_compute_metrics(cm_csv)
        
        if metrics is None:
            continue
        
        config_name = f"{model.upper()}-{freeze}-{phase}"
        results[config_name] = metrics
    
    # Find best for each metric
    best_rows = []
    
    metrics_to_check = [
        ("Accuracy", "accuracy"),
        ("Precision (Macro)", "precision_macro"),
        ("Recall (Macro)", "recall_macro"),
        ("F1-Score (Macro)", "f1_macro"),
    ]
    
    for metric_name, metric_key in metrics_to_check:
        best_config = max(results.items(), key=lambda x: x[1][metric_key])
        best_rows.append({
            "Metric": metric_name,
            "Best Configuration": best_config[0],
            "Score": f"{best_config[1][metric_key]:.4f}",
        })
    
    df = pd.DataFrame(best_rows)
    
    # Save to CSV
    output_dir = RESULTS_DIR / "evaluation_tables"
    csv_path = output_dir / "best_configurations.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS PER METRIC")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    return df


def generate_latex_table(df: pd.DataFrame, caption: str, label: str, output_file: Path):
    """Generate LaTeX table code for papers."""
    latex_code = df.to_latex(index=False, float_format="%.4f", caption=caption, label=label)
    
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ Saved LaTeX: {output_file}")


def main():
    """Generate all evaluation tables."""
    
    # Generate summary comparison
    summary_df = generate_summary_table()
    
    # Generate per-class detailed table
    print("Generating per-class detailed comparison...")
    per_class_df = generate_per_class_table()
    
    # Generate best configurations table
    best_df = generate_best_config_table()
    
    # Generate LaTeX tables for paper
    output_dir = RESULTS_DIR / "evaluation_tables"
    print("\nGenerating LaTeX tables for paper...")
    
    generate_latex_table(
        summary_df,
        caption="Overall Performance Comparison of FL Configurations",
        label="tab:fl_summary",
        output_file=output_dir / "summary_latex.tex"
    )
    
    generate_latex_table(
        best_df,
        caption="Best Performing Configurations per Metric",
        label="tab:fl_best",
        output_file=output_dir / "best_latex.tex"
    )
    
    print("\n" + "="*70)
    print("✓ All evaluation tables generated!")
    print(f"✓ Check: {output_dir}")
    print("="*70)
    print("\nFiles created:")
    print("  - summary_comparison.csv      (for Excel/analysis)")
    print("  - per_class_detailed.csv      (per-class breakdown)")
    print("  - best_configurations.csv     (best performers)")
    print("  - summary_latex.tex           (for paper)")
    print("  - best_latex.tex              (for paper)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
