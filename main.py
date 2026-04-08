"""
Main Runner — Scikit-Learn Experiment Suite
============================================
Executes all four experiments and saves metrics + plots to results/.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.exp1_pipeline import run_experiment_1
from src.exp2_hyperparameter_tuning import run_experiment_2
from src.exp3_pca_classification import run_experiment_3
from src.exp4_kmeans_clustering import run_experiment_4
from src.visualizations import (
    plot_confusion_matrix,
    plot_exp1_metrics,
    plot_exp2_comparison,
    plot_exp3_accuracy_comparison,
)

RESULTS_DIR = Path(__file__).parent / "results"
IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def save_results_markdown(
    exp1: dict,
    exp2: dict,
    exp3: dict,
    exp4: dict,
) -> None:
    """
    Write all experiment metrics to results/RESULTS.md.

    Parameters
    ----------
    exp1, exp2, exp3, exp4 : dict
        Results dictionaries from each experiment runner.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / "RESULTS.md"

    lines = [
        "# Experiment Results\n",
        "## Experiment 1 — Model Training and Evaluation Pipeline\n",
        "### Classification (Iris Dataset)\n",
        f"| Model | Accuracy |",
        f"|-------|----------|",
        f"| KNeighborsClassifier (n_neighbors=5) | {exp1['knn']['accuracy']:.4f} |",
        f"| SVC | {exp1['svc']['accuracy']:.4f} |",
        "",
        "### Regression (Synthetic Dataset)\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| MAE | {exp1['linear_regression']['mae']:.4f} |",
        f"| MSE | {exp1['linear_regression']['mse']:.4f} |",
        f"| R² | {exp1['linear_regression']['r2']:.4f} |",
        "",
        "---\n",
        "## Experiment 2 — Hyperparameter Tuning\n",
        "| Method | Best Params | Best CV Score | Test Accuracy | Runtime (s) |",
        "|--------|-------------|---------------|---------------|-------------|",
    ]

    for _, row in exp2["comparison_df"].iterrows():
        lines.append(
            f"| {row['method']} | {row['best_params']} | "
            f"{row['best_cv_score']:.4f} | {row['test_accuracy']:.4f} | "
            f"{row['runtime_seconds']:.4f} |"
        )

    lines += [
        "",
        "---\n",
        "## Experiment 3 — PCA + Classification\n",
        "| Dataset | Original Features | PCA Components (95%) | Acc (No PCA) | Acc (PCA) | Acc Drop |",
        "|---------|-------------------|----------------------|--------------|-----------|----------|",
    ]

    for _, row in exp3["summary_df"].iterrows():
        lines.append(
            f"| {row['Dataset']} | {row['Original Features']} | "
            f"{row['PCA Components (95%)']} | {row['Acc (No PCA)']} | "
            f"{row['Acc (PCA)']} | {row['Acc Drop']} |"
        )

    lines += [
        "",
        "---\n",
        "## Experiment 4 — KMeans Clustering\n",
        f"**Optimal k:** {exp4['optimal_k']}\n",
        f"**Adjusted Rand Index:** {exp4['adjusted_rand_index']:.4f}\n",
        f"**Final Silhouette Score:** {exp4['final_silhouette']:.4f}\n",
        f"**Final Inertia:** {exp4['final_inertia']:.4f}\n",
        "",
        "### Metrics per k\n",
        "| k | Inertia | Silhouette |",
        "|---|---------|------------|",
    ]

    for _, row in exp4["metrics_df"].iterrows():
        lines.append(f"| {int(row['k'])} | {row['Inertia']:.4f} | {row['Silhouette']:.4f} |")

    lines += [
        "",
        "---\n",
        "## Plots\n",
        "- `exp1_accuracy.png` — Classification accuracy bar chart",
        "- `exp1_cm_knn.png` — KNN confusion matrix",
        "- `exp1_cm_svc.png` — SVC confusion matrix",
        "- `exp2_comparison.png` — GridSearch vs RandomizedSearch comparison",
        "- `exp3_cumvar_iris.png` — PCA cumulative variance (Iris)",
        "- `exp3_cumvar_digits.png` — PCA cumulative variance (Digits)",
        "- `exp3_scatter_iris.png` — PCA 2D scatter (Iris)",
        "- `exp3_scatter_digits.png` — PCA 2D scatter (Digits)",
        "- `exp3_accuracy_comparison.png` — PCA vs No-PCA accuracy",
        "- `exp4_elbow_silhouette.png` — Elbow + silhouette plots",
        "- `exp4_pca_clusters.png` — PCA cluster scatter",
    ]

    md_path.write_text("\n".join(lines))
    print(f"\nResults saved to: {md_path}")


def main() -> None:
    """Run all experiments, generate plots, and save results."""
    np.random.seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*65)
    print("  SCIKIT-LEARN EXPERIMENT SUITE")
    print("#"*65)

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    exp1_results = run_experiment_1()

    plot_exp1_metrics(
        exp1_results,
        save_path=RESULTS_DIR / "exp1_accuracy.png",
    )
    plot_confusion_matrix(
        exp1_results["knn"]["confusion_matrix"],
        model_name="KNN",
        class_names=IRIS_CLASS_NAMES,
        save_path=RESULTS_DIR / "exp1_cm_knn.png",
    )
    plot_confusion_matrix(
        exp1_results["svc"]["confusion_matrix"],
        model_name="SVC",
        class_names=IRIS_CLASS_NAMES,
        save_path=RESULTS_DIR / "exp1_cm_svc.png",
    )

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    exp2_results = run_experiment_2()

    plot_exp2_comparison(
        exp2_results["comparison_df"],
        save_path=RESULTS_DIR / "exp2_comparison.png",
    )

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    exp3_results = run_experiment_3()

    plot_exp3_accuracy_comparison(
        exp3_results["summary_df"],
        save_path=RESULTS_DIR / "exp3_accuracy_comparison.png",
    )

    # ── Experiment 4 ──────────────────────────────────────────────────────────
    exp4_results = run_experiment_4()

    # ── Save Markdown ─────────────────────────────────────────────────────────
    save_results_markdown(exp1_results, exp2_results, exp3_results, exp4_results)

    print("\n" + "#"*65)
    print("  ALL EXPERIMENTS COMPLETE")
    print("#"*65)
    print(f"\nResults directory: {RESULTS_DIR}")
    print("Files saved:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
