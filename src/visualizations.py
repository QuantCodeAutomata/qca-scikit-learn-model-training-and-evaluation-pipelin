"""
Visualizations Module
=====================
Generates and saves all plots for the four experiments.
Uses matplotlib and seaborn for all visualizations.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Plot a confusion matrix as a heatmap using seaborn.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array.
    model_name : str
        Title label.
    class_names : list of str or None
        Tick labels for axes.
    save_path : Path or None
        If provided, save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_exp1_metrics(
    results: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """
    Bar chart comparing classification accuracy for KNN and SVC (Exp 1).

    Parameters
    ----------
    results : dict
        Output from run_experiment_1().
    save_path : Path or None
    """
    models = ["KNN", "SVC"]
    accuracies = [
        results["knn"]["accuracy"],
        results["svc"]["accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, accuracies, color=["steelblue", "darkorange"], width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 1 — Classification Accuracy (Iris)")
    ax.axhline(0.90, color="red", linestyle="--", label="90% threshold")
    ax.legend()
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_exp2_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Grouped bar chart comparing GridSearchCV vs RandomizedSearchCV (Exp 2).

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from build_comparison_table().
    save_path : Path or None
    """
    methods = comparison_df["method"].tolist()
    cv_scores = comparison_df["best_cv_score"].tolist()
    test_accs = comparison_df["test_accuracy"].tolist()
    runtimes = comparison_df["runtime_seconds"].tolist()

    x = np.arange(len(methods))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax1.bar(x - width / 2, cv_scores, width, label="CV Score", color="steelblue")
    ax1.bar(x + width / 2, test_accs, width, label="Test Accuracy", color="darkorange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Score")
    ax1.set_title("Exp 2 — CV Score vs Test Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Runtime comparison
    ax2.bar(methods, runtimes, color=["steelblue", "darkorange"], width=0.4)
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Exp 2 — Search Runtime")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, (method, rt) in enumerate(zip(methods, runtimes)):
        ax2.text(i, rt + 0.001, f"{rt:.3f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_exp3_accuracy_comparison(
    summary_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Grouped bar chart comparing accuracy with/without PCA for Iris and Digits (Exp 3).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from run_experiment_3().
    save_path : Path or None
    """
    datasets = summary_df["Dataset"].tolist()
    acc_no_pca = summary_df["Acc (No PCA)"].astype(float).tolist()
    acc_pca = summary_df["Acc (PCA)"].astype(float).tolist()

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, acc_no_pca, width, label="No PCA", color="steelblue")
    ax.bar(x + width / 2, acc_pca, width, label="With PCA (95% var)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 3 — Classification Accuracy: PCA vs No PCA")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
