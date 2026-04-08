"""
Visualization utilities for the scikit-learn ML pipeline experiment.

Generates:
  - Confusion matrix heatmaps for classification models
  - Actual vs. predicted scatter plots for regression models
  - Model comparison bar charts

All plots are saved to the results/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Any, List, Optional


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _ensure_results_dir() -> str:
    """Create results directory if it does not exist and return its path."""
    path = os.path.abspath(RESULTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrices(
    results_list: List[Dict[str, Any]],
    target_names: List[str],
    dataset_name: str = "Classification",
    filename: str = "confusion_matrices.png",
) -> str:
    """
    Plot side-by-side confusion matrix heatmaps for multiple classifiers.

    Parameters
    ----------
    results_list  : list of result dicts (each must have 'confusion_matrix' and 'model_name')
    target_names  : list of class label strings
    dataset_name  : title prefix for the figure
    filename      : output filename (saved in results/)

    Returns
    -------
    Absolute path to the saved figure.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.4f}", fontsize=13)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    fig.suptitle(f"{dataset_name} — Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_dir = _ensure_results_dir()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")
    return out_path


def plot_regression_predictions(
    results_list: List[Dict[str, Any]],
    y_test: np.ndarray,
    dataset_name: str = "Regression",
    filename: str = "regression_predictions.png",
) -> str:
    """
    Plot actual vs. predicted scatter plots for multiple regressors.

    Parameters
    ----------
    results_list : list of result dicts (each must have 'y_pred', 'model_name', 'r2', 'mse')
    y_test       : np.ndarray — true target values
    dataset_name : title prefix for the figure
    filename     : output filename (saved in results/)

    Returns
    -------
    Absolute path to the saved figure.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        y_pred = res["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3, s=20)

        # Perfect prediction line
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")

        ax.set_title(
            f"{res['model_name']}\nR²={res['r2']:.4f}  MSE={res['mse']:.4f}",
            fontsize=12,
        )
        ax.set_xlabel("Actual Values", fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle(f"{dataset_name} — Actual vs. Predicted", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_dir = _ensure_results_dir()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")
    return out_path


def plot_model_comparison(
    clf_results: List[Dict[str, Any]],
    reg_results: List[Dict[str, Any]],
    filename: str = "model_comparison.png",
) -> str:
    """
    Plot a side-by-side bar chart comparing classification accuracy and
    regression R² scores across models.

    Parameters
    ----------
    clf_results : list of classification result dicts
    reg_results : list of regression result dicts
    filename    : output filename (saved in results/)

    Returns
    -------
    Absolute path to the saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Classification accuracy
    clf_names = [r["model_name"] for r in clf_results]
    clf_accs = [r["accuracy"] for r in clf_results]
    bars1 = ax1.bar(clf_names, clf_accs, color=["steelblue", "darkorange"], edgecolor="black")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Classification — Accuracy", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_xlabel("Model", fontsize=11)
    for bar, val in zip(bars1, clf_accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Regression R²
    reg_names = [r["model_name"] for r in reg_results]
    reg_r2 = [r["r2"] for r in reg_results]
    bars2 = ax2.bar(reg_names, reg_r2, color=["mediumseagreen", "tomato"], edgecolor="black")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Regression — R² Score", fontsize=13, fontweight="bold")
    ax2.set_ylabel("R² Score", fontsize=11)
    ax2.set_xlabel("Model", fontsize=11)
    for bar, val in zip(bars2, reg_r2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Model Comparison Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_dir = _ensure_results_dir()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")
    return out_path


def plot_feature_distributions(
    X: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "Dataset",
    filename: str = "feature_distributions.png",
    max_features: int = 8,
) -> str:
    """
    Plot histograms of feature distributions (up to max_features).

    Parameters
    ----------
    X            : np.ndarray — feature matrix (unscaled)
    feature_names: list[str]  — feature names
    dataset_name : str        — title prefix
    filename     : str        — output filename
    max_features : int        — maximum number of features to plot

    Returns
    -------
    Absolute path to the saved figure.
    """
    n_features = min(X.shape[1], max_features)
    cols = 4
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(feature_names[i], fontsize=9)
        axes[i].set_xlabel("Value", fontsize=8)
        axes[i].set_ylabel("Count", fontsize=8)

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} — Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = _ensure_results_dir()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {out_path}")
    return out_path
