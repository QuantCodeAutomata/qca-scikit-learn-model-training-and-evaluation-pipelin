"""
Experiment 3: Dimensionality Reduction with PCA Followed by Classification
===========================================================================
Evaluates the effect of PCA-based dimensionality reduction on classification
performance and training time for Iris (4 features) and Digits (64 features).

Using sklearn.decomposition.PCA — Context7 confirmed (/websites/scikit-learn_dev)
"""

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load a named sklearn dataset.

    Parameters
    ----------
    name : str
        One of 'iris' or 'digits'.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    dataset_name : str
    """
    if name == "iris":
        data = load_iris()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'iris' or 'digits'.")
    return data.data, data.target, name


def build_pipeline_with_pca(
    n_components: float | int,
    n_neighbors: int = 5,
    scale: bool = False,
) -> Pipeline:
    """
    Build a Pipeline: [StandardScaler →] PCA → KNeighborsClassifier.

    Parameters
    ----------
    n_components : float or int
        If float in (0, 1), retain that fraction of variance.
        If int, use that many components.
    n_neighbors : int
        KNN neighbours.
    scale : bool
        Whether to prepend a StandardScaler step.

    Returns
    -------
    Pipeline
    """
    steps: list = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    # Using sklearn PCA with svd_solver='full' — Context7 confirmed
    steps.append(("pca", PCA(n_components=n_components, svd_solver="full")))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=n_neighbors)))
    return Pipeline(steps=steps)


def build_pipeline_no_pca(
    n_neighbors: int = 5,
    scale: bool = False,
) -> Pipeline:
    """
    Build a Pipeline: [StandardScaler →] KNeighborsClassifier (no PCA).

    Parameters
    ----------
    n_neighbors : int
        KNN neighbours.
    scale : bool
        Whether to prepend a StandardScaler step.

    Returns
    -------
    Pipeline
    """
    steps: list = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=n_neighbors)))
    return Pipeline(steps=steps)


def fit_and_evaluate(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """
    Fit a pipeline and return test accuracy and training time.

    Parameters
    ----------
    pipeline : Pipeline
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray

    Returns
    -------
    accuracy : float
    train_time : float  (seconds)
    """
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, train_time


def get_pca_n_components(pipeline: Pipeline) -> int:
    """
    Extract the number of PCA components selected from a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Must contain a 'pca' step.

    Returns
    -------
    int
    """
    return pipeline.named_steps["pca"].n_components_


def plot_cumulative_variance(
    X_train: np.ndarray,
    dataset_name: str,
    scale: bool = False,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    Fit PCA on training data and plot cumulative explained variance ratio.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled if scale=False).
    dataset_name : str
        Used in plot title.
    scale : bool
        Whether to scale before PCA.
    save_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    cumulative_variance : np.ndarray
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    pca_full = PCA(svd_solver="full")
    pca_full.fit(X_train)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o", linewidth=2, color="steelblue")
    ax.axhline(0.95, color="red", linestyle="--", label="95% variance threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title(f"PCA Cumulative Variance — {dataset_name.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)

    return cumvar


def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    scale: bool = False,
    save_path: Path | None = None,
) -> None:
    """
    Project data to first two PCA components and scatter-plot by class label.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Class labels.
    dataset_name : str
        Used in plot title.
    scale : bool
        Whether to scale before PCA.
    save_path : Path or None
        If provided, save the figure to this path.
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca2 = PCA(n_components=2, svd_solver="full")
    X_2d = pca2.fit_transform(X)

    classes = np.unique(y)
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(classes))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in classes:
        mask = y == cls
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"Class {cls}",
            alpha=0.7,
            color=cmap(cls),
            edgecolors="k",
            linewidths=0.4,
        )
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"PCA 2D Scatter — {dataset_name.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def run_single_dataset(
    dataset_name: str,
    variance_threshold: float = 0.95,
    n_neighbors: int = 5,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """
    Run the PCA experiment for a single dataset.

    Steps:
    1. Load and split 70/30.
    2. Fit PCA retaining `variance_threshold` of variance.
    3. Train KNN on original and PCA-reduced features.
    4. Report accuracy, training time, and component count.
    5. Plot cumulative variance and 2D scatter.

    Parameters
    ----------
    dataset_name : str
        'iris' or 'digits'.
    variance_threshold : float
        Fraction of variance to retain (default 0.95).
    n_neighbors : int
        KNN neighbours.
    results_dir : Path
        Directory to save plots.

    Returns
    -------
    dict with experiment results.
    """
    scale = dataset_name == "digits"  # scale digits before PCA per methodology

    X, y, name = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # ── No-PCA baseline ──────────────────────────────────────────────────────
    pipe_no_pca = build_pipeline_no_pca(n_neighbors=n_neighbors, scale=scale)
    acc_no_pca, time_no_pca = fit_and_evaluate(
        pipe_no_pca, X_train, X_test, y_train, y_test
    )

    # ── PCA pipeline ─────────────────────────────────────────────────────────
    pipe_pca = build_pipeline_with_pca(
        n_components=variance_threshold,
        n_neighbors=n_neighbors,
        scale=scale,
    )
    acc_pca, time_pca = fit_and_evaluate(
        pipe_pca, X_train, X_test, y_train, y_test
    )
    n_components_selected = get_pca_n_components(pipe_pca)

    print(f"\n{'='*55}")
    print(f"  Dataset: {name.upper()} | Original features: {X.shape[1]}")
    print(f"{'='*55}")
    print(f"  PCA components selected (95% var): {n_components_selected}")
    print(f"  Accuracy (no PCA) : {acc_no_pca:.4f}  | Time: {time_no_pca:.4f}s")
    print(f"  Accuracy (PCA)    : {acc_pca:.4f}  | Time: {time_pca:.4f}s")
    print(f"  Accuracy drop     : {acc_no_pca - acc_pca:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    cumvar = plot_cumulative_variance(
        X_train, name, scale=scale,
        save_path=results_dir / f"exp3_cumvar_{name}.png",
    )
    plot_pca_scatter(
        X, y, name, scale=scale,
        save_path=results_dir / f"exp3_scatter_{name}.png",
    )

    return {
        "dataset": name,
        "n_original_features": X.shape[1],
        "n_pca_components": n_components_selected,
        "acc_no_pca": acc_no_pca,
        "acc_pca": acc_pca,
        "time_no_pca": time_no_pca,
        "time_pca": time_pca,
        "cumulative_variance": cumvar,
    }


def run_experiment_3() -> dict[str, Any]:
    """
    Execute the full Experiment 3 workflow for Iris and Digits datasets.

    Returns
    -------
    dict containing results for both datasets and a summary DataFrame.
    """
    print("\n" + "="*60)
    print("  EXP 3 — PCA + Classification")
    print("="*60)

    iris_results = run_single_dataset("iris")
    digits_results = run_single_dataset("digits")

    # ── Summary Table ─────────────────────────────────────────────────────────
    rows = []
    for r in [iris_results, digits_results]:
        rows.append({
            "Dataset": r["dataset"],
            "Original Features": r["n_original_features"],
            "PCA Components (95%)": r["n_pca_components"],
            "Acc (No PCA)": f"{r['acc_no_pca']:.4f}",
            "Acc (PCA)": f"{r['acc_pca']:.4f}",
            "Acc Drop": f"{r['acc_no_pca'] - r['acc_pca']:.4f}",
            "Train Time No PCA (s)": f"{r['time_no_pca']:.4f}",
            "Train Time PCA (s)": f"{r['time_pca']:.4f}",
        })

    summary_df = pd.DataFrame(rows)
    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))

    return {
        "iris": iris_results,
        "digits": digits_results,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    run_experiment_3()
