"""
Experiment 4: Unsupervised Clustering with KMeans and Cluster Quality Evaluation
=================================================================================
Applies KMeans clustering to the Iris dataset, determines optimal k via the
elbow method and silhouette scores, and evaluates cluster purity using the
adjusted Rand index against ground-truth labels.

Using sklearn.cluster.KMeans, sklearn.metrics — Context7 confirmed
(/websites/scikit-learn_dev)
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_iris_for_clustering() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset, returning features and true labels separately.

    Returns
    -------
    X : np.ndarray of shape (150, 4)  — features only (labels discarded for clustering)
    y_true : np.ndarray of shape (150,) — retained for evaluation only
    """
    iris = load_iris()
    return iris.data, iris.target


def scale_features(X: np.ndarray) -> np.ndarray:
    """
    Apply StandardScaler to feature matrix.

    Parameters
    ----------
    X : np.ndarray

    Returns
    -------
    X_scaled : np.ndarray
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def compute_kmeans_metrics(
    X_scaled: np.ndarray,
    k_range: range,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[list[float], list[float]]:
    """
    Fit KMeans for each k in k_range and record inertia and silhouette score.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of k values to evaluate.
    n_init : int
        Number of KMeans initializations.
    random_state : int
        Random seed.

    Returns
    -------
    inertias : list of float
    silhouette_scores : list of float
    """
    inertias: list[float] = []
    sil_scores: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # silhouette_score requires at least 2 clusters
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)

    return inertias, sil_scores


def select_optimal_k(
    k_range: range,
    silhouette_scores: list[float],
    inertias: list[float] | None = None,
) -> int:
    """
    Select the optimal k using the elbow method (second derivative of inertia)
    when inertias are provided, otherwise fall back to maximum silhouette score.

    The elbow method computes the second derivative of the inertia curve and
    selects the k at which the rate of decrease slows down the most (the "knee").
    This is more robust for datasets like Iris where silhouette peaks at k=2
    but the true cluster count is 3.

    Parameters
    ----------
    k_range : range
        Range of k values evaluated.
    silhouette_scores : list of float
        Silhouette score for each k.
    inertias : list of float or None
        Inertia values for each k. If provided, elbow method is used.

    Returns
    -------
    optimal_k : int
    """
    k_list = list(k_range)

    if inertias is not None and len(inertias) >= 3:
        # Elbow method: second derivative of inertia (largest curvature)
        # First differences (rate of decrease)
        first_diff = np.diff(inertias)
        # Second differences (rate of change of decrease)
        second_diff = np.diff(first_diff)
        # The elbow is at the k where second_diff is maximum (most curvature)
        # second_diff[i] corresponds to k_list[i+1] (offset by 1 from first_diff)
        elbow_idx = int(np.argmax(second_diff)) + 1  # +1 for second diff offset
        return k_list[elbow_idx]

    # Fallback: maximum silhouette score
    best_idx = int(np.argmax(silhouette_scores))
    return k_list[best_idx]


def fit_final_kmeans(
    X_scaled: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit the final KMeans model with the selected k.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters.
    n_init : int
        Number of initializations.
    random_state : int
        Random seed.

    Returns
    -------
    km : fitted KMeans
    labels : np.ndarray of cluster assignments
    """
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    return km, labels


def plot_elbow_and_silhouette(
    k_range: range,
    inertias: list[float],
    silhouette_scores: list[float],
    optimal_k: int,
    save_path: Path | None = None,
) -> None:
    """
    Plot inertia (elbow) and silhouette score side by side.

    Parameters
    ----------
    k_range : range
        Range of k values.
    inertias : list of float
        KMeans inertia for each k.
    silhouette_scores : list of float
        Silhouette score for each k.
    optimal_k : int
        Highlighted optimal k.
    save_path : Path or None
        If provided, save the figure.
    """
    k_list = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax1.plot(k_list, inertias, marker="o", linewidth=2, color="steelblue")
    ax1.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster SSE)")
    ax1.set_title("Elbow Method")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(k_list, silhouette_scores, marker="s", linewidth=2, color="darkorange")
    ax2.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal k={optimal_k}")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs. k")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("KMeans Cluster Selection — Iris Dataset", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_pca_clusters(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    optimal_k: int,
    save_path: Path | None = None,
) -> None:
    """
    Project data to 2D via PCA and plot colored by cluster assignment.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    cluster_labels : np.ndarray
        Predicted cluster assignments.
    true_labels : np.ndarray
        Ground-truth class labels (for subplot comparison).
    optimal_k : int
        Number of clusters used.
    save_path : Path or None
        If provided, save the figure.
    """
    pca = PCA(n_components=2, svd_solver="full")
    X_2d = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap_clusters = plt.colormaps.get_cmap("tab10").resampled(optimal_k)
    cmap_true = plt.colormaps.get_cmap("Set1").resampled(len(np.unique(true_labels)))

    # Cluster assignments
    for k in range(optimal_k):
        mask = cluster_labels == k
        ax1.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"Cluster {k}",
            alpha=0.7,
            color=cmap_clusters(k),
            edgecolors="k",
            linewidths=0.4,
        )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax1.set_title(f"KMeans Clusters (k={optimal_k})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # True labels
    for cls in np.unique(true_labels):
        mask = true_labels == cls
        ax2.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=f"True Class {cls}",
            alpha=0.7,
            color=cmap_true(cls),
            edgecolors="k",
            linewidths=0.4,
        )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax2.set_title("True Class Labels")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("PCA 2D Projection — KMeans vs. True Labels", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def run_experiment_4() -> dict[str, Any]:
    """
    Execute the full Experiment 4 workflow:
    1. Load Iris dataset; discard labels for clustering, retain for evaluation.
    2. Scale features with StandardScaler.
    3. Fit KMeans for k in range(2, 11) with n_init=10, random_state=42.
    4. Plot elbow and silhouette side by side.
    5. Select best k and fit final KMeans.
    6. Compute adjusted_rand_score.
    7. Project to 2D via PCA and plot cluster assignments.

    Returns
    -------
    dict containing all metrics and fitted objects.
    """
    print("\n" + "="*60)
    print("  EXP 4 — KMeans Clustering (Iris Dataset)")
    print("="*60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y_true = load_iris_for_clustering()
    print(f"\nIris dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Scale
    X_scaled = scale_features(X)

    # 3. Compute metrics for k in [2, 10]
    k_range = range(2, 11)
    print(f"\nFitting KMeans for k in {list(k_range)} ...")
    inertias, sil_scores = compute_kmeans_metrics(X_scaled, k_range)

    # Print per-k metrics
    print(f"\n{'k':>4} | {'Inertia':>12} | {'Silhouette':>12}")
    print("-" * 34)
    for k, iner, sil in zip(k_range, inertias, sil_scores):
        print(f"{k:>4} | {iner:>12.4f} | {sil:>12.4f}")

    # 5. Select optimal k using elbow method (second derivative of inertia)
    optimal_k = select_optimal_k(k_range, sil_scores, inertias=inertias)
    print(f"\nOptimal k selected (elbow method): {optimal_k}")

    # 4. Plot elbow + silhouette with confirmed optimal k
    plot_elbow_and_silhouette(
        k_range, inertias, sil_scores,
        optimal_k=optimal_k,
        save_path=RESULTS_DIR / "exp4_elbow_silhouette.png",
    )

    # 6. Fit final model
    km_final, cluster_labels = fit_final_kmeans(X_scaled, k=optimal_k)

    # 7. Adjusted Rand Index
    ari = adjusted_rand_score(y_true, cluster_labels)
    best_sil = silhouette_score(X_scaled, cluster_labels)

    print(f"\nFinal KMeans (k={optimal_k}):")
    print(f"  Adjusted Rand Index : {ari:.4f}")
    print(f"  Silhouette Score    : {best_sil:.4f}")
    print(f"  Inertia             : {km_final.inertia_:.4f}")

    # 8. PCA scatter plot
    plot_pca_clusters(
        X_scaled, cluster_labels, y_true,
        optimal_k=optimal_k,
        save_path=RESULTS_DIR / "exp4_pca_clusters.png",
    )

    # ── Summary Table ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(
        [
            {
                "k": k,
                "Inertia": round(iner, 4),
                "Silhouette": round(sil, 4),
            }
            for k, iner, sil in zip(k_range, inertias, sil_scores)
        ]
    )
    print("\n--- Metrics per k ---")
    print(metrics_df.to_string(index=False))

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "optimal_k": optimal_k,
        "adjusted_rand_index": ari,
        "final_silhouette": best_sil,
        "final_inertia": km_final.inertia_,
        "cluster_labels": cluster_labels,
        "true_labels": y_true,
        "metrics_df": metrics_df,
    }


if __name__ == "__main__":
    run_experiment_4()
