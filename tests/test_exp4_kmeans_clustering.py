"""
Tests for Experiment 4: Unsupervised Clustering with KMeans and Cluster Quality Evaluation
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from src.exp4_kmeans_clustering import (
    compute_kmeans_metrics,
    fit_final_kmeans,
    load_iris_for_clustering,
    run_experiment_4,
    scale_features,
    select_optimal_k,
)


# ── Data Loading Tests ────────────────────────────────────────────────────────

def test_load_iris_for_clustering_shape():
    """Iris dataset should have 150 samples and 4 features."""
    X, y_true = load_iris_for_clustering()
    assert X.shape == (150, 4)
    assert y_true.shape == (150,)


def test_load_iris_for_clustering_true_classes():
    """Iris true labels should have exactly 3 classes."""
    _, y_true = load_iris_for_clustering()
    assert len(np.unique(y_true)) == 3


# ── Scaling Tests ─────────────────────────────────────────────────────────────

def test_scale_features_shape():
    """Scaled features should preserve input shape."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    assert X_scaled.shape == X.shape


def test_scale_features_zero_mean():
    """StandardScaler should produce approximately zero column means."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    col_means = np.abs(X_scaled.mean(axis=0))
    assert np.all(col_means < 1e-10), "Scaled features should have ~zero mean"


def test_scale_features_unit_variance():
    """StandardScaler should produce approximately unit column variance."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    col_stds = X_scaled.std(axis=0)
    np.testing.assert_allclose(col_stds, 1.0, atol=1e-10)


# ── KMeans Metrics Tests ──────────────────────────────────────────────────────

def test_compute_kmeans_metrics_lengths():
    """Inertias and silhouette scores should have same length as k_range."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k_range = range(2, 6)
    inertias, sil_scores = compute_kmeans_metrics(X_scaled, k_range)
    assert len(inertias) == len(k_range)
    assert len(sil_scores) == len(k_range)


def test_compute_kmeans_metrics_inertia_decreasing():
    """Inertia should be non-increasing as k increases."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k_range = range(2, 8)
    inertias, _ = compute_kmeans_metrics(X_scaled, k_range)
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1], (
            f"Inertia should decrease: inertia[{i}]={inertias[i]:.4f} < "
            f"inertia[{i+1}]={inertias[i+1]:.4f}"
        )


def test_compute_kmeans_metrics_silhouette_range():
    """Silhouette scores should be in [-1, 1]."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k_range = range(2, 6)
    _, sil_scores = compute_kmeans_metrics(X_scaled, k_range)
    for sil in sil_scores:
        assert -1.0 <= sil <= 1.0, f"Silhouette score {sil:.4f} out of range [-1, 1]"


def test_compute_kmeans_metrics_inertia_positive():
    """All inertia values should be positive."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k_range = range(2, 6)
    inertias, _ = compute_kmeans_metrics(X_scaled, k_range)
    assert all(iner > 0 for iner in inertias)


# ── Optimal k Selection Tests ─────────────────────────────────────────────────

def test_select_optimal_k_returns_valid_k():
    """Optimal k should be within the evaluated k_range."""
    k_range = range(2, 11)
    sil_scores = [0.3, 0.5, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
    # Inertias decreasing with elbow at k=4 (index 2)
    inertias = [300.0, 200.0, 150.0, 148.0, 147.0, 146.5, 146.0, 145.8, 145.6]
    optimal_k = select_optimal_k(k_range, sil_scores, inertias=inertias)
    assert optimal_k in list(k_range)


def test_select_optimal_k_picks_elbow():
    """Optimal k should correspond to the elbow (max second derivative of inertia)."""
    k_range = range(2, 7)
    sil_scores = [0.58, 0.46, 0.39, 0.35, 0.32]
    # Inertias with clear elbow at k=3 (index 1): large drop from k=2→3, small after
    inertias = [222.0, 140.0, 114.0, 91.0, 82.0]
    optimal_k = select_optimal_k(k_range, sil_scores, inertias=inertias)
    # second_diff = diff([140-222, 114-140, 91-114, 82-91]) = diff([-82, -26, -23, -9])
    # = [56, 3, 14] → max at index 0 → k_list[0+1] = k=3
    assert optimal_k == 3


def test_select_optimal_k_fallback_silhouette():
    """Without inertias, optimal k should use max silhouette score."""
    k_range = range(2, 7)
    sil_scores = [0.3, 0.7, 0.5, 0.4, 0.35]
    optimal_k = select_optimal_k(k_range, sil_scores, inertias=None)
    # Max silhouette is at index 1 → k=3
    assert optimal_k == 3


def test_select_optimal_k_iris():
    """Optimal k for Iris should be 3 (matching true class count) via elbow method."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k_range = range(2, 11)
    inertias, sil_scores = compute_kmeans_metrics(X_scaled, k_range)
    optimal_k = select_optimal_k(k_range, sil_scores, inertias=inertias)
    assert optimal_k == 3, f"Expected optimal k=3 for Iris, got {optimal_k}"


# ── Final KMeans Tests ────────────────────────────────────────────────────────

def test_fit_final_kmeans_returns_kmeans():
    """fit_final_kmeans should return a fitted KMeans object."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    km, labels = fit_final_kmeans(X_scaled, k=3)
    assert isinstance(km, KMeans)


def test_fit_final_kmeans_label_count():
    """Cluster labels should have same length as input samples."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    _, labels = fit_final_kmeans(X_scaled, k=3)
    assert len(labels) == len(X_scaled)


def test_fit_final_kmeans_unique_labels():
    """Number of unique cluster labels should equal k."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    k = 3
    _, labels = fit_final_kmeans(X_scaled, k=k)
    assert len(np.unique(labels)) == k


def test_fit_final_kmeans_reproducibility():
    """Same random_state should produce identical cluster assignments."""
    X, _ = load_iris_for_clustering()
    X_scaled = scale_features(X)
    _, labels1 = fit_final_kmeans(X_scaled, k=3, random_state=42)
    _, labels2 = fit_final_kmeans(X_scaled, k=3, random_state=42)
    np.testing.assert_array_equal(labels1, labels2)


# ── Adjusted Rand Index Tests ─────────────────────────────────────────────────

def test_adjusted_rand_index_above_threshold():
    """Adjusted Rand Index for k=3 on Iris should exceed 0.6.
    
    Note: KMeans with k=3 on Iris achieves ARI ~0.62 because versicolor and
    virginica overlap in feature space. The 0.6 threshold reflects this reality.
    """
    from sklearn.metrics import adjusted_rand_score
    X, y_true = load_iris_for_clustering()
    X_scaled = scale_features(X)
    _, labels = fit_final_kmeans(X_scaled, k=3)
    ari = adjusted_rand_score(y_true, labels)
    assert ari > 0.6, f"ARI {ari:.4f} below 0.6 threshold"


def test_adjusted_rand_index_range():
    """Adjusted Rand Index should be in [-1, 1]."""
    from sklearn.metrics import adjusted_rand_score
    X, y_true = load_iris_for_clustering()
    X_scaled = scale_features(X)
    _, labels = fit_final_kmeans(X_scaled, k=3)
    ari = adjusted_rand_score(y_true, labels)
    assert -1.0 <= ari <= 1.0


# ── Full Experiment Integration Test ──────────────────────────────────────────

def test_run_experiment_4_keys():
    """run_experiment_4 should return all expected keys."""
    results = run_experiment_4()
    expected_keys = {
        "k_range", "inertias", "silhouette_scores", "optimal_k",
        "adjusted_rand_index", "final_silhouette", "final_inertia",
        "cluster_labels", "true_labels", "metrics_df",
    }
    assert expected_keys.issubset(set(results.keys()))


def test_run_experiment_4_optimal_k():
    """Optimal k should be 3 for Iris dataset."""
    results = run_experiment_4()
    assert results["optimal_k"] == 3


def test_run_experiment_4_ari_above_threshold():
    """Adjusted Rand Index should exceed 0.6 (k=3 on Iris achieves ~0.62)."""
    results = run_experiment_4()
    assert results["adjusted_rand_index"] > 0.6


def test_run_experiment_4_metrics_df_shape():
    """Metrics DataFrame should have 9 rows (k from 2 to 10)."""
    results = run_experiment_4()
    assert len(results["metrics_df"]) == 9


def test_run_experiment_4_k_range():
    """k_range should cover 2 to 10 inclusive."""
    results = run_experiment_4()
    assert results["k_range"] == list(range(2, 11))


def test_run_experiment_4_silhouette_range():
    """All silhouette scores should be in [-1, 1]."""
    results = run_experiment_4()
    for sil in results["silhouette_scores"]:
        assert -1.0 <= sil <= 1.0


def test_run_experiment_4_final_silhouette_positive():
    """Final silhouette score for optimal k should be positive (well-separated clusters)."""
    results = run_experiment_4()
    assert results["final_silhouette"] > 0.0
