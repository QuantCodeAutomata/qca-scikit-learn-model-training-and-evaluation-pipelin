"""
Tests for Experiment 3: Dimensionality Reduction with PCA Followed by Classification
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.exp3_pca_classification import (
    build_pipeline_no_pca,
    build_pipeline_with_pca,
    fit_and_evaluate,
    get_pca_n_components,
    load_dataset,
    run_experiment_3,
    run_single_dataset,
)


# ── Dataset Loading Tests ─────────────────────────────────────────────────────

def test_load_iris_shape():
    """Iris dataset should have 150 samples and 4 features."""
    X, y, name = load_dataset("iris")
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert name == "iris"


def test_load_digits_shape():
    """Digits dataset should have 1797 samples and 64 features."""
    X, y, name = load_dataset("digits")
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)
    assert name == "digits"


def test_load_dataset_invalid_name():
    """Loading an unknown dataset should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown_dataset")


# ── Pipeline Construction Tests ───────────────────────────────────────────────

def test_build_pipeline_with_pca_type():
    """PCA pipeline should be a sklearn Pipeline."""
    pipe = build_pipeline_with_pca(n_components=0.95)
    assert isinstance(pipe, Pipeline)


def test_build_pipeline_with_pca_steps_no_scale():
    """PCA pipeline without scaling should have pca and knn steps."""
    pipe = build_pipeline_with_pca(n_components=0.95, scale=False)
    step_names = [name for name, _ in pipe.steps]
    assert "pca" in step_names
    assert "knn" in step_names
    assert "scaler" not in step_names


def test_build_pipeline_with_pca_steps_with_scale():
    """PCA pipeline with scaling should have scaler, pca, and knn steps."""
    pipe = build_pipeline_with_pca(n_components=0.95, scale=True)
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "pca" in step_names
    assert "knn" in step_names


def test_build_pipeline_no_pca_type():
    """No-PCA pipeline should be a sklearn Pipeline."""
    pipe = build_pipeline_no_pca()
    assert isinstance(pipe, Pipeline)


def test_build_pipeline_no_pca_steps():
    """No-PCA pipeline should have knn step but no pca step."""
    pipe = build_pipeline_no_pca()
    step_names = [name for name, _ in pipe.steps]
    assert "knn" in step_names
    assert "pca" not in step_names


# ── Fit and Evaluate Tests ────────────────────────────────────────────────────

def test_fit_and_evaluate_accuracy_range():
    """Accuracy should be between 0 and 1."""
    X, y, _ = load_dataset("iris")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    pipe = build_pipeline_no_pca()
    acc, t = fit_and_evaluate(pipe, X_train, X_test, y_train, y_test)
    assert 0.0 <= acc <= 1.0


def test_fit_and_evaluate_time_positive():
    """Training time should be positive."""
    X, y, _ = load_dataset("iris")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    pipe = build_pipeline_no_pca()
    _, t = fit_and_evaluate(pipe, X_train, X_test, y_train, y_test)
    assert t > 0.0


# ── PCA Component Count Tests ─────────────────────────────────────────────────

def test_get_pca_n_components_iris():
    """PCA on Iris with 95% variance should select fewer than 4 components."""
    X, y, _ = load_dataset("iris")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    pipe = build_pipeline_with_pca(n_components=0.95, scale=False)
    pipe.fit(X_train, y_train)
    n_comp = get_pca_n_components(pipe)
    assert n_comp <= 4
    assert n_comp >= 1


def test_get_pca_n_components_digits():
    """PCA on Digits with 95% variance should select far fewer than 64 components."""
    X, y, _ = load_dataset("digits")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    pipe = build_pipeline_with_pca(n_components=0.95, scale=True)
    pipe.fit(X_train, y_train)
    n_comp = get_pca_n_components(pipe)
    assert n_comp < 64
    assert n_comp >= 1


# ── Single Dataset Experiment Tests ──────────────────────────────────────────

def test_run_single_dataset_iris_keys():
    """run_single_dataset for iris should return all expected keys."""
    result = run_single_dataset("iris")
    expected_keys = {
        "dataset", "n_original_features", "n_pca_components",
        "acc_no_pca", "acc_pca", "time_no_pca", "time_pca",
        "cumulative_variance",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_run_single_dataset_iris_accuracy_drop():
    """Accuracy drop with PCA on Iris should be less than 2%."""
    result = run_single_dataset("iris")
    acc_drop = result["acc_no_pca"] - result["acc_pca"]
    assert acc_drop < 0.02, f"Accuracy drop {acc_drop:.4f} exceeds 2%"


def test_run_single_dataset_digits_accuracy_drop():
    """Accuracy drop with PCA on Digits should be less than 2%."""
    result = run_single_dataset("digits")
    acc_drop = result["acc_no_pca"] - result["acc_pca"]
    assert acc_drop < 0.02, f"Accuracy drop {acc_drop:.4f} exceeds 2%"


def test_run_single_dataset_iris_original_features():
    """Iris should have 4 original features."""
    result = run_single_dataset("iris")
    assert result["n_original_features"] == 4


def test_run_single_dataset_digits_original_features():
    """Digits should have 64 original features."""
    result = run_single_dataset("digits")
    assert result["n_original_features"] == 64


def test_run_single_dataset_cumulative_variance_monotone():
    """Cumulative variance should be monotonically non-decreasing."""
    result = run_single_dataset("iris")
    cumvar = result["cumulative_variance"]
    diffs = np.diff(cumvar)
    assert np.all(diffs >= -1e-10), "Cumulative variance should be non-decreasing"


def test_run_single_dataset_cumulative_variance_max():
    """Cumulative variance should reach approximately 1.0 at the last component."""
    result = run_single_dataset("iris")
    cumvar = result["cumulative_variance"]
    assert abs(cumvar[-1] - 1.0) < 1e-6, f"Final cumvar {cumvar[-1]:.6f} should be ~1.0"


# ── Full Experiment Integration Test ──────────────────────────────────────────

def test_run_experiment_3_keys():
    """run_experiment_3 should return iris, digits, and summary_df keys."""
    results = run_experiment_3()
    assert "iris" in results
    assert "digits" in results
    assert "summary_df" in results


def test_run_experiment_3_summary_df_shape():
    """Summary DataFrame should have 2 rows (iris and digits)."""
    results = run_experiment_3()
    assert len(results["summary_df"]) == 2


def test_run_experiment_3_pca_reduces_digits():
    """PCA should significantly reduce Digits dimensionality."""
    results = run_experiment_3()
    n_pca = results["digits"]["n_pca_components"]
    assert n_pca < 64, f"PCA should reduce from 64 features, got {n_pca}"
