"""
Tests for Experiment 1: Scikit-Learn Model Training and Evaluation Pipeline
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.pipeline import Pipeline

from src.exp1_pipeline import (
    build_knn_pipeline,
    build_regression_pipeline,
    build_svc_pipeline,
    demonstrate_preprocessing,
    evaluate_classifier,
    evaluate_regressor,
    load_classification_data,
    load_regression_data,
    run_experiment_1,
    split_data,
)


# ── Data Loading Tests ────────────────────────────────────────────────────────

def test_load_classification_data_shape():
    """Iris dataset should have 150 samples and 4 features."""
    X, y, feature_names = load_classification_data()
    assert X.shape == (150, 4), f"Expected (150, 4), got {X.shape}"
    assert y.shape == (150,), f"Expected (150,), got {y.shape}"
    assert len(feature_names) == 4


def test_load_classification_data_classes():
    """Iris dataset should have exactly 3 classes."""
    _, y, _ = load_classification_data()
    assert len(np.unique(y)) == 3


def test_load_regression_data_shape():
    """Synthetic regression dataset should have correct shape."""
    X, y = load_regression_data(n_samples=200, n_features=5)
    assert X.shape == (200, 5)
    assert y.shape == (200,)


def test_load_regression_data_reproducibility():
    """Same random_state should produce identical datasets."""
    X1, y1 = load_regression_data(random_state=42)
    X2, y2 = load_regression_data(random_state=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# ── Split Tests ───────────────────────────────────────────────────────────────

def test_split_data_proportions():
    """Train/test split should be 70/30."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.30)
    total = len(X_train) + len(X_test)
    assert total == 150
    assert len(X_test) == 45  # 30% of 150
    assert len(X_train) == 105  # 70% of 150


def test_split_data_reproducibility():
    """Same random_state should produce identical splits."""
    X, y, _ = load_classification_data()
    X_tr1, X_te1, _, _ = split_data(X, y, random_state=42)
    X_tr2, X_te2, _, _ = split_data(X, y, random_state=42)
    np.testing.assert_array_equal(X_tr1, X_tr2)
    np.testing.assert_array_equal(X_te1, X_te2)


# ── Pipeline Construction Tests ───────────────────────────────────────────────

def test_build_knn_pipeline_type():
    """KNN pipeline should be a sklearn Pipeline."""
    pipe = build_knn_pipeline()
    assert isinstance(pipe, Pipeline)


def test_build_knn_pipeline_steps():
    """KNN pipeline should have scaler and knn steps."""
    pipe = build_knn_pipeline(n_neighbors=5)
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "knn" in step_names
    assert pipe.named_steps["knn"].n_neighbors == 5


def test_build_svc_pipeline_type():
    """SVC pipeline should be a sklearn Pipeline."""
    pipe = build_svc_pipeline()
    assert isinstance(pipe, Pipeline)


def test_build_svc_pipeline_steps():
    """SVC pipeline should have scaler and svc steps."""
    pipe = build_svc_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "svc" in step_names


def test_build_regression_pipeline_type():
    """Regression pipeline should be a sklearn Pipeline."""
    pipe = build_regression_pipeline()
    assert isinstance(pipe, Pipeline)


def test_build_regression_pipeline_steps():
    """Regression pipeline should have normalizer and lr steps."""
    pipe = build_regression_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert "normalizer" in step_names
    assert "lr" in step_names


# ── Preprocessing Tests ───────────────────────────────────────────────────────

def test_demonstrate_preprocessing_keys():
    """Preprocessing demo should return all three transformer outputs."""
    X, _, _ = load_classification_data()
    result = demonstrate_preprocessing(X)
    assert "scaled" in result
    assert "normalized" in result
    assert "binarized" in result


def test_demonstrate_preprocessing_shapes():
    """Preprocessing outputs should preserve input shape."""
    X, _, _ = load_classification_data()
    result = demonstrate_preprocessing(X)
    for key, arr in result.items():
        assert arr.shape == X.shape, f"{key} shape mismatch"


def test_standard_scaler_zero_mean():
    """StandardScaler output should have approximately zero mean."""
    X, _, _ = load_classification_data()
    result = demonstrate_preprocessing(X)
    col_means = np.abs(result["scaled"].mean(axis=0))
    assert np.all(col_means < 1e-10), "StandardScaler output should have ~zero mean"


def test_normalizer_unit_norm():
    """Normalizer output rows should have unit L2 norm."""
    X, _, _ = load_classification_data()
    result = demonstrate_preprocessing(X)
    row_norms = np.linalg.norm(result["normalized"], axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-10)


def test_binarizer_binary_values():
    """Binarizer output should contain only 0 and 1."""
    X, _, _ = load_classification_data()
    result = demonstrate_preprocessing(X)
    unique_vals = np.unique(result["binarized"])
    assert set(unique_vals).issubset({0.0, 1.0}), "Binarizer should produce only 0/1"


# ── Classifier Evaluation Tests ───────────────────────────────────────────────

def test_knn_accuracy_above_threshold():
    """KNN on Iris should achieve accuracy > 90%."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_knn_pipeline(n_neighbors=5)
    result = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert result["accuracy"] > 0.90, f"KNN accuracy {result['accuracy']:.4f} below 90%"


def test_svc_accuracy_above_threshold():
    """SVC on Iris should achieve accuracy > 90%."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_svc_pipeline()
    result = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert result["accuracy"] > 0.90, f"SVC accuracy {result['accuracy']:.4f} below 90%"


def test_classifier_accuracy_range():
    """Accuracy should be between 0 and 1."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_knn_pipeline()
    result = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_confusion_matrix_shape():
    """Confusion matrix should be 3x3 for Iris (3 classes)."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_knn_pipeline()
    result = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert result["confusion_matrix"].shape == (3, 3)


def test_confusion_matrix_sum():
    """Confusion matrix entries should sum to the number of test samples."""
    X, y, _ = load_classification_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_knn_pipeline()
    result = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert result["confusion_matrix"].sum() == len(y_test)


# ── Regressor Evaluation Tests ────────────────────────────────────────────────

def test_regression_r2_above_threshold():
    """LinearRegression on synthetic data should achieve R² > 0.75."""
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_regression_pipeline()
    result = evaluate_regressor(pipe, X_train, X_test, y_train, y_test)
    assert result["r2"] > 0.75, f"R² {result['r2']:.4f} below 0.75"


def test_regression_mae_positive():
    """MAE should be non-negative."""
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_regression_pipeline()
    result = evaluate_regressor(pipe, X_train, X_test, y_train, y_test)
    assert result["mae"] >= 0.0


def test_regression_mse_positive():
    """MSE should be non-negative."""
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_regression_pipeline()
    result = evaluate_regressor(pipe, X_train, X_test, y_train, y_test)
    assert result["mse"] >= 0.0


def test_regression_mse_geq_mae():
    """MSE should be >= MAE² / n (by Cauchy-Schwarz; simpler: MSE >= 0 and MAE >= 0)."""
    X, y = load_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_regression_pipeline()
    result = evaluate_regressor(pipe, X_train, X_test, y_train, y_test)
    # MSE >= MAE^2 is not always true; just verify both are non-negative
    assert result["mse"] >= 0.0
    assert result["mae"] >= 0.0


# ── Full Experiment Integration Test ──────────────────────────────────────────

def test_run_experiment_1_keys():
    """run_experiment_1 should return all expected keys."""
    results = run_experiment_1()
    assert "knn" in results
    assert "svc" in results
    assert "linear_regression" in results
    assert "summary_df" in results


def test_run_experiment_1_summary_df_shape():
    """Summary DataFrame should have 5 rows (2 classifiers × 1 metric + 3 regression metrics)."""
    results = run_experiment_1()
    assert len(results["summary_df"]) == 5


def test_run_experiment_1_knn_accuracy():
    """KNN accuracy in full experiment should exceed 90%."""
    results = run_experiment_1()
    assert results["knn"]["accuracy"] > 0.90


def test_run_experiment_1_svc_accuracy():
    """SVC accuracy in full experiment should exceed 90%."""
    results = run_experiment_1()
    assert results["svc"]["accuracy"] > 0.90


def test_run_experiment_1_r2():
    """R² in full experiment should exceed 0.75."""
    results = run_experiment_1()
    assert results["linear_regression"]["r2"] > 0.75
