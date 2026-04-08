"""
Tests for src/classification_pipeline.py

Validates:
  - Logistic Regression and KNN training
  - Prediction shapes and value ranges
  - Evaluation metrics (accuracy, confusion matrix, classification report)
  - Expected accuracy thresholds on Iris dataset (≥ 90%)
  - Edge cases
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classification_pipeline import (
    train_logistic_regression,
    train_knn_classifier,
    evaluate_classifier,
    run_classification_experiment,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def iris_split():
    """Return scaled train/test split of the Iris dataset."""
    raw = load_iris()
    X, y = raw.data, raw.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, list(raw.target_names)


# ── Model training ────────────────────────────────────────────────────────────

class TestTrainLogisticRegression:
    def test_returns_fitted_logistic_regression(self, iris_split):
        X_train, _, y_train, _, _ = iris_split
        model = train_logistic_regression(X_train, y_train)
        assert isinstance(model, LogisticRegression)
        assert hasattr(model, "coef_"), "Model should be fitted (has coef_)"

    def test_predict_returns_correct_shape(self, iris_split):
        X_train, X_test, y_train, y_test, _ = iris_split
        model = train_logistic_regression(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_predict_values_are_valid_classes(self, iris_split):
        X_train, X_test, y_train, _, _ = iris_split
        model = train_logistic_regression(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1, 2})


class TestTrainKNNClassifier:
    def test_returns_fitted_knn(self, iris_split):
        X_train, _, y_train, _, _ = iris_split
        model = train_knn_classifier(X_train, y_train, n_neighbors=5)
        assert isinstance(model, KNeighborsClassifier)
        assert model.n_neighbors == 5

    def test_predict_returns_correct_shape(self, iris_split):
        X_train, X_test, y_train, y_test, _ = iris_split
        model = train_knn_classifier(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_default_k_is_5(self, iris_split):
        X_train, _, y_train, _, _ = iris_split
        model = train_knn_classifier(X_train, y_train)
        assert model.n_neighbors == 5


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestEvaluateClassifier:
    def test_accuracy_is_between_0_and_1(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_logistic_regression(X_train, y_train)
        results = evaluate_classifier(model, X_test, y_test, target_names, "LR")
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_logistic_regression(X_train, y_train)
        results = evaluate_classifier(model, X_test, y_test, target_names, "LR")
        n_classes = len(target_names)
        assert results["confusion_matrix"].shape == (n_classes, n_classes)

    def test_confusion_matrix_sum_equals_test_size(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_logistic_regression(X_train, y_train)
        results = evaluate_classifier(model, X_test, y_test, target_names, "LR")
        assert results["confusion_matrix"].sum() == len(y_test)

    def test_classification_report_is_string(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_logistic_regression(X_train, y_train)
        results = evaluate_classifier(model, X_test, y_test, target_names, "LR")
        assert isinstance(results["classification_report"], str)
        assert "precision" in results["classification_report"]

    def test_y_pred_shape_matches_y_test(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_knn_classifier(X_train, y_train)
        results = evaluate_classifier(model, X_test, y_test, target_names, "KNN")
        assert results["y_pred"].shape == y_test.shape


# ── Full experiment ───────────────────────────────────────────────────────────

class TestRunClassificationExperiment:
    def test_returns_two_result_dicts(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        lr_res, knn_res = run_classification_experiment(
            X_train, X_test, y_train, y_test, target_names
        )
        assert isinstance(lr_res, dict)
        assert isinstance(knn_res, dict)

    def test_lr_accuracy_above_90_percent_on_iris(self, iris_split):
        """
        Regression test: Logistic Regression should achieve ≥ 90% accuracy
        on the Iris dataset (expected outcome from experiment spec).
        """
        X_train, X_test, y_train, y_test, target_names = iris_split
        lr_res, _ = run_classification_experiment(
            X_train, X_test, y_train, y_test, target_names
        )
        assert lr_res["accuracy"] >= 0.90, (
            f"LR accuracy {lr_res['accuracy']:.4f} is below expected 90% threshold"
        )

    def test_knn_accuracy_above_90_percent_on_iris(self, iris_split):
        """
        Regression test: KNN (k=5) should achieve ≥ 90% accuracy on Iris.
        """
        X_train, X_test, y_train, y_test, target_names = iris_split
        _, knn_res = run_classification_experiment(
            X_train, X_test, y_train, y_test, target_names
        )
        assert knn_res["accuracy"] >= 0.90, (
            f"KNN accuracy {knn_res['accuracy']:.4f} is below expected 90% threshold"
        )

    def test_result_dicts_have_required_keys(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        lr_res, knn_res = run_classification_experiment(
            X_train, X_test, y_train, y_test, target_names
        )
        required_keys = {"model_name", "accuracy", "confusion_matrix",
                         "classification_report", "y_pred"}
        assert required_keys.issubset(lr_res.keys())
        assert required_keys.issubset(knn_res.keys())

    def test_model_names_are_correct(self, iris_split):
        X_train, X_test, y_train, y_test, target_names = iris_split
        lr_res, knn_res = run_classification_experiment(
            X_train, X_test, y_train, y_test, target_names
        )
        assert "Logistic" in lr_res["model_name"]
        assert "KNeighbors" in knn_res["model_name"] or "KNN" in knn_res["model_name"]


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_class_data_raises_value_error(self):
        """
        LogisticRegression requires at least 2 classes.
        Verify it raises ValueError for single-class data (expected sklearn behaviour).
        """
        import pytest
        X = np.random.randn(20, 4)
        y = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="at least 2 classes"):
            train_logistic_regression(X, y)

    def test_large_k_knn(self, iris_split):
        """KNN with k equal to training size should still work."""
        X_train, X_test, y_train, y_test, target_names = iris_split
        model = train_knn_classifier(X_train, y_train, n_neighbors=len(y_train))
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_two_class_problem(self):
        """Binary classification should work correctly."""
        from sklearn.datasets import load_breast_cancer
        raw = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            raw.data, raw.target, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = train_logistic_regression(X_train_s, y_train)
        results = evaluate_classifier(
            model, X_test_s, y_test, list(raw.target_names), "LR Binary"
        )
        assert results["accuracy"] > 0.90
        assert results["confusion_matrix"].shape == (2, 2)
