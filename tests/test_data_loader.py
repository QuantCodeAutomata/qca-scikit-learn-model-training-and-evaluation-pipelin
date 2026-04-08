"""
Tests for src/data_loader.py

Validates:
  - Dataset loading (Iris, Breast Cancer, California Housing)
  - Dataset inspection (no crashes, correct output)
  - Train/test split ratios and shapes
  - StandardScaler applied correctly (no data leakage)
  - Edge cases and input validation
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    load_classification_dataset,
    load_regression_dataset,
    inspect_dataset,
    split_and_scale,
)


# ── Dataset loading ──────────────────────────────────────────────────────────

class TestLoadClassificationDataset:
    def test_iris_loads_correctly(self):
        ds = load_classification_dataset("iris")
        assert "X" in ds and "y" in ds
        assert ds["X"].shape == (150, 4)
        assert ds["y"].shape == (150,)
        assert len(ds["feature_names"]) == 4
        assert len(ds["target_names"]) == 3

    def test_breast_cancer_loads_correctly(self):
        ds = load_classification_dataset("breast_cancer")
        assert ds["X"].shape[0] == 569
        assert ds["X"].shape[1] == 30
        assert len(ds["target_names"]) == 2

    def test_unknown_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown classification dataset"):
            load_classification_dataset("unknown_dataset")

    def test_iris_no_missing_values(self):
        ds = load_classification_dataset("iris")
        assert not np.isnan(ds["X"]).any(), "Iris features should have no NaN values"

    def test_iris_target_values_in_range(self):
        ds = load_classification_dataset("iris")
        assert set(np.unique(ds["y"])).issubset({0, 1, 2})


class TestLoadRegressionDataset:
    def test_california_housing_loads_correctly(self):
        ds = load_regression_dataset("california_housing")
        assert "X" in ds and "y" in ds
        assert ds["X"].shape[1] == 8
        assert ds["X"].shape[0] > 10000
        assert len(ds["feature_names"]) == 8

    def test_california_housing_no_missing_values(self):
        ds = load_regression_dataset("california_housing")
        assert not np.isnan(ds["X"]).any()
        assert not np.isnan(ds["y"]).any()

    def test_unknown_regression_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown regression dataset"):
            load_regression_dataset("boston")


# ── Dataset inspection ───────────────────────────────────────────────────────

class TestInspectDataset:
    def test_inspect_classification_does_not_crash(self, capsys):
        ds = load_classification_dataset("iris")
        inspect_dataset(ds, task="classification")
        captured = capsys.readouterr()
        assert "Shape" in captured.out
        assert "Feature names" in captured.out
        assert "Target names" in captured.out

    def test_inspect_regression_does_not_crash(self, capsys):
        ds = load_regression_dataset("california_housing")
        inspect_dataset(ds, task="regression")
        captured = capsys.readouterr()
        assert "Shape" in captured.out


# ── Split and scale ──────────────────────────────────────────────────────────

class TestSplitAndScale:
    @pytest.fixture
    def iris_data(self):
        ds = load_classification_dataset("iris")
        return ds["X"], ds["y"]

    def test_split_ratio_80_20(self, iris_data):
        X, y = iris_data
        X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
        total = len(y_train) + len(y_test)
        assert total == len(y)
        # Allow ±1 sample tolerance for rounding
        assert abs(len(y_test) / total - 0.2) < 0.02

    def test_split_shapes_consistent(self, iris_data):
        X, y = iris_data
        X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1] == X.shape[1]

    def test_scaler_returns_standard_scaler(self, iris_data):
        X, y = iris_data
        _, _, _, _, scaler = split_and_scale(X, y)
        assert isinstance(scaler, StandardScaler)

    def test_scaled_train_mean_near_zero(self, iris_data):
        """After StandardScaler, training data should have ~0 mean per feature."""
        X, y = iris_data
        X_train_scaled, _, _, _, _ = split_and_scale(X, y)
        means = np.abs(X_train_scaled.mean(axis=0))
        assert np.all(means < 1e-10), f"Training means not near zero: {means}"

    def test_scaled_train_std_near_one(self, iris_data):
        """After StandardScaler, training data should have ~1 std per feature."""
        X, y = iris_data
        X_train_scaled, _, _, _, _ = split_and_scale(X, y)
        stds = X_train_scaled.std(axis=0)
        assert np.allclose(stds, 1.0, atol=1e-10), f"Training stds not near 1: {stds}"

    def test_reproducibility_with_same_random_state(self, iris_data):
        """Same random_state must produce identical splits."""
        X, y = iris_data
        X_tr1, X_te1, y_tr1, y_te1, _ = split_and_scale(X, y, random_state=42)
        X_tr2, X_te2, y_tr2, y_te2, _ = split_and_scale(X, y, random_state=42)
        np.testing.assert_array_equal(y_tr1, y_tr2)
        np.testing.assert_array_equal(y_te1, y_te2)

    def test_different_random_states_produce_different_splits(self, iris_data):
        """Different random states should (almost certainly) produce different splits."""
        X, y = iris_data
        _, _, y_tr1, _, _ = split_and_scale(X, y, random_state=42)
        _, _, y_tr2, _, _ = split_and_scale(X, y, random_state=99)
        assert not np.array_equal(y_tr1, y_tr2)

    def test_no_data_leakage_test_scaler_not_fitted_on_test(self, iris_data):
        """
        Verify no data leakage: the scaler is fit only on training data.
        Test set mean should NOT be zero (it's transformed, not fitted).
        """
        X, y = iris_data
        X_train_scaled, X_test_scaled, _, _, scaler = split_and_scale(X, y)
        # The test set mean after transform should differ from 0 (not fitted on test)
        test_means = X_test_scaled.mean(axis=0)
        # At least one feature mean should be non-zero for the test set
        assert not np.allclose(test_means, 0.0, atol=1e-3), (
            "Test set means are all zero — scaler may have been fitted on test data (leakage)."
        )
