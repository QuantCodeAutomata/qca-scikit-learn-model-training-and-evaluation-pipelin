"""
Tests for Experiment 2: Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV
"""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exp2_hyperparameter_tuning import (
    build_comparison_table,
    build_param_grid,
    evaluate_best_estimator,
    load_iris_data,
    run_experiment_2,
    run_grid_search,
    run_randomized_search,
)


# ── Data Loading Tests ────────────────────────────────────────────────────────

def test_load_iris_data_shape():
    """Iris dataset should have 150 samples and 4 features."""
    X, y = load_iris_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_iris_data_classes():
    """Iris dataset should have 3 classes."""
    _, y = load_iris_data()
    assert len(np.unique(y)) == 3


# ── Parameter Grid Tests ──────────────────────────────────────────────────────

def test_build_param_grid_keys():
    """Parameter grid should contain n_neighbors and metric keys."""
    grid = build_param_grid()
    assert "n_neighbors" in grid
    assert "metric" in grid


def test_build_param_grid_n_neighbors():
    """n_neighbors should be [1, 3, 5, 7, 9, 11]."""
    grid = build_param_grid()
    assert grid["n_neighbors"] == [1, 3, 5, 7, 9, 11]


def test_build_param_grid_metrics():
    """metric should be ['euclidean', 'manhattan']."""
    grid = build_param_grid()
    assert set(grid["metric"]) == {"euclidean", "manhattan"}


def test_param_grid_total_combinations():
    """Total grid combinations should be 6 × 2 = 12."""
    grid = build_param_grid()
    total = len(grid["n_neighbors"]) * len(grid["metric"])
    assert total == 12


# ── GridSearchCV Tests ────────────────────────────────────────────────────────

def test_run_grid_search_returns_fitted_object():
    """GridSearchCV should return a fitted object with best_params_."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    grid, runtime = run_grid_search(X_train, y_train, build_param_grid())
    assert isinstance(grid, GridSearchCV)
    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_score_")


def test_run_grid_search_best_params_valid():
    """Best params from GridSearchCV should be within the defined grid."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    param_grid = build_param_grid()
    grid, _ = run_grid_search(X_train, y_train, param_grid)
    assert grid.best_params_["n_neighbors"] in param_grid["n_neighbors"]
    assert grid.best_params_["metric"] in param_grid["metric"]


def test_run_grid_search_cv_score_range():
    """Best CV score should be between 0 and 1."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    grid, _ = run_grid_search(X_train, y_train, build_param_grid())
    assert 0.0 <= grid.best_score_ <= 1.0


def test_run_grid_search_runtime_positive():
    """GridSearchCV runtime should be positive."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    _, runtime = run_grid_search(X_train, y_train, build_param_grid())
    assert runtime > 0.0


# ── RandomizedSearchCV Tests ──────────────────────────────────────────────────

def test_run_randomized_search_returns_fitted_object():
    """RandomizedSearchCV should return a fitted object with best_params_."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    rand, runtime = run_randomized_search(X_train, y_train, build_param_grid())
    assert isinstance(rand, RandomizedSearchCV)
    assert hasattr(rand, "best_params_")
    assert hasattr(rand, "best_score_")


def test_run_randomized_search_n_iter():
    """RandomizedSearchCV should evaluate exactly n_iter parameter combinations."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    rand, _ = run_randomized_search(X_train, y_train, build_param_grid(), n_iter=10)
    assert len(rand.cv_results_["params"]) == 10


def test_run_randomized_search_cv_score_range():
    """Best CV score from RandomizedSearchCV should be between 0 and 1."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    rand, _ = run_randomized_search(X_train, y_train, build_param_grid())
    assert 0.0 <= rand.best_score_ <= 1.0


def test_run_randomized_search_reproducibility():
    """Same random_state should produce identical best_params_."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
    rand1, _ = run_randomized_search(X_train, y_train, build_param_grid(), random_state=42)
    rand2, _ = run_randomized_search(X_train, y_train, build_param_grid(), random_state=42)
    assert rand1.best_params_ == rand2.best_params_


# ── Evaluation Tests ──────────────────────────────────────────────────────────

def test_evaluate_best_estimator_accuracy_range():
    """Test accuracy should be between 0 and 1."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    grid, _ = run_grid_search(X_train, y_train, build_param_grid())
    acc = evaluate_best_estimator(grid, X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_evaluate_best_estimator_above_threshold():
    """Best estimator test accuracy should exceed 90% on Iris."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    grid, _ = run_grid_search(X_train, y_train, build_param_grid())
    acc = evaluate_best_estimator(grid, X_test, y_test)
    assert acc > 0.90, f"Test accuracy {acc:.4f} below 90%"


# ── Comparison Table Tests ────────────────────────────────────────────────────

def test_build_comparison_table_shape():
    """Comparison table should have 2 rows and 5 columns."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    param_grid = build_param_grid()
    grid, g_rt = run_grid_search(X_train, y_train, param_grid)
    rand, r_rt = run_randomized_search(X_train, y_train, param_grid)
    g_acc = evaluate_best_estimator(grid, X_test, y_test)
    r_acc = evaluate_best_estimator(rand, X_test, y_test)
    df = build_comparison_table(grid, rand, g_rt, r_rt, g_acc, r_acc)
    assert df.shape == (2, 5)


def test_build_comparison_table_columns():
    """Comparison table should have the correct column names."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    param_grid = build_param_grid()
    grid, g_rt = run_grid_search(X_train, y_train, param_grid)
    rand, r_rt = run_randomized_search(X_train, y_train, param_grid)
    g_acc = evaluate_best_estimator(grid, X_test, y_test)
    r_acc = evaluate_best_estimator(rand, X_test, y_test)
    df = build_comparison_table(grid, rand, g_rt, r_rt, g_acc, r_acc)
    expected_cols = {"method", "best_params", "best_cv_score", "test_accuracy", "runtime_seconds"}
    assert set(df.columns) == expected_cols


def test_build_comparison_table_methods():
    """Comparison table should contain GridSearchCV and RandomizedSearchCV rows."""
    X, y = load_iris_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    param_grid = build_param_grid()
    grid, g_rt = run_grid_search(X_train, y_train, param_grid)
    rand, r_rt = run_randomized_search(X_train, y_train, param_grid)
    g_acc = evaluate_best_estimator(grid, X_test, y_test)
    r_acc = evaluate_best_estimator(rand, X_test, y_test)
    df = build_comparison_table(grid, rand, g_rt, r_rt, g_acc, r_acc)
    assert "GridSearchCV" in df["method"].values
    assert "RandomizedSearchCV" in df["method"].values


# ── Full Experiment Integration Test ──────────────────────────────────────────

def test_run_experiment_2_keys():
    """run_experiment_2 should return all expected keys."""
    results = run_experiment_2()
    expected_keys = {
        "grid_search", "rand_search", "grid_runtime", "rand_runtime",
        "grid_test_acc", "rand_test_acc", "comparison_df",
    }
    assert expected_keys.issubset(set(results.keys()))


def test_run_experiment_2_grid_accuracy():
    """GridSearchCV best estimator should achieve > 90% test accuracy."""
    results = run_experiment_2()
    assert results["grid_test_acc"] > 0.90


def test_run_experiment_2_rand_accuracy():
    """RandomizedSearchCV best estimator should achieve > 90% test accuracy."""
    results = run_experiment_2()
    assert results["rand_test_acc"] > 0.90


def test_run_experiment_2_grid_explores_all_combinations():
    """GridSearchCV should evaluate all 12 parameter combinations."""
    results = run_experiment_2()
    n_evaluated = len(results["grid_search"].cv_results_["params"])
    assert n_evaluated == 12
