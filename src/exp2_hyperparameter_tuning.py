"""
Experiment 2: Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV
=============================================================================
Compares exhaustive grid search vs. randomized search for KNeighborsClassifier
on the Iris dataset, measuring performance and computational cost.

Using scikit-learn GridSearchCV / RandomizedSearchCV — Context7 confirmed
(/websites/scikit-learn_dev)
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier


def load_iris_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset.

    Returns
    -------
    X : np.ndarray of shape (150, 4)
    y : np.ndarray of shape (150,)
    """
    iris = load_iris()
    return iris.data, iris.target


def build_param_grid() -> dict[str, list]:
    """
    Define the hyperparameter grid for KNeighborsClassifier.

    Grid covers:
    - n_neighbors: [1, 3, 5, 7, 9, 11]
    - metric: ['euclidean', 'manhattan']

    Returns
    -------
    dict mapping parameter names to lists of candidate values.
    """
    return {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
        "metric": ["euclidean", "manhattan"],
    }


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list],
    cv: int = 5,
    scoring: str = "accuracy",
) -> tuple[GridSearchCV, float]:
    """
    Run exhaustive GridSearchCV over the parameter grid.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    param_grid : dict
        Hyperparameter grid.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.

    Returns
    -------
    grid_search : fitted GridSearchCV object
    runtime_seconds : float
        Wall-clock time for the search.
    """
    estimator = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    runtime = time.time() - t0

    return grid_search, runtime


def run_randomized_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list],
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> tuple[RandomizedSearchCV, float]:
    """
    Run RandomizedSearchCV over the parameter grid.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    param_grid : dict
        Hyperparameter distributions / lists.
    n_iter : int
        Number of parameter settings sampled.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    rand_search : fitted RandomizedSearchCV object
    runtime_seconds : float
        Wall-clock time for the search.
    """
    estimator = KNeighborsClassifier()
    rand_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        verbose=1,
    )

    t0 = time.time()
    rand_search.fit(X_train, y_train)
    runtime = time.time() - t0

    return rand_search, runtime


def evaluate_best_estimator(
    search_obj: GridSearchCV | RandomizedSearchCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Evaluate the best estimator from a search object on the held-out test set.

    Parameters
    ----------
    search_obj : GridSearchCV or RandomizedSearchCV
        Fitted search object.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    test_accuracy : float
    """
    y_pred = search_obj.best_estimator_.predict(X_test)
    return accuracy_score(y_test, y_pred)


def build_comparison_table(
    grid_search: GridSearchCV,
    rand_search: RandomizedSearchCV,
    grid_runtime: float,
    rand_runtime: float,
    grid_test_acc: float,
    rand_test_acc: float,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame summarising both search strategies.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted grid search object.
    rand_search : RandomizedSearchCV
        Fitted randomized search object.
    grid_runtime : float
        Wall-clock time for grid search (seconds).
    rand_runtime : float
        Wall-clock time for randomized search (seconds).
    grid_test_acc : float
        Test accuracy of best grid search estimator.
    rand_test_acc : float
        Test accuracy of best randomized search estimator.

    Returns
    -------
    pd.DataFrame with columns: method, best_params, best_cv_score,
                                test_accuracy, runtime_seconds
    """
    rows = [
        {
            "method": "GridSearchCV",
            "best_params": str(grid_search.best_params_),
            "best_cv_score": round(grid_search.best_score_, 4),
            "test_accuracy": round(grid_test_acc, 4),
            "runtime_seconds": round(grid_runtime, 4),
        },
        {
            "method": "RandomizedSearchCV",
            "best_params": str(rand_search.best_params_),
            "best_cv_score": round(rand_search.best_score_, 4),
            "test_accuracy": round(rand_test_acc, 4),
            "runtime_seconds": round(rand_runtime, 4),
        },
    ]
    return pd.DataFrame(rows)


def run_experiment_2() -> dict[str, Any]:
    """
    Execute the full Experiment 2 workflow:
    1. Load Iris dataset and split 70/30.
    2. Define KNN parameter grid.
    3. Run GridSearchCV (cv=5, scoring='accuracy').
    4. Run RandomizedSearchCV (n_iter=10, cv=5, random_state=42).
    5. Evaluate both best estimators on the test set.
    6. Print comparison table.

    Returns
    -------
    dict containing search objects, runtimes, and comparison DataFrame.
    """
    np.random.seed(42)

    print("\n" + "="*60)
    print("  EXP 2 — Hyperparameter Tuning")
    print("="*60)

    # 1. Data
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"\nIris | Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. Parameter grid
    param_grid = build_param_grid()
    total_combinations = len(param_grid["n_neighbors"]) * len(param_grid["metric"])
    print(f"Grid combinations: {total_combinations}")

    # 3. GridSearchCV
    print("\n--- Running GridSearchCV ---")
    grid_search, grid_runtime = run_grid_search(X_train, y_train, param_grid)
    grid_test_acc = evaluate_best_estimator(grid_search, X_test, y_test)

    print(f"  Best params : {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    print(f"  Test accuracy: {grid_test_acc:.4f}")
    print(f"  Runtime      : {grid_runtime:.4f}s")

    # 4. RandomizedSearchCV
    print("\n--- Running RandomizedSearchCV ---")
    rand_search, rand_runtime = run_randomized_search(
        X_train, y_train, param_grid, n_iter=10
    )
    rand_test_acc = evaluate_best_estimator(rand_search, X_test, y_test)

    print(f"  Best params : {rand_search.best_params_}")
    print(f"  Best CV score: {rand_search.best_score_:.4f}")
    print(f"  Test accuracy: {rand_test_acc:.4f}")
    print(f"  Runtime      : {rand_runtime:.4f}s")

    # 5. Comparison table
    comparison_df = build_comparison_table(
        grid_search, rand_search,
        grid_runtime, rand_runtime,
        grid_test_acc, rand_test_acc,
    )

    print("\n--- Comparison Table ---")
    print(comparison_df.to_string(index=False))

    return {
        "grid_search": grid_search,
        "rand_search": rand_search,
        "grid_runtime": grid_runtime,
        "rand_runtime": rand_runtime,
        "grid_test_acc": grid_test_acc,
        "rand_test_acc": rand_test_acc,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    run_experiment_2()
