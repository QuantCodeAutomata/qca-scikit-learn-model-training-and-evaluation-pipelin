"""
Experiment 1: Scikit-Learn Model Training and Evaluation Pipeline
=================================================================
Implements a standard supervised learning workflow covering:
- Data loading and train/test splitting
- Preprocessing (StandardScaler, Normalizer, Binarizer)
- Classification: KNeighborsClassifier, SVC
- Regression: LinearRegression
- Evaluation metrics for both task types

Using scikit-learn Pipeline — Context7 confirmed (/websites/scikit-learn_dev)
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, Normalizer, StandardScaler
from sklearn.svm import SVC


def load_classification_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the Iris dataset for classification tasks.

    Returns
    -------
    X : np.ndarray of shape (150, 4)
    y : np.ndarray of shape (150,)
    feature_names : list of str
    """
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names)


def load_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    noise: float = 20.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset via make_regression.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Total number of features.
    noise : float
        Standard deviation of Gaussian noise added to the output.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training (70%) and test (30%) sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray tuples
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_knn_pipeline(n_neighbors: int = 5) -> Pipeline:
    """
    Build a Pipeline: StandardScaler → KNeighborsClassifier.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours for KNN.

    Returns
    -------
    Pipeline
    """
    # Using scikit-learn Pipeline — Context7 confirmed
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def build_svc_pipeline() -> Pipeline:
    """
    Build a Pipeline: StandardScaler → SVC.

    Returns
    -------
    Pipeline
    """
    # Using scikit-learn Pipeline — Context7 confirmed
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=42)),
        ]
    )


def build_regression_pipeline() -> Pipeline:
    """
    Build a Pipeline: Normalizer → LinearRegression.

    Returns
    -------
    Pipeline
    """
    # Using scikit-learn Pipeline — Context7 confirmed
    return Pipeline(
        steps=[
            ("normalizer", Normalizer()),
            ("lr", LinearRegression()),
        ]
    )


def evaluate_classifier(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Classifier",
) -> dict[str, Any]:
    """
    Fit a classification pipeline and compute evaluation metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline ending with a classifier.
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Target vectors.
    model_name : str
        Label for display purposes.

    Returns
    -------
    dict with keys: model_name, accuracy, confusion_matrix
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {"model_name": model_name, "accuracy": acc, "confusion_matrix": cm}


def evaluate_regressor(
    pipeline: Pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Regressor",
) -> dict[str, Any]:
    """
    Fit a regression pipeline and compute evaluation metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline ending with a regressor.
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Target vectors.
    model_name : str
        Label for display purposes.

    Returns
    -------
    dict with keys: model_name, mae, mse, r2
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"model_name": model_name, "mae": mae, "mse": mse, "r2": r2}


def demonstrate_preprocessing(X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Demonstrate StandardScaler, Normalizer, and Binarizer on a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix.

    Returns
    -------
    dict mapping transformer name → transformed array (first 3 rows shown)
    """
    scaler = StandardScaler()
    normalizer = Normalizer()
    binarizer = Binarizer(threshold=0.0)

    X_scaled = scaler.fit_transform(X)
    X_normalized = normalizer.fit_transform(X)
    X_binarized = binarizer.fit_transform(X)

    print("\n--- Preprocessing Demonstration (first 3 rows) ---")
    print(f"Original:\n{X[:3]}")
    print(f"StandardScaler:\n{X_scaled[:3]}")
    print(f"Normalizer:\n{X_normalized[:3]}")
    print(f"Binarizer:\n{X_binarized[:3]}")

    return {
        "scaled": X_scaled,
        "normalized": X_normalized,
        "binarized": X_binarized,
    }


def run_experiment_1() -> dict[str, Any]:
    """
    Execute the full Experiment 1 workflow:
    1. Load Iris (classification) and synthetic regression datasets.
    2. Split 70/30 with random_state=42.
    3. Demonstrate preprocessing transformers.
    4. Fit and evaluate KNN and SVC classifiers.
    5. Fit and evaluate LinearRegression.

    Returns
    -------
    dict containing all evaluation results.
    """
    np.random.seed(42)
    results: dict[str, Any] = {}

    # ── Classification ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXP 1 — Classification (Iris Dataset)")
    print("="*60)

    X_cls, y_cls, feature_names = load_classification_data()
    X_train_c, X_test_c, y_train_c, y_test_c = split_data(X_cls, y_cls)

    print(f"\nDataset: Iris | Train: {len(X_train_c)} | Test: {len(X_test_c)}")
    print(f"Features: {feature_names}")

    # Preprocessing demo on raw classification features
    demonstrate_preprocessing(X_cls)

    # KNN
    knn_pipe = build_knn_pipeline(n_neighbors=5)
    knn_results = evaluate_classifier(
        knn_pipe, X_train_c, X_test_c, y_train_c, y_test_c,
        model_name="KNeighborsClassifier (n_neighbors=5)"
    )
    results["knn"] = knn_results

    # SVC
    svc_pipe = build_svc_pipeline()
    svc_results = evaluate_classifier(
        svc_pipe, X_train_c, X_test_c, y_train_c, y_test_c,
        model_name="SVC"
    )
    results["svc"] = svc_results

    # ── Regression ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXP 1 — Regression (Synthetic Dataset)")
    print("="*60)

    X_reg, y_reg = load_regression_data()
    X_train_r, X_test_r, y_train_r, y_test_r = split_data(X_reg, y_reg)

    print(f"\nDataset: make_regression | Train: {len(X_train_r)} | Test: {len(X_test_r)}")

    lr_pipe = build_regression_pipeline()
    lr_results = evaluate_regressor(
        lr_pipe, X_train_r, X_test_r, y_train_r, y_test_r,
        model_name="LinearRegression"
    )
    results["linear_regression"] = lr_results

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXP 1 — Summary")
    print("="*60)
    summary_rows = []
    for key in ["knn", "svc"]:
        r = results[key]
        summary_rows.append({
            "Model": r["model_name"],
            "Task": "Classification",
            "Metric": "Accuracy",
            "Value": f"{r['accuracy']:.4f}",
        })
    r = results["linear_regression"]
    for metric, val in [("MAE", r["mae"]), ("MSE", r["mse"]), ("R²", r["r2"])]:
        summary_rows.append({
            "Model": r["model_name"],
            "Task": "Regression",
            "Metric": metric,
            "Value": f"{val:.4f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    results["summary_df"] = df_summary
    return results


if __name__ == "__main__":
    run_experiment_1()
