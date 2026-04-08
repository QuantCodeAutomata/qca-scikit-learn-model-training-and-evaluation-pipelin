"""
Data loading utilities for the scikit-learn ML pipeline experiment.

Loads benchmark datasets from sklearn.datasets as specified in the experiment
methodology (Iris for classification, California Housing for regression).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any


def load_classification_dataset(
    dataset_name: str = "iris",
) -> Dict[str, Any]:
    """
    Load a classification benchmark dataset from sklearn.datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. One of 'iris' or 'breast_cancer'.

    Returns
    -------
    dict with keys:
        X : np.ndarray  — feature matrix
        y : np.ndarray  — target vector
        feature_names : list[str]
        target_names  : list[str]
        description   : str
    """
    if dataset_name == "iris":
        raw = load_iris()
    elif dataset_name == "breast_cancer":
        raw = load_breast_cancer()
    else:
        raise ValueError(f"Unknown classification dataset: {dataset_name!r}. "
                         "Choose 'iris' or 'breast_cancer'.")

    return {
        "X": raw.data,
        "y": raw.target,
        "feature_names": list(raw.feature_names),
        "target_names": list(raw.target_names),
        "description": raw.DESCR,
    }


def load_regression_dataset(
    dataset_name: str = "california_housing",
) -> Dict[str, Any]:
    """
    Load a regression benchmark dataset from sklearn.datasets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Currently supports 'california_housing'.

    Returns
    -------
    dict with keys:
        X : np.ndarray  — feature matrix
        y : np.ndarray  — target vector
        feature_names : list[str]
        description   : str
    """
    if dataset_name == "california_housing":
        raw = fetch_california_housing()
    else:
        raise ValueError(f"Unknown regression dataset: {dataset_name!r}. "
                         "Choose 'california_housing'.")

    return {
        "X": raw.data,
        "y": raw.target,
        "feature_names": list(raw.feature_names),
        "description": raw.DESCR,
    }


def inspect_dataset(dataset: Dict[str, Any], task: str = "classification") -> None:
    """
    Print a summary of the dataset: shape, feature names, target names,
    and missing value counts — as required by methodology step 2.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary returned by load_classification_dataset or
        load_regression_dataset.
    task : str
        'classification' or 'regression'.
    """
    X, y = dataset["X"], dataset["y"]
    print(f"  Shape          : X={X.shape}, y={y.shape}")
    print(f"  Feature names  : {dataset['feature_names']}")
    if task == "classification":
        print(f"  Target names   : {dataset['target_names']}")
    print(f"  Missing values : {np.isnan(X).sum()} (features), "
          f"{np.isnan(y.astype(float)).sum()} (target)")


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split data into train/test sets (80/20) and apply StandardScaler.

    The scaler is fit ONLY on training data to prevent data leakage
    (methodology steps 3–4).

    Parameters
    ----------
    X : np.ndarray  — feature matrix
    y : np.ndarray  — target vector
    test_size : float  — fraction of data for test set (default 0.2)
    random_state : int — random seed for reproducibility (default 42)

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit scaler on training data only — prevents data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
