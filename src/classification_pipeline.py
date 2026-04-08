"""
Classification pipeline for the scikit-learn ML experiment.

Implements methodology steps 5–7:
  - Train LogisticRegression and KNeighborsClassifier (k=5)
  - Generate predictions on the test set
  - Evaluate with accuracy_score, confusion_matrix, classification_report

Using scikit-learn — Context7 confirmed (/websites/scikit-learn_stable)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, Tuple


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier on the scaled training data.

    Parameters
    ----------
    X_train : np.ndarray  — scaled training features
    y_train : np.ndarray  — training labels
    random_state : int    — random seed (default 42)
    max_iter : int        — maximum iterations for solver convergence

    Returns
    -------
    Fitted LogisticRegression model.
    """
    # Using sklearn LogisticRegression — Context7 confirmed
    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def train_knn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbours classifier (k=5) on the scaled training data.

    Parameters
    ----------
    X_train : np.ndarray  — scaled training features
    y_train : np.ndarray  — training labels
    n_neighbors : int     — number of neighbours (default 5 per methodology)

    Returns
    -------
    Fitted KNeighborsClassifier model.
    """
    # Using sklearn KNeighborsClassifier — Context7 confirmed
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a fitted classifier on the test set.

    Computes accuracy_score, confusion_matrix, and classification_report
    as required by methodology step 7.

    Parameters
    ----------
    model       : fitted sklearn classifier
    X_test      : np.ndarray — scaled test features
    y_test      : np.ndarray — true test labels
    target_names: list[str]  — class label names for the report
    model_name  : str        — display name for printing

    Returns
    -------
    dict with keys: 'accuracy', 'confusion_matrix', 'classification_report'
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Results")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


def run_classification_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full classification experiment: train both models and evaluate them.

    Implements methodology steps 5–7 for classification.

    Parameters
    ----------
    X_train, X_test : np.ndarray — scaled feature matrices
    y_train, y_test : np.ndarray — label vectors
    target_names    : list[str]  — class names

    Returns
    -------
    (lr_results, knn_results) — dicts with evaluation metrics for each model
    """
    # Step 5: Train models
    lr_model = train_logistic_regression(X_train, y_train)
    knn_model = train_knn_classifier(X_train, y_train, n_neighbors=5)

    # Steps 6–7: Predict and evaluate
    lr_results = evaluate_classifier(
        lr_model, X_test, y_test, target_names, "Logistic Regression"
    )
    knn_results = evaluate_classifier(
        knn_model, X_test, y_test, target_names, "KNeighborsClassifier (k=5)"
    )

    return lr_results, knn_results
