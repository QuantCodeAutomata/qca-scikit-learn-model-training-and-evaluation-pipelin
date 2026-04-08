"""
Regression pipeline for the scikit-learn ML experiment.

Implements methodology steps 8–9:
  - Train LinearRegression and Ridge (alpha=1.0)
  - Evaluate with mean_squared_error and r2_score

Using scikit-learn — Context7 confirmed (/websites/scikit-learn_stable)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """
    Train an Ordinary Least Squares LinearRegression model.

    Parameters
    ----------
    X_train : np.ndarray — scaled training features
    y_train : np.ndarray — training target values

    Returns
    -------
    Fitted LinearRegression model.
    """
    # Using sklearn LinearRegression — Context7 confirmed
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """
    Train a Ridge (L2-regularised) regression model.

    Parameters
    ----------
    X_train : np.ndarray — scaled training features
    y_train : np.ndarray — training target values
    alpha   : float      — regularisation strength (default 1.0 per methodology)

    Returns
    -------
    Fitted Ridge model.
    """
    # Using sklearn Ridge — Context7 confirmed
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a fitted regressor on the test set.

    Computes mean_squared_error and r2_score as required by methodology step 9.

    Parameters
    ----------
    model      : fitted sklearn regressor
    X_test     : np.ndarray — scaled test features
    y_test     : np.ndarray — true test target values
    model_name : str        — display name for printing

    Returns
    -------
    dict with keys: 'mse', 'rmse', 'r2'
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Regression Results")
    print(f"{'='*60}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {
        "model_name": model_name,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
    }


def run_regression_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full regression experiment: train both models and evaluate them.

    Implements methodology steps 8–9 for regression.

    Parameters
    ----------
    X_train, X_test : np.ndarray — scaled feature matrices
    y_train, y_test : np.ndarray — target vectors

    Returns
    -------
    (lr_results, ridge_results) — dicts with evaluation metrics for each model
    """
    # Step 8: Train models
    lr_model = train_linear_regression(X_train, y_train)
    ridge_model = train_ridge_regression(X_train, y_train, alpha=1.0)

    # Step 9: Evaluate
    lr_results = evaluate_regressor(lr_model, X_test, y_test, "Linear Regression")
    ridge_results = evaluate_regressor(ridge_model, X_test, y_test, "Ridge Regression (alpha=1.0)")

    return lr_results, ridge_results
