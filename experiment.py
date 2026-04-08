"""
Main experiment runner for the Scikit-Learn Model Training and Evaluation Pipeline.

Follows the methodology exactly as described in exp_1:
  Steps 1–7  : Classification experiment (Iris dataset)
  Steps 8–9  : Regression experiment (California Housing dataset)
  Step 10    : Summarise all results in a comparison table

Usage:
    python experiment.py
"""

import os
import sys

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import (
    load_classification_dataset,
    load_regression_dataset,
    inspect_dataset,
    split_and_scale,
)
from classification_pipeline import run_classification_experiment
from regression_pipeline import run_regression_experiment
from visualization import (
    plot_confusion_matrices,
    plot_regression_predictions,
    plot_model_comparison,
    plot_feature_distributions,
)
from reporting import save_results_markdown


def run_classification_section() -> tuple:
    """
    Execute methodology steps 1–7 for the classification experiment.

    Returns
    -------
    (lr_results, knn_results, target_names)
    """
    print("\n" + "=" * 70)
    print("  CLASSIFICATION EXPERIMENT — Iris Dataset")
    print("=" * 70)

    # Step 1: Load dataset
    print("\n[Step 1] Loading Iris classification dataset...")
    dataset = load_classification_dataset("iris")

    # Step 2: Inspect dataset
    print("\n[Step 2] Dataset inspection:")
    inspect_dataset(dataset, task="classification")

    X, y = dataset["X"], dataset["y"]
    feature_names = dataset["feature_names"]
    target_names = dataset["target_names"]

    # Step 3 & 4: Split and scale
    print("\n[Steps 3–4] Splitting (80/20, random_state=42) and scaling with StandardScaler...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train size: {X_train.shape[0]} samples | Test size: {X_test.shape[0]} samples")

    # Steps 5–7: Train and evaluate
    print("\n[Steps 5–7] Training and evaluating classifiers...")
    lr_results, knn_results = run_classification_experiment(
        X_train, X_test, y_train, y_test, list(target_names)
    )

    # Visualise feature distributions
    print("\n[Viz] Plotting feature distributions...")
    plot_feature_distributions(
        X, list(feature_names), "Iris", "clf_feature_distributions.png"
    )

    return lr_results, knn_results, list(target_names), X, list(feature_names)


def run_regression_section() -> tuple:
    """
    Execute methodology steps 8–9 for the regression experiment.

    Returns
    -------
    (lr_results, ridge_results, y_test)
    """
    print("\n" + "=" * 70)
    print("  REGRESSION EXPERIMENT — California Housing Dataset")
    print("=" * 70)

    # Step 8a: Load dataset
    print("\n[Step 8] Loading California Housing regression dataset...")
    dataset = load_regression_dataset("california_housing")

    # Inspect
    print("\n[Step 8] Dataset inspection:")
    inspect_dataset(dataset, task="regression")

    X, y = dataset["X"], dataset["y"]
    feature_names = dataset["feature_names"]

    # Split and scale
    print("\n[Steps 8] Splitting (80/20, random_state=42) and scaling with StandardScaler...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train size: {X_train.shape[0]} samples | Test size: {X_test.shape[0]} samples")

    # Step 8b–9: Train and evaluate
    print("\n[Steps 8–9] Training and evaluating regressors...")
    lr_results, ridge_results = run_regression_experiment(
        X_train, X_test, y_train, y_test
    )

    # Visualise feature distributions
    print("\n[Viz] Plotting feature distributions...")
    plot_feature_distributions(
        X, list(feature_names), "California Housing", "reg_feature_distributions.png"
    )

    return lr_results, ridge_results, y_test


def print_comparison_table(
    clf_results: list,
    reg_results: list,
) -> None:
    """
    Step 10: Print a formatted comparison table of all model results.

    Parameters
    ----------
    clf_results : list of classification result dicts
    reg_results : list of regression result dicts
    """
    print("\n" + "=" * 70)
    print("  STEP 10 — RESULTS COMPARISON TABLE")
    print("=" * 70)

    print("\n  Classification (Iris Dataset)")
    print(f"  {'Model':<40} {'Accuracy':>10}")
    print("  " + "-" * 52)
    for r in clf_results:
        print(f"  {r['model_name']:<40} {r['accuracy']:>10.4f}")

    print("\n  Regression (California Housing Dataset)")
    print(f"  {'Model':<40} {'MSE':>10} {'RMSE':>10} {'R²':>10}")
    print("  " + "-" * 72)
    for r in reg_results:
        print(
            f"  {r['model_name']:<40} {r['mse']:>10.4f} "
            f"{r['rmse']:>10.4f} {r['r2']:>10.4f}"
        )
    print()


def main() -> None:
    """
    Orchestrate the full experiment pipeline end-to-end.
    """
    print("\n" + "#" * 70)
    print("  Scikit-Learn Model Training and Evaluation Pipeline — Experiment")
    print("#" * 70)

    # ── Classification ──────────────────────────────────────────────────────
    lr_clf, knn_clf, target_names, X_clf, feature_names_clf = run_classification_section()

    # ── Regression ──────────────────────────────────────────────────────────
    lr_reg, ridge_reg, y_test_reg = run_regression_section()

    # ── Visualisations ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING VISUALISATIONS")
    print("=" * 70)

    plot_confusion_matrices(
        [lr_clf, knn_clf],
        target_names,
        dataset_name="Iris",
        filename="confusion_matrices.png",
    )

    plot_regression_predictions(
        [lr_reg, ridge_reg],
        y_test_reg,
        dataset_name="California Housing",
        filename="regression_predictions.png",
    )

    plot_model_comparison(
        [lr_clf, knn_clf],
        [lr_reg, ridge_reg],
        filename="model_comparison.png",
    )

    # ── Step 10: Comparison table ────────────────────────────────────────────
    print_comparison_table([lr_clf, knn_clf], [lr_reg, ridge_reg])

    # ── Save RESULTS.md ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)
    save_results_markdown(
        [lr_clf, knn_clf],
        [lr_reg, ridge_reg],
        clf_dataset="Iris",
        reg_dataset="California Housing",
    )

    print("\n✅  Experiment complete. All results saved to results/")


if __name__ == "__main__":
    main()
