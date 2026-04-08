"""
Results reporting utilities for the scikit-learn ML pipeline experiment.

Generates a Markdown summary table (results/RESULTS.md) with all
evaluation metrics from classification and regression experiments.
"""

import os
from datetime import datetime
from typing import Dict, Any, List


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _ensure_results_dir() -> str:
    path = os.path.abspath(RESULTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def build_results_markdown(
    clf_results: List[Dict[str, Any]],
    reg_results: List[Dict[str, Any]],
    clf_dataset: str = "Iris",
    reg_dataset: str = "California Housing",
) -> str:
    """
    Build a Markdown string summarising all experiment results.

    Parameters
    ----------
    clf_results  : list of classification result dicts
    reg_results  : list of regression result dicts
    clf_dataset  : name of the classification dataset
    reg_dataset  : name of the regression dataset

    Returns
    -------
    Markdown-formatted string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Experiment Results — Scikit-Learn ML Pipeline",
        "",
        f"**Generated:** {now}",
        "",
        "---",
        "",
        "## Experiment Overview",
        "",
        "This experiment follows the standard Scikit-Learn workflow:",
        "1. Load benchmark datasets (Iris for classification, California Housing for regression)",
        "2. Inspect dataset properties",
        "3. Split data 80/20 with `random_state=42`",
        "4. Apply `StandardScaler` (fit on training data only)",
        "5. Train and evaluate classification models: Logistic Regression vs. KNN (k=5)",
        "6. Train and evaluate regression models: Linear Regression vs. Ridge (alpha=1.0)",
        "",
        "---",
        "",
        f"## Classification Results — {clf_dataset} Dataset",
        "",
        "| Model | Accuracy |",
        "|-------|----------|",
    ]

    for r in clf_results:
        lines.append(f"| {r['model_name']} | {r['accuracy']:.4f} |")

    lines += [
        "",
        "### Confusion Matrices",
        "",
    ]
    for r in clf_results:
        lines.append(f"**{r['model_name']}**")
        lines.append("")
        lines.append("```")
        lines.append(str(r["confusion_matrix"]))
        lines.append("```")
        lines.append("")

    lines += [
        "### Classification Reports",
        "",
    ]
    for r in clf_results:
        lines.append(f"**{r['model_name']}**")
        lines.append("")
        lines.append("```")
        lines.append(r["classification_report"])
        lines.append("```")
        lines.append("")

    lines += [
        "---",
        "",
        f"## Regression Results — {reg_dataset} Dataset",
        "",
        "| Model | MSE | RMSE | R² |",
        "|-------|-----|------|----|",
    ]

    for r in reg_results:
        lines.append(
            f"| {r['model_name']} | {r['mse']:.4f} | {r['rmse']:.4f} | {r['r2']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate findings based on results
    best_clf = max(clf_results, key=lambda x: x["accuracy"])
    best_reg = max(reg_results, key=lambda x: x["r2"])

    lines += [
        f"- **Best classifier**: {best_clf['model_name']} "
        f"(Accuracy = {best_clf['accuracy']:.4f})",
        f"- **Best regressor**: {best_reg['model_name']} "
        f"(R² = {best_reg['r2']:.4f}, MSE = {best_reg['mse']:.4f})",
        "",
        "### Expected Outcome Validation",
        "",
    ]

    # Validate expected outcomes from the experiment spec
    clf_above_90 = all(r["accuracy"] >= 0.90 for r in clf_results)
    reg_above_055 = all(r["r2"] >= 0.55 for r in reg_results)
    reg_above_06 = all(r["r2"] >= 0.60 for r in reg_results)

    lines.append(
        f"- Classification accuracy ≥ 90% for all models: "
        f"{'✅ YES' if clf_above_90 else '❌ NO'}"
    )
    lines.append(
        f"- Regression R² ≥ 0.55 for all models: "
        f"{'✅ YES' if reg_above_055 else '❌ NO'}"
    )
    best_r2 = max(r["r2"] for r in reg_results)
    reg_06_note = (
        "✅ YES" if reg_above_06
        else f"⚠️  Actual R² ≈ {best_r2:.4f} (OLS on California Housing typically ~0.576)"
    )
    lines.append(
        f"- Regression R² ≥ 0.60 for all models (spec target): {reg_06_note}"
    )

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "",
        "| Plot | File |",
        "|------|------|",
        "| Confusion Matrices | `results/confusion_matrices.png` |",
        "| Regression Predictions | `results/regression_predictions.png` |",
        "| Model Comparison | `results/model_comparison.png` |",
        "| Feature Distributions (Classification) | `results/clf_feature_distributions.png` |",
        "| Feature Distributions (Regression) | `results/reg_feature_distributions.png` |",
        "",
    ]

    return "\n".join(lines)


def save_results_markdown(
    clf_results: List[Dict[str, Any]],
    reg_results: List[Dict[str, Any]],
    clf_dataset: str = "Iris",
    reg_dataset: str = "California Housing",
) -> str:
    """
    Build and save the RESULTS.md file to the results/ directory.

    Parameters
    ----------
    clf_results  : list of classification result dicts
    reg_results  : list of regression result dicts
    clf_dataset  : name of the classification dataset
    reg_dataset  : name of the regression dataset

    Returns
    -------
    Absolute path to the saved RESULTS.md file.
    """
    md_content = build_results_markdown(clf_results, reg_results, clf_dataset, reg_dataset)

    out_dir = _ensure_results_dir()
    out_path = os.path.join(out_dir, "RESULTS.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"  [Saved] {out_path}")
    return out_path
