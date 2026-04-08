"""
Tests for src/reporting.py

Validates:
  - Markdown content generation
  - Expected sections are present
  - Outcome validation logic (≥90% accuracy, ≥0.60 R²)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reporting import build_results_markdown


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_clf_results(acc1: float = 0.97, acc2: float = 0.93):
    import numpy as np
    return [
        {
            "model_name": "Logistic Regression",
            "accuracy": acc1,
            "confusion_matrix": [[10, 0], [0, 20]],
            "classification_report": "precision recall f1-score\n...",
        },
        {
            "model_name": "KNeighborsClassifier (k=5)",
            "accuracy": acc2,
            "confusion_matrix": [[9, 1], [1, 19]],
            "classification_report": "precision recall f1-score\n...",
        },
    ]


def _make_reg_results(r2_1: float = 0.62, r2_2: float = 0.63):
    return [
        {
            "model_name": "Linear Regression",
            "mse": 0.55,
            "rmse": 0.74,
            "r2": r2_1,
        },
        {
            "model_name": "Ridge Regression (alpha=1.0)",
            "mse": 0.54,
            "rmse": 0.73,
            "r2": r2_2,
        },
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildResultsMarkdown:
    def test_returns_string(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert isinstance(md, str)

    def test_contains_classification_section(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert "Classification Results" in md

    def test_contains_regression_section(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert "Regression Results" in md

    def test_contains_model_names(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert "Logistic Regression" in md
        assert "KNeighborsClassifier" in md
        assert "Linear Regression" in md
        assert "Ridge Regression" in md

    def test_contains_accuracy_values(self):
        md = build_results_markdown(_make_clf_results(0.97, 0.93), _make_reg_results())
        assert "0.9700" in md
        assert "0.9300" in md

    def test_contains_r2_values(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results(0.62, 0.63))
        assert "0.6200" in md or "0.62" in md

    def test_outcome_validation_passes_when_above_thresholds(self):
        md = build_results_markdown(
            _make_clf_results(0.95, 0.92),
            _make_reg_results(0.65, 0.66),
        )
        assert "✅ YES" in md

    def test_outcome_validation_fails_when_below_clf_threshold(self):
        md = build_results_markdown(
            _make_clf_results(0.85, 0.88),  # below 90%
            _make_reg_results(0.65, 0.66),
        )
        # Classification check should fail
        assert "❌ NO" in md

    def test_outcome_validation_fails_when_below_reg_threshold(self):
        md = build_results_markdown(
            _make_clf_results(0.95, 0.92),
            _make_reg_results(0.50, 0.55),  # below 0.60
        )
        assert "❌ NO" in md

    def test_contains_key_findings_section(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert "Key Findings" in md

    def test_contains_best_classifier(self):
        md = build_results_markdown(_make_clf_results(0.97, 0.93), _make_reg_results())
        assert "Best classifier" in md
        assert "Logistic Regression" in md  # higher accuracy

    def test_contains_plots_section(self):
        md = build_results_markdown(_make_clf_results(), _make_reg_results())
        assert "confusion_matrices.png" in md
        assert "regression_predictions.png" in md
        assert "model_comparison.png" in md

    def test_custom_dataset_names_appear_in_output(self):
        md = build_results_markdown(
            _make_clf_results(),
            _make_reg_results(),
            clf_dataset="Breast Cancer",
            reg_dataset="Boston Housing",
        )
        assert "Breast Cancer" in md
        assert "Boston Housing" in md
