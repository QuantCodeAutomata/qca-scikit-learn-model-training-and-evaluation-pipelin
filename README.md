# Scikit-Learn Model Training and Evaluation Pipeline

A complete machine learning experiment implementing the standard Scikit-Learn workflow for training, evaluating, and comparing classification and regression models on benchmark datasets.

## Experiment Overview

This experiment follows the methodology from the DataCamp Scikit-Learn Cheat Sheet:

1. **Classification** (Iris dataset): Logistic Regression vs. KNeighborsClassifier (k=5)
2. **Regression** (California Housing dataset): LinearRegression vs. Ridge (alpha=1.0)

All models use `StandardScaler` preprocessing (fit on training data only) and an 80/20 train/test split with `random_state=42`.

## Project Structure

```
.
├── experiment.py              # Main experiment runner
├── src/
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── classification_pipeline.py  # Classification models and evaluation
│   ├── regression_pipeline.py      # Regression models and evaluation
│   ├── visualization.py       # Plot generation
│   └── reporting.py           # RESULTS.md generation
├── tests/
│   ├── test_data_loader.py
│   ├── test_classification_pipeline.py
│   ├── test_regression_pipeline.py
│   ├── test_reporting.py
│   └── test_experiment_integration.py
├── results/
│   ├── RESULTS.md             # Experiment results summary
│   ├── confusion_matrices.png
│   ├── regression_predictions.png
│   ├── model_comparison.png
│   ├── clf_feature_distributions.png
│   └── reg_feature_distributions.png
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Experiment

```bash
python experiment.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Expected Results

| Task | Model | Metric | Expected |
|------|-------|--------|----------|
| Classification | Logistic Regression | Accuracy | ≥ 90% |
| Classification | KNN (k=5) | Accuracy | ≥ 90% |
| Regression | Linear Regression | R² | ≥ 0.60 |
| Regression | Ridge (α=1.0) | R² | ≥ 0.60 |

## Key Design Decisions

- **No data leakage**: `StandardScaler` is fit exclusively on training data
- **Reproducibility**: `random_state=42` used throughout
- **Library usage**: All models use scikit-learn directly (Context7 confirmed)
