# Experiment Results — Scikit-Learn ML Pipeline

**Generated:** 2026-04-08 11:46:56

---

## Experiment Overview

This experiment follows the standard Scikit-Learn workflow:
1. Load benchmark datasets (Iris for classification, California Housing for regression)
2. Inspect dataset properties
3. Split data 80/20 with `random_state=42`
4. Apply `StandardScaler` (fit on training data only)
5. Train and evaluate classification models: Logistic Regression vs. KNN (k=5)
6. Train and evaluate regression models: Linear Regression vs. Ridge (alpha=1.0)

---

## Classification Results — Iris Dataset

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 1.0000 |
| KNeighborsClassifier (k=5) | 1.0000 |

### Confusion Matrices

**Logistic Regression**

```
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

**KNeighborsClassifier (k=5)**

```
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

### Classification Reports

**Logistic Regression**

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

```

**KNeighborsClassifier (k=5)**

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

```

---

## Regression Results — California Housing Dataset

| Model | MSE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.5559 | 0.7456 | 0.5758 |
| Ridge Regression (alpha=1.0) | 0.5559 | 0.7456 | 0.5758 |

---

## Key Findings

- **Best classifier**: Logistic Regression (Accuracy = 1.0000)
- **Best regressor**: Ridge Regression (alpha=1.0) (R² = 0.5758, MSE = 0.5559)

### Expected Outcome Validation

- Classification accuracy ≥ 90% for all models: ✅ YES
- Regression R² ≥ 0.55 for all models: ✅ YES
- Regression R² ≥ 0.60 for all models (spec target): ⚠️  Actual R² ≈ 0.5758 (OLS on California Housing typically ~0.576)

---

## Plots

| Plot | File |
|------|------|
| Confusion Matrices | `results/confusion_matrices.png` |
| Regression Predictions | `results/regression_predictions.png` |
| Model Comparison | `results/model_comparison.png` |
| Feature Distributions (Classification) | `results/clf_feature_distributions.png` |
| Feature Distributions (Regression) | `results/reg_feature_distributions.png` |
