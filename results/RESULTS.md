# Experiment Results

## Experiment 1 — Model Training and Evaluation Pipeline

### Classification (Iris Dataset)

| Model | Accuracy |
|-------|----------|
| KNeighborsClassifier (n_neighbors=5) | 1.0000 |
| SVC | 1.0000 |

### Regression (Synthetic Dataset)

| Metric | Value |
|--------|-------|
| MAE | 27.8257 |
| MSE | 1364.2908 |
| R² | 0.9337 |

---

## Experiment 2 — Hyperparameter Tuning

| Method | Best Params | Best CV Score | Test Accuracy | Runtime (s) |
|--------|-------------|---------------|---------------|-------------|
| GridSearchCV | {'metric': 'euclidean', 'n_neighbors': 1} | 0.9524 | 1.0000 | 0.1635 |
| RandomizedSearchCV | {'n_neighbors': 1, 'metric': 'euclidean'} | 0.9524 | 1.0000 | 0.1376 |

---

## Experiment 3 — PCA + Classification

| Dataset | Original Features | PCA Components (95%) | Acc (No PCA) | Acc (PCA) | Acc Drop |
|---------|-------------------|----------------------|--------------|-----------|----------|
| iris | 4 | 2 | 1.0000 | 1.0000 | 0.0000 |
| digits | 64 | 40 | 0.9759 | 0.9778 | -0.0019 |

---

## Experiment 4 — KMeans Clustering

**Optimal k:** 3

**Adjusted Rand Index:** 0.6201

**Final Silhouette Score:** 0.4599

**Final Inertia:** 139.8205


### Metrics per k

| k | Inertia | Silhouette |
|---|---------|------------|
| 2 | 222.3617 | 0.5818 |
| 3 | 139.8205 | 0.4599 |
| 4 | 114.0925 | 0.3869 |
| 5 | 90.9275 | 0.3459 |
| 6 | 81.5444 | 0.3171 |
| 7 | 72.6311 | 0.3202 |
| 8 | 62.5406 | 0.3387 |
| 9 | 55.1195 | 0.3424 |
| 10 | 47.3910 | 0.3518 |

---

## Plots

- `exp1_accuracy.png` — Classification accuracy bar chart
- `exp1_cm_knn.png` — KNN confusion matrix
- `exp1_cm_svc.png` — SVC confusion matrix
- `exp2_comparison.png` — GridSearch vs RandomizedSearch comparison
- `exp3_cumvar_iris.png` — PCA cumulative variance (Iris)
- `exp3_cumvar_digits.png` — PCA cumulative variance (Digits)
- `exp3_scatter_iris.png` — PCA 2D scatter (Iris)
- `exp3_scatter_digits.png` — PCA 2D scatter (Digits)
- `exp3_accuracy_comparison.png` — PCA vs No-PCA accuracy
- `exp4_elbow_silhouette.png` — Elbow + silhouette plots
- `exp4_pca_clusters.png` — PCA cluster scatter