# Scikit-Learn Model Training and Evaluation Pipeline

A comprehensive scikit-learn experiment suite covering supervised learning pipelines, hyperparameter tuning, dimensionality reduction, and unsupervised clustering.

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Model Training and Evaluation Pipeline (KNN, SVC, LinearRegression) |
| exp_2 | Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV |
| exp_3 | Dimensionality Reduction with PCA Followed by Classification |
| exp_4 | Unsupervised Clustering with KMeans and Cluster Quality Evaluation |

## Repository Structure

```
.
├── main.py                          # Main runner — executes all experiments
├── requirements.txt
├── src/
│   ├── exp1_pipeline.py             # Exp 1: Supervised learning pipeline
│   ├── exp2_hyperparameter_tuning.py # Exp 2: GridSearchCV vs RandomizedSearchCV
│   ├── exp3_pca_classification.py   # Exp 3: PCA + KNN classification
│   ├── exp4_kmeans_clustering.py    # Exp 4: KMeans clustering
│   └── visualizations.py           # Shared plotting utilities
├── tests/
│   ├── conftest.py
│   ├── test_exp1_pipeline.py
│   ├── test_exp2_hyperparameter_tuning.py
│   ├── test_exp3_pca_classification.py
│   └── test_exp4_kmeans_clustering.py
└── results/
    ├── RESULTS.md                   # All metrics in Markdown
    └── *.png                        # Generated plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Run All Experiments

```bash
python main.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Expected Outcomes

- **Exp 1**: KNN and SVC accuracy > 90% on Iris; LinearRegression R² > 0.75 on synthetic data
- **Exp 2**: Both search strategies achieve > 90% test accuracy; GridSearch explores all 12 combinations
- **Exp 3**: PCA reduces Digits from 64 → ~29 features with < 2% accuracy drop
- **Exp 4**: Optimal k=3 identified; Adjusted Rand Index > 0.7
