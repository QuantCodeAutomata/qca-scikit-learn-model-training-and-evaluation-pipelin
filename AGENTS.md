# Repository Knowledge — qca-scikit-learn-model-training-and-evaluation-pipeline

## Project Purpose
Scikit-Learn ML pipeline experiment: classification (Iris) and regression (California Housing).
Follows the DataCamp Scikit-Learn Cheat Sheet methodology exactly.

## Structure
- `experiment.py` — main runner, executes all 10 methodology steps
- `src/data_loader.py` — dataset loading, inspection, split+scale
- `src/classification_pipeline.py` — LogisticRegression + KNN training/evaluation
- `src/regression_pipeline.py` — LinearRegression + Ridge training/evaluation
- `src/visualization.py` — matplotlib/seaborn plots saved to results/
- `src/reporting.py` — generates results/RESULTS.md
- `tests/` — pytest test suite (unit + integration)
- `results/` — all output files (plots + RESULTS.md)

## Key Parameters (from methodology)
- test_size=0.2, random_state=42 (all splits)
- KNN: n_neighbors=5
- Ridge: alpha=1.0
- LogisticRegression: max_iter=1000 (for convergence)

## Library Choices (Context7 verified)
- All ML: scikit-learn (/websites/scikit-learn_stable)
- Plots: matplotlib + seaborn
- No external financial data (no massive API needed)

## Expected Outcomes
- Classification accuracy ≥ 90% on Iris for both models
- Regression R² ≥ 0.60 on California Housing for both models

## Running
```bash
python experiment.py        # full experiment
pytest tests/ -v            # all tests
```
