# Improved Detection of Fraud Cases (Weeks 5-6)

This repository houses my solution for Adey Innovations' blended e-commerce and credit-card fraud detection challenge. The project spans Weeks 5-6 and combines geolocation enrichment, aggressive feature engineering, imbalanced-learning techniques, and SHAP explainability across two public datasets.

## Repository Layout

```
.
├── data/               # Raw files (ignored) and processed artefacts
├── notebooks/          # Exploratory notebooks
├── reports/            # Generated figures, metrics JSON, narrative drafts
├── scripts/            # Reproducible data preparation and EDA routines
├── src/                # Reusable preprocessing/helpers
├── tests/              # To be populated in later milestones
└── Technical Content/  # Reference materials (ignored from git)
```

Key scripts:

- `scripts/run_task1_analysis.py` cleans both datasets, engineers Task 1 features, merges IP geolocation, and saves EDA charts in `reports/images/`.
- `scripts/run_task2_modeling.py` trains stratified Logistic Regression and tuned Random Forest baselines for both datasets, logging test metrics and 5-fold cross-validation summaries to `reports/task2_metrics.json`.
- `scripts/run_task3_explainability.py` reloads the Task 2 best Random Forest models, produces feature-importance and SHAP diagnostics, and saves plots plus JSON summaries for reporting.

## Environment Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place the provided CSVs into `data/raw/` (see `scripts/README.md`).
    # Windows PowerShell
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

## Running the Pipelines

- **Task 1 – Data Preparation & EDA**
   ```bash
   python scripts/run_task1_analysis.py
   ```
   Outputs: processed tables under `data/processed/` and EDA visuals such as [reports/images/fraud_class_distribution.png](reports/images/fraud_class_distribution.png).

- **Task 2 – Modeling & Evaluation**
   ```bash
   python scripts/run_task2_modeling.py
   ```
   Outputs: evaluation bundle at [reports/task2_metrics.json](reports/task2_metrics.json) covering Logistic Regression and Random Forest results for both datasets.

- **Task 3 – Explainability**
   ```bash
   python scripts/run_task3_explainability.py
   ```
   Outputs: SHAP summaries and force plots under `reports/images/` plus the consolidated report [reports/task3_shap_summary.json](reports/task3_shap_summary.json).


## Deliverables Snapshot

- **Task 1**: Cleaning, geolocation enrichment, feature engineering, SMOTE preview. Visuals reside in `reports/images/`.
- **Task 2**: Stratified modeling with Logistic Regression baseline and tuned Random Forest, metrics logged to JSON for reproducible grading.
- **Task 3**: SHAP global/individual explanations and feature-importance comparisons for both datasets.

## Roadmap

- Expand test coverage and automation around the pipelines.
- Promote notebook experiments into version-controlled analyses.
- Integrate threshold tuning and cost-sensitive evaluation prior to production deployment.

### Thank You!