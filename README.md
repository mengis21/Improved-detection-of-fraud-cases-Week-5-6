# Improved Detection of Fraud Cases (Weeks 5-6)

This repository houses my solution for Adey Innovations' blended e-commerce and credit-card fraud detection challenge. The project spans two weeks and combines geolocation enrichment, aggressive feature engineering, and imbalanced-learning techniques across two public datasets.

## Repository Layout

```
.
├── data/               # Raw files (ignored) and processed artefacts
├── notebooks/          # Exploratory notebooks (documented, not yet tracked)
├── reports/            # Generated figures for interim/final reports
├── scripts/            # Reproducible data preparation and EDA routines
├── src/                # Reusable preprocessing/helpers
├── tests/              # To be populated in later milestones
└── Technical Content/  # Reference materials (ignored from git)
```

Key scripts:

- `scripts/run_task1_analysis.py` cleans both datasets, engineers Task 1 features, merges IP geolocation, and saves EDA charts in `reports/images/`.

## Interim-1 Scope (Task 1)

The current milestone focuses on:

1. **Data cleaning** – timestamp parsing, duplicate removal, type correction.
2. **EDA** – fraud versus legitimate behaviour across value, device, country, and signup timing.
3. **Geolocation merge** – IP to country lookup via range join.
4. **Feature engineering** – transaction velocity, recency, time-since-signup, time-of-day, device/browser usage.
5. **Class imbalance analysis** – baseline sampling ratios and SMOTE preview for downstream modeling.

Outputs are summarised in the accompanying interim report (not tracked in git) and illustrated via PNGs inside `reports/images/`.

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place the provided CSVs into `data/raw/` (see `scripts/README.md`).
3. Run the Task 1 pipeline:
   ```bash
   python scripts/run_task1_analysis.py
   ```

Processed tables land under `data/processed/` (ignored by git). Figures for documentation appear in `reports/images/`.

## Next Steps

- Extend notebooks for Task 2 modeling experiments.
- Stand up continuous integration with linting and unit tests once pipelines solidify.
- Add SHAP-based explainability narratives for the final submission.
