# Scripts

- `run_task1_analysis.py`: end-to-end pipeline that cleans the raw datasets, engineers task-one features, performs exploratory statistics, and writes figures to `reports/images/` and processed tables to `data/processed/`.

Run with:

```bash
python scripts/run_task1_analysis.py --raw-dir data/raw --processed-dir data/processed --reports-dir reports
```

The script expects the raw CSV files inside `data/raw/`:

- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`
