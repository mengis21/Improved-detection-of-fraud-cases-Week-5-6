"""Task 1 pipeline: cleaning, EDA, feature engineering, and imbalance diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE

from src.features import engineer_user_level_features, preprocess_transactions, summarize_class_balance
from src.ip_utils import enrich_with_country


sns.set_theme(style="whitegrid")


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fraud_path = raw_dir / "Fraud_Data.csv"
    ip_path = raw_dir / "IpAddress_to_Country.csv"
    credit_path = raw_dir / "creditcard.csv"
    if not fraud_path.exists() or not ip_path.exists() or not credit_path.exists():
        missing = [p.name for p in [fraud_path, ip_path, credit_path] if not p.exists()]
        raise FileNotFoundError(f"Missing raw files: {', '.join(missing)}")
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    credit_df = pd.read_csv(credit_path)
    return fraud_df, ip_df, credit_df


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_class_distribution(series: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, v in enumerate(series.values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_country_fraud(enriched: pd.DataFrame, path: Path) -> None:
    country_rates = (
        enriched.groupby("country")["class"].mean().sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=country_rates.values, y=country_rates.index, ax=ax, palette="Reds")
    ax.set_xlabel("Fraud Rate")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Countries by Fraud Rate")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_time_since_signup(df: pd.DataFrame, path: Path) -> None:
    data = df[["time_since_signup_hours", "class"]].dropna()
    if len(data) > 50000:
        data = data.sample(50000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=data,
        x="time_since_signup_hours",
        hue="class",
        common_norm=False,
        fill=True,
        ax=ax,
    )
    ax.set_xlim(left=0)
    ax.set_title("Time Since Signup vs Fraud")
    ax.set_xlabel("Hours from signup to purchase")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_credit_amount(df: pd.DataFrame, path: Path) -> None:
    data = df[["Class", "Amount"]]
    if len(data) > 200000:
        data = data.sample(200000, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Class", y="Amount", data=data, ax=ax)
    ax.set_title("Transaction Amount by Class (Credit Card)")
    ax.set_xlabel("Class (0=legit, 1=fraud)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def estimate_smote_distribution(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    if features.empty or target.empty:
        return pd.Series(dtype="float64", name="share")

    sample_features = features
    sample_target = target
    if len(features) > 50000:
        sample_idx = features.sample(50000, random_state=42).index
        sample_features = features.loc[sample_idx]
        sample_target = target.loc[sample_idx]

    smote = SMOTE(random_state=42)
    resampled_features, resampled_target = smote.fit_resample(sample_features, sample_target)
    return resampled_target.value_counts(normalize=True).sort_index().rename("share")


def write_processed(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe as CSV (parquet optional when engine available)."""
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)


def main(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> None:
    ensure_dirs(processed_dir, reports_dir / "images")
    fraud_raw, ip_lookup, credit_raw = load_raw_data(raw_dir)

    fraud_clean = preprocess_transactions(fraud_raw)
    fraud_geo = enrich_with_country(fraud_clean, ip_lookup)
    user_level = engineer_user_level_features(fraud_geo)

    write_processed(fraud_geo, processed_dir / "fraud_transactions_enriched")
    write_processed(user_level, processed_dir / "fraud_user_features")

    write_processed(credit_raw, processed_dir / "creditcard_raw")

    fraud_balance = summarize_class_balance(fraud_geo, "class")
    credit_balance = summarize_class_balance(credit_raw, "Class")

    save_class_distribution(
        fraud_balance,
        reports_dir / "images" / "fraud_class_distribution.png",
        "Fraud_Data Class Distribution",
    )
    save_class_distribution(
        credit_balance,
        reports_dir / "images" / "creditcard_class_distribution.png",
        "Credit Card Class Distribution",
    )
    plot_country_fraud(
        fraud_geo,
        reports_dir / "images" / "fraud_rate_by_country.png",
    )
    plot_time_since_signup(
        fraud_geo,
        reports_dir / "images" / "time_since_signup_density.png",
    )
    plot_credit_amount(
        credit_raw,
        reports_dir / "images" / "credit_amount_boxplot.png",
    )

    feature_cols = [
        "time_since_signup_hours",
        "hour_of_day",
        "purchase_value",
        "source",
        "device_id",
        "browser",
    ]
    fraud_subset = fraud_geo[feature_cols].copy()
    fraud_encoded = pd.get_dummies(
        fraud_subset,
        columns=["source", "device_id", "browser"],
        drop_first=True,
    )
    smote_distribution = estimate_smote_distribution(fraud_encoded, fraud_geo["class"])
    smote_distribution.to_csv(
        reports_dir / "smote_class_distribution.csv",
        header=True,
    )

    fraud_balance.to_csv(reports_dir / "fraud_class_distribution.csv", header=True)
    credit_balance.to_csv(reports_dir / "creditcard_class_distribution.csv", header=True)

    summary = {
        "fraud_shape": fraud_geo.shape,
        "credit_shape": credit_raw.shape,
        "user_features_shape": user_level.shape,
        "time_since_signup_missing_share": float(fraud_geo["time_since_signup_hours"].isna().mean()),
        "median_time_since_signup_by_class": fraud_geo.groupby("class")[
            "time_since_signup_hours"
        ].median().to_dict(),
        "fraud_class_distribution": fraud_balance.to_dict(),
        "credit_class_distribution": credit_balance.to_dict(),
        "smote_distribution": smote_distribution.to_dict(),
        "top_countries_by_fraud_rate": (
            fraud_geo.groupby("country")["class"].mean().sort_values(ascending=False).head(5).to_dict()
        ),
    }

    with open(reports_dir / "task1_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1 data preparation and EDA.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir, args.reports_dir)
