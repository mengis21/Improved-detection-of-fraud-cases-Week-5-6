"""Feature engineering helpers for Fraud_Data preprocessing."""
from __future__ import annotations

import pandas as pd


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean base fraud transaction table."""
    tx = df.copy()
    tx["signup_time"] = pd.to_datetime(tx["signup_time"], utc=True)
    tx["purchase_time"] = pd.to_datetime(tx["purchase_time"], utc=True)
    tx = tx.drop_duplicates()
    # Align categorical string cases
    string_cols = ["source", "browser", "sex"]
    for col in string_cols:
        tx[col] = tx[col].astype("string").str.strip().str.lower()
    tx["device_id"] = tx["device_id"].astype("string").str.lower()
    tx["time_since_signup_hours"] = (
        (tx["purchase_time"] - tx["signup_time"]).dt.total_seconds() / 3600.0
    )
    tx["hour_of_day"] = tx["purchase_time"].dt.hour
    tx["day_of_week"] = tx["purchase_time"].dt.day_name()
    tx["purchase_day"] = tx["purchase_time"].dt.date
    return tx


def engineer_user_level_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to user-level feature table."""
    tx = transactions.copy()
    agg = (
        tx.sort_values("purchase_time")
        .groupby("user_id")
        .agg(
            signup_time=("signup_time", "min"),
            first_purchase=("purchase_time", "min"),
            last_purchase=("purchase_time", "max"),
            purchases=("purchase_time", "count"),
            avg_purchase_value=("purchase_value", "mean"),
            median_purchase_value=("purchase_value", "median"),
            total_purchase_value=("purchase_value", "sum"),
            avg_time_since_signup_hours=("time_since_signup_hours", "mean"),
            share_mobile=("source", lambda x: (x == "mobile").mean()),
            share_ads=("source", lambda x: (x == "ads").mean()),
            share_seo=("source", lambda x: (x == "seo").mean()),
            unique_devices=("device_id", pd.Series.nunique),
            unique_browsers=("browser", pd.Series.nunique),
            avg_hour=("hour_of_day", "mean"),
        )
    )
    agg["active_days"] = (
        (agg["last_purchase"].dt.floor("d") - agg["first_purchase"].dt.floor("d")).dt.days
    ).clip(lower=1)
    agg["transactions_per_day"] = agg["purchases"] / agg["active_days"]
    agg["velocity_24h"] = (
        tx.groupby("user_id")["purchase_time"]
        .diff()
        .dt.total_seconds()
        .lt(24 * 3600)
        .groupby(tx["user_id"])
        .mean()
        .fillna(0.0)
    )
    return agg.reset_index()


def summarize_class_balance(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Return percentage distribution for target column."""
    counts = df[target_col].value_counts().sort_index()
    return (counts / counts.sum()).rename("share")
