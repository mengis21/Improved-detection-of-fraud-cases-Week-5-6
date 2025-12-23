"""Utilities for converting and enriching IP address fields."""
from __future__ import annotations

import ipaddress
import pandas as pd


def ip_to_int(ip: str | int | None) -> int | None:
    """Convert dotted-quad IP strings to integer representation.

    Returns None when the value is missing or malformed.
    """
    if ip is None or (isinstance(ip, float) and pd.isna(ip)):
        return None
    if isinstance(ip, (int, pd.Int64Dtype)):
        return int(ip)
    try:
        return int(ipaddress.ip_address(str(ip)))
    except ValueError:
        return None


def prepare_ip_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lower/upper bound columns are numeric and sorted for asof merges."""
    df = df.copy()
    df["lower_bound_ip_int"] = pd.to_numeric(df["lower_bound_ip_address"].map(ip_to_int), errors="coerce")
    df["upper_bound_ip_int"] = pd.to_numeric(df["upper_bound_ip_address"].map(ip_to_int), errors="coerce")
    df = df.dropna(subset=["lower_bound_ip_int", "upper_bound_ip_int"])
    return df.sort_values("lower_bound_ip_int").reset_index(drop=True)


def enrich_with_country(transactions: pd.DataFrame, ip_lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach country metadata to each transaction via IP range lookup."""
    tx = transactions.copy()
    tx["ip_int"] = pd.to_numeric(tx["ip_address"].map(ip_to_int), errors="coerce").fillna(-1)

    lookup = prepare_ip_lookup(ip_lookup)
    enriched = pd.merge_asof(
        tx.sort_values("ip_int"),
        lookup[["lower_bound_ip_int", "upper_bound_ip_int", "country"]],
        left_on="ip_int",
        right_on="lower_bound_ip_int",
        direction="backward",
    )
    enriched.loc[
        (enriched["ip_int"].isna())
        | (enriched["ip_int"] > enriched["upper_bound_ip_int"]),
        "country",
    ] = "Unknown"
    enriched.loc[enriched["ip_int"] < 0, "country"] = "Unknown"
    return enriched
