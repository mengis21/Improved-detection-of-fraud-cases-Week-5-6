"""Task 2 pipeline: model training and evaluation for fraud detection datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
MAX_ROWS_FRAUD = 60_000
MAX_ROWS_CREDIT = 60_000


@dataclass
class MetricSummary:
    f1: float
    average_precision: float
    confusion_matrix: list[list[int]]
    positive_support: int
    negative_support: int


@dataclass
class CrossValidationSummary:
    f1_mean: float
    f1_std: float
    average_precision_mean: float
    average_precision_std: float


@dataclass
class ModelResult:
    test: MetricSummary
    cross_validation: CrossValidationSummary
    best_params: Dict[str, Any] | None = None


def downsample_for_training(
    X: pd.DataFrame, y: pd.Series, max_rows: int
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_sample, y_sample


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))
    return ColumnTransformer(transformers)


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> MetricSummary:
    cm = confusion_matrix(y_true, y_pred)
    return MetricSummary(
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        average_precision=float(average_precision_score(y_true, y_proba)),
        confusion_matrix=cm.tolist(),
        positive_support=int(y_true.sum()),
        negative_support=int((y_true == 0).sum()),
    )


def summarize_cv(cv_results: dict[str, np.ndarray]) -> CrossValidationSummary:
    return CrossValidationSummary(
        f1_mean=float(cv_results["test_f1"].mean()),
        f1_std=float(cv_results["test_f1"].std()),
        average_precision_mean=float(cv_results["test_average_precision"].mean()),
        average_precision_std=float(cv_results["test_average_precision"].std()),
    )


def run_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    dataset_label: str,
) -> ModelResult:
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cv = cross_validate(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring={"f1": "f1", "average_precision": "average_precision"},
        n_jobs=-1,
        error_score="raise",
    )

    return ModelResult(
        test=summarize_metrics(y_test.values, y_pred, y_proba),
        cross_validation=summarize_cv(cv),
    )


def run_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    dataset_label: str,
) -> ModelResult:
    param_options = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 3},
    ]
    best_score = -np.inf
    best_params = None
    best_cv_results = None
    best_pipeline = None

    for params in param_options:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **params,
                    ),
                ),
            ]
        )

        cv = cross_validate(
            pipeline,
            X,
            y,
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            scoring={"f1": "f1", "average_precision": "average_precision"},
            n_jobs=-1,
            error_score="raise",
        )

        avg_prec = cv["test_average_precision"].mean()
        if avg_prec > best_score:
            best_score = avg_prec
            best_params = params
            best_cv_results = cv
            best_pipeline = pipeline

    if best_pipeline is None or best_cv_results is None:
        raise RuntimeError(f"Failed to evaluate Random Forest for {dataset_label}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    return ModelResult(
        test=summarize_metrics(y_test.values, y_pred, y_proba),
        cross_validation=summarize_cv(best_cv_results),
        best_params=best_params,
    )


def prepare_fraud_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = pd.read_csv(path)
    feature_cols = [
        "purchase_value",
        "age",
        "time_since_signup_hours",
        "hour_of_day",
        "day_of_week",
        "country",
        "source",
        "browser",
        "sex",
    ]
    df = df.dropna(subset=feature_cols + ["class"])
    X = df[feature_cols]
    y = df["class"].astype(int)
    X, y = downsample_for_training(X, y, MAX_ROWS_FRAUD)
    numeric_features = ["purchase_value", "age", "time_since_signup_hours", "hour_of_day"]
    categorical_features = ["day_of_week", "country", "source", "browser", "sex"]
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return X, y, preprocessor


def prepare_credit_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    X, y = downsample_for_training(X, y, MAX_ROWS_CREDIT)
    numeric_features = list(X.columns)
    preprocessor = build_preprocessor(numeric_features, [])
    return X, y, preprocessor


def main(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    fraud_path = processed_dir / "fraud_transactions_enriched.csv"
    credit_path = processed_dir / "creditcard_raw.csv"
    if not fraud_path.exists() or not credit_path.exists():
        missing = [str(p) for p in [fraud_path, credit_path] if not p.exists()]
        raise FileNotFoundError(
            "Missing processed datasets. Run scripts/run_task1_analysis.py first. Missing: "
            + ", ".join(missing)
        )

    fraud_X, fraud_y, fraud_preprocessor = prepare_fraud_dataset(fraud_path)
    credit_X, credit_y, credit_preprocessor = prepare_credit_dataset(credit_path)

    results: dict[str, dict[str, Any]] = {}

    fraud_log_reg = run_logistic_regression(fraud_X, fraud_y, fraud_preprocessor, "Fraud_Data")
    fraud_rf = run_random_forest(fraud_X, fraud_y, fraud_preprocessor, "Fraud_Data")
    results["fraud_data"] = {
        "logistic_regression": asdict(fraud_log_reg),
        "random_forest": asdict(fraud_rf),
    }

    credit_log_reg = run_logistic_regression(credit_X, credit_y, credit_preprocessor, "creditcard")
    credit_rf = run_random_forest(credit_X, credit_y, credit_preprocessor, "creditcard")
    results["creditcard"] = {
        "logistic_regression": asdict(credit_log_reg),
        "random_forest": asdict(credit_rf),
    }

    output_path = reports_dir / "task2_metrics.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Task 2 modeling completed. Metrics saved to {output_path}.")


if __name__ == "__main__":
    project_root = Path.cwd()
    main(
        raw_dir=project_root / "data" / "raw",
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "reports",
    )
