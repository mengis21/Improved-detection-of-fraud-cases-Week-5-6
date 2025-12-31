"""Task 3 pipeline: SHAP-based explainability for fraud detection models."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MAX_ROWS_FRAUD = 60_000
MAX_ROWS_CREDIT = 60_000
SHAP_TOP_FEATURES = 10
SHAP_TOP_CONTRIBUTORS = 5
SHAP_MAX_SAMPLES = 4000

FRAUD_RF_PARAMS = {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 3}
CREDIT_RF_PARAMS = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1}


@dataclass
class SampleExplanation:
    prediction: int
    probability: float
    actual: int
    top_contributors: List[Dict[str, float]]


@dataclass
class DatasetExplainabilitySummary:
    f1: float
    average_precision: float
    feature_importance: List[Dict[str, float]]
    shap_importance: List[Dict[str, float]]
    examples: Dict[str, SampleExplanation]


def downsample_for_training(
    X: pd.DataFrame, y: pd.Series, max_rows: int
) -> Tuple[pd.DataFrame, pd.Series]:
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


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))
    return ColumnTransformer(transformers)


def prepare_fraud_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    if not path.exists():
        raise FileNotFoundError(f"Missing fraud dataset at {path}")
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


def prepare_credit_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    if not path.exists():
        raise FileNotFoundError(f"Missing credit dataset at {path}")
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    X, y = downsample_for_training(X, y, MAX_ROWS_CREDIT)
    numeric_features = list(X.columns)
    preprocessor = build_preprocessor(numeric_features, [])
    return X, y, preprocessor


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    params: Dict[str, object],
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
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
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return pipeline, X_train, X_test, y_test, y_pred, y_proba


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_proba: Iterable[float]) -> Tuple[float, float]:
    return (
        float(f1_score(y_true, y_pred, zero_division=0)),
        float(average_precision_score(y_true, y_proba)),
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        feature_names: List[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        return feature_names


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, title: str, output_path: Path) -> None:
    order = np.argsort(importances)[::-1][:SHAP_TOP_FEATURES]
    top_features = [feature_names[i] for i in order]
    top_importances = importances[order]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, orient="h", palette="viridis")
    plt.title(title)
    plt.xlabel("Gini importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_shap_summary(shap_values: np.ndarray, data: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, data, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_shap_bar(shap_values: np.ndarray, feature_names: List[str], title: str, output_path: Path) -> None:
    abs_vals = np.abs(shap_values).mean(axis=0)
    order = np.argsort(abs_vals)[::-1][:SHAP_TOP_FEATURES]
    top_features = [feature_names[i] for i in order]
    top_importance = abs_vals[order]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importance, y=top_features, orient="h", palette="magma")
    plt.title(title)
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_force_plot(
    base_value: float,
    shap_row: np.ndarray,
    feature_row: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    shap.force_plot(
        base_value,
        shap_row,
        feature_row,
        matplotlib=True,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def extract_top_contributors(
    shap_row: np.ndarray, feature_row: pd.Series, feature_names: List[str], top_k: int
) -> List[Dict[str, float]]:
    order = np.argsort(np.abs(shap_row))[::-1][:top_k]
    contributors: List[Dict[str, float]] = []
    for idx in order:
        contributors.append(
            {
                "feature": feature_names[idx],
                "value": float(feature_row.iloc[idx]),
                "shap_value": float(shap_row[idx]),
            }
        )
    return contributors


def pick_example_indices(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    rng = np.random.default_rng(RANDOM_STATE)

    if len(tp) > 0:
        indices["true_positive"] = int(tp[0])
    else:
        indices["true_positive"] = int(np.argmax(y_proba))

    if len(fp) > 0:
        indices["false_positive"] = int(fp[0])
    else:
        negative_mask = np.where(y_true == 0)[0]
        if len(negative_mask) == 0:
            indices["false_positive"] = indices["true_positive"]
        else:
            candidates = negative_mask[np.argsort(y_proba[negative_mask])[::-1]]
            indices["false_positive"] = int(candidates[0])

    if len(fn) > 0:
        indices["false_negative"] = int(fn[0])
    else:
        positive_mask = np.where(y_true == 1)[0]
        if len(positive_mask) == 0:
            indices["false_negative"] = indices["true_positive"]
        else:
            candidates = positive_mask[np.argsort(y_proba[positive_mask])]
            indices["false_negative"] = int(candidates[0])

    return indices


def summarize_dataset(
    dataset_label: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    images_dir: Path,
) -> DatasetExplainabilitySummary:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if isinstance(X_test_transformed, np.ndarray):
        X_test_frame = pd.DataFrame(X_test_transformed, columns=feature_names)
    else:
        X_test_frame = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)

    background = X_train_transformed
    if hasattr(background, "toarray"):
        background = background.toarray()
    if isinstance(background, np.ndarray) and background.shape[0] > 2000:
        background = background[:2000]

    indices = pick_example_indices(y_test.values, y_pred, y_proba)

    shap_indices = set(indices.values())
    if len(X_test_frame) <= SHAP_MAX_SAMPLES:
        shap_indices.update(range(len(X_test_frame)))
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        additional = [
            idx
            for idx in rng.permutation(len(X_test_frame))
            if idx not in shap_indices
        ][: max(SHAP_MAX_SAMPLES - len(shap_indices), 0)]
        shap_indices.update(additional)
    shap_indices = sorted(shap_indices)

    X_shap = X_test_frame.iloc[shap_indices]

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_names=feature_names,
        model_output="probability",
    )
    try:
        shap_values_full = explainer.shap_values(
            X_shap,
            check_additivity=False,
            approximate=True,
        )
    except shap.utils._exceptions.ExplainerError:
        # SHAP occasionally raises additivity errors for tree ensembles due to numerical drift;
        # rerun without supplying the background dataset to relax constraints further.
        shap_values_full = explainer.shap_values(
            X_shap.to_numpy(),
            check_additivity=False,
            approximate=True,
        )
    if isinstance(shap_values_full, shap.Explanation):
        shap_values = shap_values_full.values
        base_values = np.array(shap_values_full.base_values)
    elif isinstance(shap_values_full, list):
        shap_component = shap_values_full[1]
        if isinstance(shap_component, shap.Explanation):
            shap_values = shap_component.values
            base_values = np.array(shap_component.base_values)
        else:
            shap_values = shap_component
            base_values = np.full(shap_values.shape[0], explainer.expected_value[1])
    else:
        shap_values = shap_values_full
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            positive_base = float(expected[1]) if len(expected) > 1 else float(expected[0])
        else:
            positive_base = float(expected)
        base_values = np.full(shap_values.shape[0], positive_base)

    shap_values = np.asarray(shap_values, dtype=float)
    base_values = np.asarray(base_values, dtype=float)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    if base_values.ndim > 1:
        base_values = base_values[..., -1]

    dataset_slug = dataset_label.lower().replace(" ", "_")

    plot_feature_importance(
        feature_names,
        model.feature_importances_,
        f"{dataset_label} – Random Forest Feature Importance",
        images_dir / f"{dataset_slug}_feature_importance.png",
    )

    plot_shap_summary(
        shap_values,
        X_shap,
        f"{dataset_label} – SHAP Summary",
        images_dir / f"{dataset_slug}_shap_summary.png",
    )

    plot_shap_bar(
        shap_values,
        feature_names,
        f"{dataset_label} – Mean |SHAP|",
        images_dir / f"{dataset_slug}_shap_bar.png",
    )

    shap_lookup = {orig_idx: pos for pos, orig_idx in enumerate(shap_indices)}

    example_outputs: Dict[str, SampleExplanation] = {}
    for label, idx in indices.items():
        shap_pos = shap_lookup[idx]
        shap_row = shap_values[shap_pos]
        feature_row = X_test_frame.iloc[idx]
        plot_title = f"{dataset_label} – {label.replace('_', ' ').title()}"
        plot_path = images_dir / f"{dataset_slug}_{label}_force.png"
        save_force_plot(base_values[shap_pos], shap_row, feature_row, plot_title, plot_path)
        contributors = extract_top_contributors(shap_row, feature_row, feature_names, SHAP_TOP_CONTRIBUTORS)
        example_outputs[label] = SampleExplanation(
            prediction=int(y_pred[idx]),
            probability=float(y_proba[idx]),
            actual=int(y_test.iloc[idx]),
            top_contributors=contributors,
        )

    f1, avg_precision = compute_classification_metrics(y_test, y_pred, y_proba)

    importance_order = np.argsort(model.feature_importances_)[::-1][:SHAP_TOP_FEATURES]
    feature_importance_summary = [
        {
            "feature": feature_names[i],
            "importance": float(model.feature_importances_[i]),
        }
        for i in importance_order
    ]

    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_order = np.argsort(shap_abs)[::-1][:SHAP_TOP_FEATURES]
    shap_importance_summary = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": float(shap_abs[i]),
        }
        for i in shap_order
    ]

    return DatasetExplainabilitySummary(
        f1=f1,
        average_precision=avg_precision,
        feature_importance=feature_importance_summary,
        shap_importance=shap_importance_summary,
        examples=example_outputs,
    )


def main(processed_dir: Path, reports_dir: Path) -> None:
    images_dir = reports_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fraud_path = processed_dir / "fraud_transactions_enriched.csv"
    credit_path = processed_dir / "creditcard_raw.csv"

    fraud_X, fraud_y, fraud_preprocessor = prepare_fraud_dataset(fraud_path)
    credit_X, credit_y, credit_preprocessor = prepare_credit_dataset(credit_path)

    fraud_pipeline, fraud_X_train, fraud_X_test, fraud_y_test, fraud_y_pred, fraud_y_proba = train_random_forest(
        fraud_X, fraud_y, fraud_preprocessor, FRAUD_RF_PARAMS
    )
    credit_pipeline, credit_X_train, credit_X_test, credit_y_test, credit_y_pred, credit_y_proba = train_random_forest(
        credit_X, credit_y, credit_preprocessor, CREDIT_RF_PARAMS
    )

    summary: Dict[str, DatasetExplainabilitySummary] = {}
    summary["fraud_data"] = summarize_dataset(
        "Fraud Dataset",
        fraud_pipeline,
        fraud_X_train,
        fraud_X_test,
        fraud_y_test,
        fraud_y_pred,
        fraud_y_proba,
        images_dir,
    )
    summary["creditcard"] = summarize_dataset(
        "Credit Card Dataset",
        credit_pipeline,
        credit_X_train,
        credit_X_test,
        credit_y_test,
        credit_y_pred,
        credit_y_proba,
        images_dir,
    )

    output_path = reports_dir / "task3_shap_summary.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump({key: asdict(value) for key, value in summary.items()}, fp, indent=2)

    print(f"Task 3 explainability completed. Summary saved to {output_path}.")


if __name__ == "__main__":
    project_root = Path.cwd()
    main(
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "reports",
    )
