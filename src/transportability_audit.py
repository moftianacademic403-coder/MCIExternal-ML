from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from light_modeling import model_ready_frame
from mci_qc import write_csv
from split_development import create_locked_split


AUDIT_SEED = 20260730


def _numeric_drift(development: pd.Series, external: pd.Series) -> dict[str, float]:
    left = pd.to_numeric(development, errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(external, errors="coerce").dropna().to_numpy(dtype=float)
    left_var = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    right_var = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    pooled_sd = float(np.sqrt((left_var + right_var) / 2.0))
    standardized_mean_difference = (
        float((np.mean(right) - np.mean(left)) / pooled_sd)
        if pooled_sd > 0
        else 0.0
    )
    ks_statistic = (
        float(ks_2samp(left, right, alternative="two-sided").statistic)
        if len(left) and len(right)
        else np.nan
    )
    return {
        "development_mean": float(np.mean(left)) if len(left) else np.nan,
        "external_mean": float(np.mean(right)) if len(right) else np.nan,
        "development_median": float(np.median(left)) if len(left) else np.nan,
        "external_median": float(np.median(right)) if len(right) else np.nan,
        "standardized_mean_difference": standardized_mean_difference,
        "ks_statistic": ks_statistic,
        "total_variation_distance": np.nan,
    }


def _categorical_drift(
    development: pd.Series,
    external: pd.Series,
) -> dict[str, float]:
    left = development.astype("string").fillna("<missing>")
    right = external.astype("string").fillna("<missing>")
    levels = sorted(set(left.unique()) | set(right.unique()))
    left_probabilities = left.value_counts(normalize=True).reindex(levels, fill_value=0.0)
    right_probabilities = right.value_counts(normalize=True).reindex(levels, fill_value=0.0)
    total_variation_distance = float(
        0.5 * np.abs(left_probabilities - right_probabilities).sum()
    )
    return {
        "development_mean": np.nan,
        "external_mean": np.nan,
        "development_median": np.nan,
        "external_median": np.nan,
        "standardized_mean_difference": np.nan,
        "ks_statistic": np.nan,
        "total_variation_distance": total_variation_distance,
    }


def _featurewise_drift(
    development: pd.DataFrame,
    external: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in predictors:
        left = development[feature]
        right = external[feature]
        drift = (
            _numeric_drift(left, right)
            if variable_types[feature] == "numeric"
            else _categorical_drift(left, right)
        )
        missing_left = float(left.isna().mean())
        missing_right = float(right.isna().mean())
        components = [
            abs(float(drift["standardized_mean_difference"]))
            if pd.notna(drift["standardized_mean_difference"])
            else 0.0,
            float(drift["ks_statistic"])
            if pd.notna(drift["ks_statistic"])
            else 0.0,
            float(drift["total_variation_distance"])
            if pd.notna(drift["total_variation_distance"])
            else 0.0,
            abs(missing_right - missing_left),
        ]
        rows.append(
            {
                "canonical_name": feature,
                "variable_type": variable_types[feature],
                "development_rows": int(len(left)),
                "external_rows": int(len(right)),
                "development_missing_pct": 100.0 * missing_left,
                "external_missing_pct": 100.0 * missing_right,
                "missingness_difference_percentage_points": 100.0
                * (missing_right - missing_left),
                **drift,
                "drift_priority_score": max(components),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["drift_priority_score", "canonical_name"],
        ascending=[False, True],
    )


def _site_classifier(
    development: pd.DataFrame,
    external: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    combined = pd.concat(
        [development[predictors], external[predictors]],
        ignore_index=True,
    )
    labels = np.concatenate(
        [np.zeros(len(development), dtype=int), np.ones(len(external), dtype=int)]
    )
    numeric = [name for name in predictors if variable_types[name] == "numeric"]
    categorical = [
        name for name in predictors if variable_types[name] == "categorical"
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=True,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=AUDIT_SEED,
                ),
            ),
        ]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=AUDIT_SEED)
    probabilities = cross_val_predict(
        pipeline,
        combined,
        labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    pipeline.fit(combined, labels)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "site_log_odds_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    summary = {
        "status": "posthoc_predictor_shift_audit_completed",
        "development_rows": int(len(development)),
        "external_rows": int(len(external)),
        "predictor_count": int(len(predictors)),
        "site_classifier": "L2 logistic regression with training-fold preprocessing",
        "site_classifier_cv": "5-fold stratified cross-validation",
        "site_classifier_auc": auc,
        "interpretation": (
            "AUC quantifies predictor-distribution separability between sites; "
            "it does not use MCI outcomes and must not be used to re-select the "
            "primary model after locked external validation."
        ),
        "participant_level_outputs_written": False,
    }
    return summary, coefficient_frame


def run_transportability_audit(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    development = split["development_eligible_in_memory"]
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    registry = split["registry"]
    predictors = registry.loc[
        registry["role"].eq("predictor"), "canonical_name"
    ].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train_features = model_ready_frame(train, predictors, variable_types)
    external_features = model_ready_frame(external, predictors, variable_types)

    drift = _featurewise_drift(
        train_features,
        external_features,
        predictors,
        variable_types,
    )
    summary, coefficients = _site_classifier(
        train_features,
        external_features,
        predictors,
        variable_types,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(drift, output_dir / "featurewise_transport_drift.csv")
    write_csv(coefficients, output_dir / "site_classifier_coefficients.csv")
    (output_dir / "transportability_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"drift": drift, "coefficients": coefficients, "manifest": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc predictor-distribution audit between study sites."
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_transportability_audit(
        args.development,
        args.external,
        args.qc_output,
        args.output,
    )
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    main()
