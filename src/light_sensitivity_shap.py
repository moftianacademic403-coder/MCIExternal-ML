from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from light_modeling import _pipeline, model_ready_frame
from mci_qc import write_csv
from mrmr_stability import rank_mrmr
from split_development import create_locked_split


CORE_AVAILABILITY_EXCLUSIONS = {
    "iadl",
    "vitamin_d",
    "total_pfat",
    "android_pfat",
    "gynoid_pfat",
}
FUNCTIONAL_OVERLAP_EXCLUSIONS = {"adl", "iadl"}


def _screening_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    values = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if threshold is not None:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        values["sensitivity"] = float(tp / (tp + fn)) if tp + fn else np.nan
        values["specificity"] = float(tn / (tn + fp)) if tn + fp else np.nan
    return values


def _partition_data(split: dict):
    development = split["development_eligible_in_memory"]
    return {
        "train_80": development.iloc[split["train_relative_indices"]].reset_index(drop=True),
        "internal_test_20_locked": development.iloc[
            split["test_relative_indices"]
        ].reset_index(drop=True),
        "external_validation": split["external_harmonized_in_memory"].reset_index(
            drop=True
        ),
    }


def _subgroup_rows(
    partition_name: str,
    frame: pd.DataFrame,
    bundle: dict,
) -> list[dict]:
    valid = frame["mci"].isin(["yes", "no"])
    analysis = frame.loc[valid].reset_index(drop=True)
    labels = analysis["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    selected = bundle["selected_features"]
    features = model_ready_frame(analysis, selected, bundle["variable_types"])
    probabilities = bundle["pipeline"].predict_proba(features[selected])[:, 1]
    threshold = float(bundle["threshold_from_training_oof_youden"])
    subgroup_definitions = {
        "sex": analysis["sex"].astype("string").fillna("missing"),
        "age_group": pd.cut(
            pd.to_numeric(analysis["age"], errors="coerce"),
            bins=[-np.inf, 64, 74, np.inf],
            labels=["under_65", "65_to_74", "75_plus"],
        ).astype("string").fillna("missing"),
        "selected_feature_completeness": np.where(
            analysis[selected].isna().any(axis=1),
            "at_least_one_missing",
            "complete_selected_predictors",
        ),
    }
    rows = []
    for subgroup_name, subgroup_values in subgroup_definitions.items():
        for level in sorted(pd.Series(subgroup_values).dropna().unique().tolist()):
            mask = np.asarray(subgroup_values == level)
            subgroup_labels = labels[mask]
            if int(mask.sum()) < 30 or np.unique(subgroup_labels).size < 2:
                continue
            metrics = _screening_metrics(
                subgroup_labels,
                probabilities[mask],
                threshold,
            )
            rows.extend(
                {
                    "partition": partition_name,
                    "model_name": bundle["model_name"],
                    "subgroup": subgroup_name,
                    "level": str(level),
                    "rows": int(mask.sum()),
                    "mci_yes_n": int(subgroup_labels.sum()),
                    "prevalence": float(subgroup_labels.mean()),
                    "metric": metric,
                    "estimate": estimate,
                    "threshold": threshold if metric in {"sensitivity", "specificity"} else np.nan,
                }
                for metric, estimate in metrics.items()
            )
    return rows


def _evaluate_sensitivity_model(
    scenario: str,
    model_name: str,
    pipeline,
    selected: list[str],
    variable_types: dict[str, str],
    partitions: dict[str, pd.DataFrame],
) -> list[dict]:
    rows = []
    for partition_name in ("internal_test_20_locked", "external_validation"):
        frame = partitions[partition_name]
        valid = frame["mci"].isin(["yes", "no"])
        analysis = frame.loc[valid].reset_index(drop=True)
        labels = analysis["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
        features = model_ready_frame(analysis, selected, variable_types)
        probabilities = pipeline.predict_proba(features[selected])[:, 1]
        metrics = _screening_metrics(labels, probabilities)
        rows.extend(
            {
                "scenario": scenario,
                "partition": partition_name,
                "model_name": model_name,
                "k": len(selected),
                "selected_features": "|".join(selected),
                "metric": metric,
                "estimate": estimate,
            }
            for metric, estimate in metrics.items()
        )
    return rows


def _original_feature_name(transformed_name: str, selected: list[str]) -> str:
    remainder = transformed_name.split("__", 1)[-1]
    if remainder.startswith("missingindicator_"):
        remainder = remainder[len("missingindicator_") :]
    for feature in sorted(selected, key=len, reverse=True):
        if remainder == feature or remainder.startswith(feature + "_"):
            return feature
    return remainder


def _tree_shap_rows(
    bundle: dict,
    train: pd.DataFrame,
    internal_test: pd.DataFrame,
) -> list[dict]:
    if bundle["model_name"] not in {"random_forest", "xgboost"}:
        return []
    selected = bundle["selected_features"]
    train_features = model_ready_frame(train, selected, bundle["variable_types"])
    test_features = model_ready_frame(
        internal_test, selected, bundle["variable_types"]
    )
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]
    estimator = bundle["pipeline"].named_steps["model"]
    transformed_test = preprocessor.transform(test_features[selected])
    transformed_names = preprocessor.get_feature_names_out().tolist()
    if bundle["model_name"] == "xgboost":
        contributions = estimator.get_booster().predict(
            xgb.DMatrix(transformed_test), pred_contribs=True
        )
        values = np.asarray(contributions[:, :-1])
    else:
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(transformed_test, check_additivity=False)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    transformed_frame = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [
                _original_feature_name(name, selected) for name in transformed_names
            ],
            "mean_abs_shap": mean_abs,
        }
    )
    grouped = (
        transformed_frame.groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return [
        {
            "model_name": bundle["model_name"],
            "explained_partition": "internal_test_20_locked",
            "rows_explained": int(len(internal_test)),
            **row._asdict(),
        }
        for row in grouped.itertuples(index=False)
    ]


def run_light_sensitivity_shap(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    partitions = _partition_data(split)
    train = partitions["train_80"]
    registry = split["registry"]
    predictors = registry.loc[registry["role"] == "predictor", "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train_labels_text = train["mci"].reset_index(drop=True)
    train_labels = train_labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)

    subgroup_rows = []
    sensitivity_rows = []
    shap_rows = []
    for model_path in sorted((light_output_dir / "models").glob("*.joblib")):
        bundle = joblib.load(model_path)
        for partition_name in ("internal_test_20_locked", "external_validation"):
            subgroup_rows.extend(
                _subgroup_rows(partition_name, partitions[partition_name], bundle)
            )

        selected_primary = bundle["selected_features"]
        sensitivity_rows.extend(
            _evaluate_sensitivity_model(
                "primary_locked",
                bundle["model_name"],
                bundle["pipeline"],
                selected_primary,
                bundle["variable_types"],
                partitions,
            )
        )

        scale_numeric = bundle["family"] in {"linear", "kernel"}
        for scenario, exclusions in (
            ("exclude_adl_iadl_overlap", FUNCTIONAL_OVERLAP_EXCLUSIONS),
            ("transportable_core", CORE_AVAILABILITY_EXCLUSIONS),
        ):
            candidates = [name for name in predictors if name not in exclusions]
            candidate_types = {name: variable_types[name] for name in candidates}
            train_features = model_ready_frame(train, candidates, candidate_types)
            ranking, _ = rank_mrmr(
                train_features,
                train_labels_text,
                candidate_types,
            )
            selected = ranking[: min(int(bundle["k"]), len(ranking))]
            pipeline = _pipeline(
                bundle["model_name"],
                bundle["params"],
                selected,
                variable_types,
                scale_numeric,
            )
            pipeline.fit(train_features[selected], train_labels)
            sensitivity_rows.extend(
                _evaluate_sensitivity_model(
                    scenario,
                    bundle["model_name"],
                    pipeline,
                    selected,
                    {name: variable_types[name] for name in selected},
                    partitions,
                )
            )

        numeric_selected = [
            name for name in selected_primary if variable_types[name] == "numeric"
        ]
        winsor_train = train.copy()
        winsor_partitions = {name: frame.copy() for name, frame in partitions.items()}
        for feature in numeric_selected:
            numeric = pd.to_numeric(train[feature], errors="coerce")
            lower = float(numeric.quantile(0.01))
            upper = float(numeric.quantile(0.99))
            winsor_train[feature] = numeric.clip(lower, upper)
            for partition_name in ("internal_test_20_locked", "external_validation"):
                values = pd.to_numeric(
                    winsor_partitions[partition_name][feature], errors="coerce"
                )
                winsor_partitions[partition_name][feature] = values.clip(lower, upper)
        winsor_partitions["train_80"] = winsor_train
        winsor_types = {name: variable_types[name] for name in selected_primary}
        winsor_pipeline = _pipeline(
            bundle["model_name"],
            bundle["params"],
            selected_primary,
            variable_types,
            scale_numeric,
        )
        winsor_features = model_ready_frame(
            winsor_train, selected_primary, winsor_types
        )
        winsor_pipeline.fit(winsor_features[selected_primary], train_labels)
        sensitivity_rows.extend(
            _evaluate_sensitivity_model(
                "winsorize_train_p01_p99",
                bundle["model_name"],
                winsor_pipeline,
                selected_primary,
                winsor_types,
                winsor_partitions,
            )
        )
        shap_rows.extend(
            _tree_shap_rows(
                bundle,
                train,
                partitions["internal_test_20_locked"],
            )
        )

    subgroup = pd.DataFrame(subgroup_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    shap_importance = pd.DataFrame(shap_rows)
    manifest = {
        "status": "lightweight_exploratory_not_for_confirmatory_claims",
        "subgroups": ["sex", "age_group", "selected_feature_completeness"],
        "minimum_subgroup_rows": 30,
        "subgroup_ci": "not computed in lightweight run",
        "sensitivity_scenarios": [
            "primary_locked",
            "exclude_adl_iadl_overlap",
            "transportable_core",
            "winsorize_train_p01_p99",
        ],
        "mice_status": "deferred_to_heavy_run",
        "shap_models": ["random_forest", "xgboost"],
        "shap_partition": "internal_test_20_locked",
        "shap_output": "mean absolute SHAP aggregated to original variables",
        "participant_level_outputs_written": False,
    }
    write_csv(subgroup, light_output_dir / "light_subgroup_metrics.csv")
    write_csv(sensitivity, light_output_dir / "light_sensitivity_metrics.csv")
    write_csv(shap_importance, light_output_dir / "light_shap_importance.csv")
    (light_output_dir / "light_sensitivity_shap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "subgroup": subgroup,
        "sensitivity": sensitivity,
        "shap_importance": shap_importance,
        "manifest": manifest,
    }
