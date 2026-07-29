from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC

from mci_qc import write_csv
from mrmr_stability import rank_mrmr
from split_development import create_locked_split

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - recorded in manifest
    xgboost = None
    XGBClassifier = None


LIGHT_SEED = 20260729
INNER_FOLDS = 3
K_CANDIDATES = (5, 10, 15, 20)


def model_ready_frame(
    frame: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
) -> pd.DataFrame:
    prepared = frame[predictors].copy()
    for name in predictors:
        if variable_types[name] == "categorical":
            values = prepared[name].astype(object)
            prepared[name] = values.where(pd.notna(values), np.nan)
        else:
            prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    return prepared


def _preprocessor(
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric = [name for name in selected_features if variable_types[name] == "numeric"]
    categorical = [
        name for name in selected_features if variable_types[name] == "categorical"
    ]
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def _model_specs() -> list[dict]:
    specs = [
        {
            "model_name": "logistic_regression",
            "family": "linear",
            "scale_numeric": True,
            "grid": {"C": [0.1, 1.0, 10.0]},
        },
        {
            "model_name": "svm_rbf",
            "family": "kernel",
            "scale_numeric": True,
            "grid": {"C": [0.5, 2.0]},
        },
        {
            "model_name": "random_forest",
            "family": "bagged_trees",
            "scale_numeric": False,
            "grid": {
                "max_depth": [None, 8],
                "min_samples_leaf": [2, 5],
            },
        },
    ]
    if XGBClassifier is not None:
        specs.append(
            {
                "model_name": "xgboost",
                "family": "boosted_trees",
                "scale_numeric": False,
                "grid": {
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                },
            }
        )
    return specs


def _make_estimator(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=LIGHT_SEED,
        )
    if model_name == "svm_rbf":
        return SVC(
            C=params["C"],
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=LIGHT_SEED,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=LIGHT_SEED,
        )
    if model_name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=160,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=LIGHT_SEED,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _pipeline(
    model_name: str,
    params: dict,
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                _preprocessor(selected_features, variable_types, scale_numeric),
            ),
            ("model", _make_estimator(model_name, params)),
        ]
    )


def _youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def run_light_tuning(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    development = split["development_eligible_in_memory"]
    registry = split["registry"]
    predictors = registry.loc[registry["role"] == "predictor", "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    features = model_ready_frame(train, predictors, variable_types)
    labels_text = train["mci"].reset_index(drop=True)
    labels = labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)

    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=LIGHT_SEED)
    fold_cache = []
    fold_feature_rows = []
    for fold, (fit_indices, validation_indices) in enumerate(
        cv.split(features, labels), start=1
    ):
        ranking, _ = rank_mrmr(
            features.iloc[fit_indices],
            labels_text.iloc[fit_indices],
            {name: variable_types[name] for name in predictors},
        )
        fold_cache.append((fit_indices, validation_indices, ranking))
        for rank, feature in enumerate(ranking, start=1):
            fold_feature_rows.append(
                {"fold": fold, "canonical_name": feature, "rank": rank}
            )

    tuning_rows = []
    for spec in _model_specs():
        for k, params in itertools.product(K_CANDIDATES, ParameterGrid(spec["grid"])):
            fold_aucs = []
            for fit_indices, validation_indices, ranking in fold_cache:
                selected = ranking[:k]
                pipeline = _pipeline(
                    spec["model_name"],
                    params,
                    selected,
                    variable_types,
                    spec["scale_numeric"],
                )
                pipeline.fit(features.iloc[fit_indices][selected], labels[fit_indices])
                probabilities = pipeline.predict_proba(
                    features.iloc[validation_indices][selected]
                )[:, 1]
                fold_aucs.append(
                    roc_auc_score(labels[validation_indices], probabilities)
                )
            tuning_rows.append(
                {
                    "model_name": spec["model_name"],
                    "family": spec["family"],
                    "k": k,
                    "params_json": json.dumps(params, sort_keys=True),
                    "mean_inner_auc": float(np.mean(fold_aucs)),
                    "sd_inner_auc": float(np.std(fold_aucs, ddof=1)),
                    **{
                        f"fold_{fold}_auc": float(value)
                        for fold, value in enumerate(fold_aucs, start=1)
                    },
                }
            )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["model_name", "mean_inner_auc", "k"],
        ascending=[True, False, True],
    )
    selected_rows = []
    model_dir = light_output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    full_ranking, _ = rank_mrmr(
        features,
        labels_text,
        {name: variable_types[name] for name in predictors},
    )
    spec_by_name = {spec["model_name"]: spec for spec in _model_specs()}
    for model_name, group in tuning.groupby("model_name", sort=True):
        best = group.sort_values(
            ["mean_inner_auc", "k"], ascending=[False, True]
        ).iloc[0]
        spec = spec_by_name[model_name]
        params = json.loads(best["params_json"])
        k = int(best["k"])
        oof_probabilities = np.full(len(features), np.nan, dtype=float)
        for fit_indices, validation_indices, ranking in fold_cache:
            selected_fold = ranking[:k]
            pipeline = _pipeline(
                model_name,
                params,
                selected_fold,
                variable_types,
                spec["scale_numeric"],
            )
            pipeline.fit(features.iloc[fit_indices][selected_fold], labels[fit_indices])
            oof_probabilities[validation_indices] = pipeline.predict_proba(
                features.iloc[validation_indices][selected_fold]
            )[:, 1]
        threshold = _youden_threshold(labels, oof_probabilities)
        selected_full = full_ranking[:k]
        final_pipeline = _pipeline(
            model_name,
            params,
            selected_full,
            variable_types,
            spec["scale_numeric"],
        )
        final_pipeline.fit(features[selected_full], labels)
        bundle = {
            "model_name": model_name,
            "family": spec["family"],
            "pipeline": final_pipeline,
            "selected_features": selected_full,
            "k": k,
            "params": params,
            "threshold_from_training_oof_youden": threshold,
            "mean_inner_auc": float(best["mean_inner_auc"]),
            "sd_inner_auc": float(best["sd_inner_auc"]),
            "variable_types": {name: variable_types[name] for name in selected_full},
        }
        joblib.dump(bundle, model_dir / f"{model_name}.joblib")
        selected_rows.append(
            {
                "model_name": model_name,
                "family": spec["family"],
                "k": k,
                "params_json": json.dumps(params, sort_keys=True),
                "mean_inner_auc": float(best["mean_inner_auc"]),
                "sd_inner_auc": float(best["sd_inner_auc"]),
                "training_oof_youden_threshold": threshold,
                "selected_features": "|".join(selected_full),
            }
        )

    selected_models = pd.DataFrame(selected_rows).sort_values(
        "mean_inner_auc", ascending=False
    )
    fold_features = pd.DataFrame(fold_feature_rows)
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "input_partition": "train_80_only",
        "train_rows": int(len(train)),
        "internal_test_rows_used": 0,
        "external_rows_used": 0,
        "inner_cv": f"{INNER_FOLDS}-fold StratifiedKFold",
        "inner_cv_seed": LIGHT_SEED,
        "mrmr_rerun_inside_each_fold": True,
        "k_candidates": list(K_CANDIDATES),
        "models": selected_models["model_name"].tolist(),
        "tabpfn_status": "deferred_not_installed_local_run",
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgboost, "__version__", None),
        "participant_level_predictions_written": False,
        "final_heavy_plan": "repeated nested CV on Kaggle; this run is a pipeline smoke test",
    }
    light_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tuning, light_output_dir / "light_tuning_results.csv")
    write_csv(selected_models, light_output_dir / "light_selected_models.csv")
    write_csv(fold_features, light_output_dir / "light_mrmr_fold_rankings.csv")
    (light_output_dir / "light_tuning_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "tuning": tuning,
        "selected_models": selected_models,
        "fold_features": fold_features,
        "manifest": manifest,
    }
