from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    ParameterSampler,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC

from light_modeling import model_ready_frame
from mci_qc import write_csv
from mrmr_stability import rank_mrmr
from split_development import create_locked_split

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - recorded in manifest
    xgboost = None
    XGBClassifier = None

try:
    import tabpfn
    from tabpfn import TabPFNClassifier
except ImportError:  # pragma: no cover - allowed only with --skip-tabpfn
    tabpfn = None
    TabPFNClassifier = None


HEAVY_SEED = 20260729
K_CANDIDATES = (5, 10, 15, 20, 25, 30)
TABPFN_K_CANDIDATES = (10, 20, 30)


@dataclass(frozen=True)
class RunMode:
    outer_folds: int
    outer_repeats: int
    inner_folds: int
    classical_candidates: int
    bootstrap_repeats: int
    smoke: bool


def _run_mode(smoke: bool) -> RunMode:
    if smoke:
        return RunMode(
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            classical_candidates=2,
            bootstrap_repeats=200,
            smoke=True,
        )
    return RunMode(
        outer_folds=5,
        outer_repeats=3,
        inner_folds=4,
        classical_candidates=40,
        bootstrap_repeats=2000,
        smoke=False,
    )


def _classical_preprocessor(
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric = [
        feature
        for feature in selected_features
        if variable_types[feature] == "numeric"
    ]
    categorical = [
        feature
        for feature in selected_features
        if variable_types[feature] == "categorical"
    ]
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric)
        )
    if categorical:
        transformers.append(
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
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def _model_spaces() -> dict[str, dict[str, Any]]:
    spaces: dict[str, dict[str, Any]] = {
        "elastic_net_logistic": {
            "family": "linear",
            "scale_numeric": True,
            "space": {
                "k": list(K_CANDIDATES),
                "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        },
        "svm_rbf": {
            "family": "kernel",
            "scale_numeric": True,
            "space": {
                "k": list(K_CANDIDATES),
                "C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
            },
        },
        "random_forest": {
            "family": "bagged_trees",
            "scale_numeric": False,
            "space": {
                "k": list(K_CANDIDATES),
                "n_estimators": [300, 500, 800],
                "max_depth": [None, 6, 10, 16],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.5, 0.8],
            },
        },
        "xgboost": {
            "family": "boosted_trees",
            "scale_numeric": False,
            "space": {
                "k": list(K_CANDIDATES),
                "n_estimators": [300, 500, 800],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.70, 0.85, 1.0],
                "colsample_bytree": [0.70, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [1.0, 5.0, 10.0],
                "reg_alpha": [0.0, 0.1, 1.0],
            },
        },
        "tabpfn": {
            "family": "tabular_foundation_model",
            "scale_numeric": False,
            "space": {
                "k": list(TABPFN_K_CANDIDATES),
                "n_estimators": [8],
            },
        },
    }
    return spaces


def _candidate_list(
    model_name: str,
    space: dict[str, list[Any]],
    run_mode: RunMode,
    seed: int,
) -> list[dict[str, Any]]:
    if model_name == "tabpfn":
        candidates = [
            dict(zip(space.keys(), values))
            for values in itertools.product(*space.values())
        ]
        return candidates[:1] if run_mode.smoke else candidates
    return list(
        ParameterSampler(
            space,
            n_iter=run_mode.classical_candidates,
            random_state=seed,
        )
    )


def _make_classical_estimator(
    model_name: str,
    params: dict[str, Any],
    seed: int,
):
    if model_name == "elastic_net_logistic":
        return LogisticRegression(
            C=float(params["C"]),
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(params["l1_ratio"]),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    if model_name == "svm_rbf":
        return SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            cache_size=4096,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is required for the heavy run.")
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=seed,
        )
    raise ValueError(f"Unsupported classical model: {model_name}")


def _classical_pipeline(
    model_name: str,
    params: dict[str, Any],
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
    seed: int,
) -> Pipeline:
    estimator_params = {key: value for key, value in params.items() if key != "k"}
    return Pipeline(
        [
            (
                "preprocessor",
                _classical_preprocessor(
                    selected_features,
                    variable_types,
                    scale_numeric,
                ),
            ),
            (
                "model",
                _make_classical_estimator(model_name, estimator_params, seed),
            ),
        ]
    )


def _tabpfn_frame(
    frame: pd.DataFrame,
    selected_features: list[str],
    variable_types: dict[str, str],
) -> pd.DataFrame:
    prepared = frame[selected_features].copy()
    for feature in selected_features:
        if variable_types[feature] == "numeric":
            prepared[feature] = pd.to_numeric(
                prepared[feature], errors="coerce"
            ).astype(float)
        else:
            prepared[feature] = prepared[feature].astype("string")
    return prepared


def _tabpfn_predict(
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    validation_frame: pd.DataFrame,
    selected_features: list[str],
    variable_types: dict[str, str],
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    if TabPFNClassifier is None:
        raise RuntimeError(
            "tabpfn is unavailable. Install tabpfn==8.2.0 or use --skip-tabpfn."
        )
    if not _cuda_available():
        raise RuntimeError("TabPFN heavy run requires a CUDA GPU.")
    train_prepared = _tabpfn_frame(
        train_frame, selected_features, variable_types
    )
    validation_prepared = _tabpfn_frame(
        validation_frame, selected_features, variable_types
    )
    categorical_indices = [
        index
        for index, feature in enumerate(selected_features)
        if variable_types[feature] == "categorical"
    ]
    classifier = TabPFNClassifier(
        n_estimators=int(params["n_estimators"]),
        categorical_features_indices=categorical_indices,
        device="cuda",
        random_state=seed,
        show_progress_bar=False,
    )
    classifier.fit(train_prepared, train_labels)
    probabilities = classifier.predict_proba(validation_prepared)[:, 1]
    del classifier
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return np.asarray(probabilities, dtype=float)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    validation_frame: pd.DataFrame,
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
    seed: int,
) -> np.ndarray:
    return _fit_predict_many(
        model_name,
        params,
        train_frame,
        train_labels,
        [validation_frame],
        selected_features,
        variable_types,
        scale_numeric,
        seed,
    )[0]


def _fit_predict_many(
    model_name: str,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    validation_frames: list[pd.DataFrame],
    selected_features: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
    seed: int,
) -> list[np.ndarray]:
    if model_name == "tabpfn":
        if TabPFNClassifier is None:
            raise RuntimeError(
                "tabpfn is unavailable. Install tabpfn==8.2.0 or use "
                "--skip-tabpfn."
            )
        if not _cuda_available():
            raise RuntimeError("TabPFN heavy run requires a CUDA GPU.")
        train_prepared = _tabpfn_frame(
            train_frame, selected_features, variable_types
        )
        validation_prepared = [
            _tabpfn_frame(frame, selected_features, variable_types)
            for frame in validation_frames
        ]
        categorical_indices = [
            index
            for index, feature in enumerate(selected_features)
            if variable_types[feature] == "categorical"
        ]
        classifier = TabPFNClassifier(
            n_estimators=int(params["n_estimators"]),
            categorical_features_indices=categorical_indices,
            device="cuda",
            random_state=seed,
            show_progress_bar=False,
        )
        classifier.fit(train_prepared, train_labels)
        predictions = [
            np.asarray(classifier.predict_proba(frame)[:, 1], dtype=float)
            for frame in validation_prepared
        ]
        del classifier
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return predictions
    pipeline = _classical_pipeline(
        model_name,
        params,
        selected_features,
        variable_types,
        scale_numeric,
        seed,
    )
    pipeline.fit(train_frame[selected_features], train_labels)
    probabilities = [
        np.asarray(
            pipeline.predict_proba(frame[selected_features])[:, 1],
            dtype=float,
        )
        for frame in validation_frames
    ]
    del pipeline
    gc.collect()
    return probabilities


def _inner_cache(
    outer_train: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
    inner_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    labels_text = outer_train["mci"].reset_index(drop=True)
    labels = labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    features = model_ready_frame(outer_train, predictors, variable_types)
    cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    cache = []
    for fit_indices, validation_indices in cv.split(features, labels):
        ranking, _ = rank_mrmr(
            features.iloc[fit_indices],
            labels_text.iloc[fit_indices],
            {feature: variable_types[feature] for feature in predictors},
        )
        cache.append((fit_indices, validation_indices, ranking))
    return cache


def _evaluate_candidate(
    model_name: str,
    params: dict[str, Any],
    outer_train: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
    scale_numeric: bool,
    inner_cache: list[tuple[np.ndarray, np.ndarray, list[str]]],
    seed: int,
) -> dict[str, Any]:
    labels = outer_train["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    features = model_ready_frame(outer_train, predictors, variable_types)
    fold_auc = []
    fold_ap = []
    fold_brier = []
    k = int(params["k"])
    for inner_fold, (fit_indices, validation_indices, ranking) in enumerate(
        inner_cache, start=1
    ):
        selected_features = ranking[:k]
        probabilities = _fit_predict(
            model_name,
            params,
            features.iloc[fit_indices],
            labels[fit_indices],
            features.iloc[validation_indices],
            selected_features,
            variable_types,
            scale_numeric,
            seed + inner_fold,
        )
        fold_labels = labels[validation_indices]
        fold_auc.append(roc_auc_score(fold_labels, probabilities))
        fold_ap.append(average_precision_score(fold_labels, probabilities))
        fold_brier.append(brier_score_loss(fold_labels, probabilities))
    return {
        "mean_inner_auc": float(np.mean(fold_auc)),
        "sd_inner_auc": float(np.std(fold_auc, ddof=1)),
        "mean_inner_average_precision": float(np.mean(fold_ap)),
        "mean_inner_brier": float(np.mean(fold_brier)),
    }


def _choose_candidate(tuning: pd.DataFrame) -> pd.Series:
    return tuning.sort_values(
        ["mean_inner_auc", "mean_inner_brier", "k", "candidate_index"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _bootstrap_mean_interval(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.mean(
        values[rng.integers(0, len(values), size=(repeats, len(values)))],
        axis=1,
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize_outer_metrics(
    outer_metrics: pd.DataFrame,
    run_mode: RunMode,
) -> pd.DataFrame:
    rows = []
    for model_name, group in outer_metrics.groupby("model_name", sort=True):
        row: dict[str, Any] = {
            "model_name": model_name,
            "family": group["family"].iloc[0],
            "outer_fold_evaluations": int(len(group)),
        }
        for metric in ("roc_auc", "average_precision", "brier_score"):
            values = group[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_interval(
                values,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + sum(ord(char) for char in model_name + metric),
            )
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"sd_{metric}"] = float(np.std(values, ddof=1))
            row[f"mean_{metric}_ci_lower"] = lower
            row[f"mean_{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_roc_auc", "mean_brier_score", "model_name"],
        ascending=[False, True, True],
    )


def _paired_auc_differences(
    outer_metrics: pd.DataFrame,
    run_mode: RunMode,
) -> pd.DataFrame:
    pivot = outer_metrics.pivot(
        index=["outer_repeat", "outer_fold"],
        columns="model_name",
        values="roc_auc",
    )
    rows = []
    for left, right in itertools.combinations(sorted(pivot.columns), 2):
        differences = (pivot[left] - pivot[right]).to_numpy(dtype=float)
        lower, upper = _bootstrap_mean_interval(
            differences,
            run_mode.bootstrap_repeats,
            HEAVY_SEED + sum(ord(char) for char in left + right),
        )
        rows.append(
            {
                "model_left": left,
                "model_right": right,
                "mean_auc_difference_left_minus_right": float(
                    np.mean(differences)
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "outer_fold_pairs": int(len(differences)),
                "interpretation": (
                    "descriptive paired outer-fold bootstrap; repeated folds are "
                    "not fully independent"
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_tabpfn_preflight(skip_tabpfn: bool, smoke: bool) -> None:
    if skip_tabpfn:
        return
    if TabPFNClassifier is None:
        raise RuntimeError("Install tabpfn==8.2.0 before the heavy run.")
    if not _cuda_available():
        raise RuntimeError("Enable a Kaggle GPU before running TabPFN.")
    if smoke:
        return
    if not os.environ.get("TABPFN_TOKEN"):
        print(
            "Warning: TABPFN_TOKEN is not set. The first checkpoint download may "
            "require an interactive Prior Labs license/login flow. For unattended "
            "Kaggle execution, store TABPFN_TOKEN as a Kaggle Secret.",
            flush=True,
        )


def run_heavy_nested_cv(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    heavy_output_dir: Path,
    external_education_path: Path | None = None,
    smoke: bool = False,
    skip_tabpfn: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    run_mode = _run_mode(smoke)
    _validate_tabpfn_preflight(skip_tabpfn, smoke)
    split = create_locked_split(
        development_path,
        external_path,
        qc_output_dir,
        external_education_path=external_education_path,
    )
    development = split["development_eligible_in_memory"]
    registry = split["registry"]
    predictors = registry.loc[
        registry["role"].eq("predictor"), "canonical_name"
    ].tolist()
    variable_types = (
        registry.set_index("canonical_name")["variable_type"].to_dict()
    )
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    labels_text = train["mci"].reset_index(drop=True)
    labels = labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    features = model_ready_frame(train, predictors, variable_types)

    spaces = _model_spaces()
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required for the manuscript-grade run.")
    if skip_tabpfn:
        spaces.pop("tabpfn")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=run_mode.outer_folds,
        n_repeats=run_mode.outer_repeats,
        random_state=HEAVY_SEED,
    )
    outer_metric_rows = []
    tuning_rows = []
    selected_config_rows = []
    feature_rows = []
    for outer_index, (outer_train_indices, outer_validation_indices) in enumerate(
        outer_cv.split(features, labels), start=1
    ):
        outer_repeat = (outer_index - 1) // run_mode.outer_folds + 1
        outer_fold = (outer_index - 1) % run_mode.outer_folds + 1
        current_train = train.iloc[outer_train_indices].reset_index(drop=True)
        current_labels_text = current_train["mci"].reset_index(drop=True)
        current_labels = current_labels_text.map(
            {"no": 0, "yes": 1}
        ).to_numpy(dtype=int)
        current_features = model_ready_frame(
            current_train,
            predictors,
            variable_types,
        )
        validation_features = features.iloc[outer_validation_indices].reset_index(
            drop=True
        )
        validation_labels = labels[outer_validation_indices]
        inner_seed = HEAVY_SEED + outer_index * 100
        current_inner_cache = _inner_cache(
            current_train,
            predictors,
            variable_types,
            run_mode.inner_folds,
            inner_seed,
        )
        outer_ranking, _ = rank_mrmr(
            current_features,
            current_labels_text,
            {feature: variable_types[feature] for feature in predictors},
        )

        for model_offset, (model_name, spec) in enumerate(spaces.items(), start=1):
            print(
                f"Outer {outer_index}/{run_mode.outer_folds * run_mode.outer_repeats} "
                f"- tuning {model_name}",
                flush=True,
            )
            candidates = _candidate_list(
                model_name,
                spec["space"],
                run_mode,
                inner_seed + model_offset,
            )
            model_tuning_rows = []
            for candidate_index, params in enumerate(candidates, start=1):
                scores = _evaluate_candidate(
                    model_name,
                    params,
                    current_train,
                    predictors,
                    variable_types,
                    bool(spec["scale_numeric"]),
                    current_inner_cache,
                    inner_seed + model_offset * 1000 + candidate_index * 10,
                )
                row = {
                    "outer_repeat": outer_repeat,
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "family": spec["family"],
                    "candidate_index": candidate_index,
                    "k": int(params["k"]),
                    "params_json": json.dumps(
                        params,
                        sort_keys=True,
                        default=str,
                    ),
                    **scores,
                }
                tuning_rows.append(row)
                model_tuning_rows.append(row)
            best = _choose_candidate(pd.DataFrame(model_tuning_rows))
            best_params = json.loads(best["params_json"])
            selected_features = outer_ranking[: int(best["k"])]
            probabilities = _fit_predict(
                model_name,
                best_params,
                current_features,
                current_labels,
                validation_features,
                selected_features,
                variable_types,
                bool(spec["scale_numeric"]),
                HEAVY_SEED + outer_index * 10000 + model_offset,
            )
            outer_metric_rows.append(
                {
                    "outer_repeat": outer_repeat,
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "family": spec["family"],
                    "validation_rows": int(len(validation_labels)),
                    "mci_prevalence": float(np.mean(validation_labels)),
                    "roc_auc": float(
                        roc_auc_score(validation_labels, probabilities)
                    ),
                    "average_precision": float(
                        average_precision_score(validation_labels, probabilities)
                    ),
                    "brier_score": float(
                        brier_score_loss(validation_labels, probabilities)
                    ),
                }
            )
            selected_config_rows.append(
                {
                    "outer_repeat": outer_repeat,
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "family": spec["family"],
                    "k": int(best["k"]),
                    "params_json": best["params_json"],
                    "mean_inner_auc": float(best["mean_inner_auc"]),
                    "mean_inner_brier": float(best["mean_inner_brier"]),
                }
            )
            feature_rows.extend(
                {
                    "outer_repeat": outer_repeat,
                    "outer_fold": outer_fold,
                    "model_name": model_name,
                    "canonical_name": feature,
                    "rank_in_outer_training": rank,
                    "selected": rank <= int(best["k"]),
                    "selected_k": int(best["k"]),
                }
                for rank, feature in enumerate(outer_ranking, start=1)
            )

    outer_metrics = pd.DataFrame(outer_metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    selected_configs = pd.DataFrame(selected_config_rows)
    feature_selection = pd.DataFrame(feature_rows)
    summary = _summarize_outer_metrics(outer_metrics, run_mode)
    paired_differences = _paired_auc_differences(outer_metrics, run_mode)
    selected_model = summary.iloc[0]

    status = (
        "smoke_test_only_no_final_selection"
        if smoke
        else "development_only_nested_cv_model_family_selected"
    )
    selection = {
        "status": status,
        "selected_model_name": (
            None if smoke else str(selected_model["model_name"])
        ),
        "selected_family": None if smoke else str(selected_model["family"]),
        "selection_partition": "locked Development train-80 only",
        "primary_criterion": "mean outer-fold AUROC",
        "secondary_criterion": "mean outer-fold Brier score",
        "external_used_for_selection": False,
        "internal_test_20_used_for_selection": False,
        "tabpfn_included": "tabpfn" in spaces,
        "warning": (
            "Outer-fold bootstrap intervals are descriptive because folds from "
            "repeated CV are not fully independent."
        ),
    }
    manifest = {
        "status": status,
        "development_sha256": split["manifest"]["development_sha256"],
        "external_sha256": split["manifest"]["external_sha256"],
        "input_partition": "locked Development train-80 only",
        "train_rows": int(len(train)),
        "internal_test_rows_used": 0,
        "external_rows_used": 0,
        "predictor_count": int(len(predictors)),
        "outer_folds": run_mode.outer_folds,
        "outer_repeats": run_mode.outer_repeats,
        "inner_folds": run_mode.inner_folds,
        "classical_candidates_per_outer_fold": run_mode.classical_candidates,
        "tabpfn_candidates_per_outer_fold": (
            len(TABPFN_K_CANDIDATES) if "tabpfn" in spaces and not smoke else 0
        ),
        "mrmr_nested": (
            "recomputed in every inner-training fold; recomputed on the full "
            "outer-training fold before outer validation"
        ),
        "preprocessing_nested": True,
        "models": list(spaces),
        "tabpfn_version": getattr(tabpfn, "__version__", None),
        "tabpfn_model": "default TabPFN-3",
        "tabpfn_preprocessing": (
            "native missing/categorical handling; no one-hot encoding or scaling"
        ),
        "xgboost_version": getattr(xgboost, "__version__", None),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "cuda_available": _cuda_available(),
        "random_seed": HEAVY_SEED,
        "participant_level_predictions_written": False,
        "education_harmonization_mode": split["manifest"][
            "education_harmonization_mode"
        ],
        "next_stage": (
            "freeze the selected family, tune/finalize on Development train-80, "
            "then evaluate once on locked internal test and External"
        ),
    }
    heavy_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(outer_metrics, heavy_output_dir / "nested_outer_metrics.csv")
    write_csv(tuning, heavy_output_dir / "nested_tuning_candidates.csv")
    write_csv(
        selected_configs,
        heavy_output_dir / "nested_selected_configs.csv",
    )
    write_csv(
        feature_selection,
        heavy_output_dir / "nested_feature_selection.csv",
    )
    write_csv(summary, heavy_output_dir / "nested_model_summary.csv")
    write_csv(
        paired_differences,
        heavy_output_dir / "nested_paired_auc_differences.csv",
    )
    (heavy_output_dir / "nested_cv_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (heavy_output_dir / "model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "outer_metrics": outer_metrics,
        "tuning": tuning,
        "selected_configs": selected_configs,
        "feature_selection": feature_selection,
        "summary": summary,
        "paired_differences": paired_differences,
        "selection": selection,
        "manifest": manifest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only repeated nested CV for MCI model selection."
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-education", type=Path)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-tabpfn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_heavy_nested_cv(
        args.development,
        args.external,
        args.qc_output,
        args.output,
        external_education_path=args.external_education,
        smoke=args.smoke,
        skip_tabpfn=args.skip_tabpfn,
    )
    print(json.dumps(result["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
