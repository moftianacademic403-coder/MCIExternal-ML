from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from light_calibration_dca import _calibration_bins, _calibration_statistics
from light_evaluation import _bootstrap_intervals, _metric_values
from light_modeling import (
    INNER_FOLDS,
    LIGHT_SEED,
    _model_specs,
    _pipeline,
    model_ready_frame,
)
from mci_qc import write_csv
from mrmr_stability import rank_mrmr
from split_development import create_locked_split


TARGET_SENSITIVITIES = (0.80, 0.85, 0.90)
SCENARIO_PREVALENCES = (0.10, 0.20, 0.30)
RECALIBRATION_FOLDS = 10
LIGHT_BOOTSTRAP_REPEATS = 300


def _target_sensitivity_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> float:
    """Largest observed threshold attaining at least the requested sensitivity."""
    candidates = np.sort(np.unique(probabilities))[::-1]
    valid = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        true_positive = int(((labels == 1) & predictions).sum())
        false_negative = int(((labels == 1) & ~predictions).sum())
        sensitivity = true_positive / (true_positive + false_negative)
        if sensitivity >= target_sensitivity:
            valid.append(float(threshold))
    if not valid:
        return float(np.nextafter(np.min(probabilities), -np.inf))
    return max(valid)


def _training_oof_probabilities(
    train: pd.DataFrame,
    predictors: list[str],
    variable_types: dict[str, str],
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray]:
    features = model_ready_frame(train, predictors, variable_types)
    labels_text = train["mci"].reset_index(drop=True)
    labels = labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    probabilities = np.full(len(train), np.nan, dtype=float)
    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=LIGHT_SEED,
    )
    specs = {spec["model_name"]: spec for spec in _model_specs()}
    spec = specs[bundle["model_name"]]
    for fit_indices, validation_indices in cv.split(features, labels):
        ranking, _ = rank_mrmr(
            features.iloc[fit_indices],
            labels_text.iloc[fit_indices],
            {name: variable_types[name] for name in predictors},
        )
        selected = ranking[: int(bundle["k"])]
        pipeline = _pipeline(
            bundle["model_name"],
            bundle["params"],
            selected,
            variable_types,
            spec["scale_numeric"],
        )
        pipeline.fit(features.iloc[fit_indices][selected], labels[fit_indices])
        probabilities[validation_indices] = pipeline.predict_proba(
            features.iloc[validation_indices][selected]
        )[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Training OOF probabilities were not filled completely.")
    return labels, probabilities


def _predict_bundle(bundle: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    valid = frame["mci"].isin(["yes", "no"])
    analysis = frame.loc[valid].reset_index(drop=True)
    labels = analysis["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    selected = bundle["selected_features"]
    features = model_ready_frame(analysis, selected, bundle["variable_types"])
    probabilities = bundle["pipeline"].predict_proba(features[selected])[:, 1]
    return labels, probabilities


def _predictive_values(
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> tuple[float, float]:
    ppv_denominator = (
        sensitivity * prevalence
        + (1 - specificity) * (1 - prevalence)
    )
    npv_denominator = (
        (1 - sensitivity) * prevalence
        + specificity * (1 - prevalence)
    )
    ppv = sensitivity * prevalence / ppv_denominator
    npv = specificity * (1 - prevalence) / npv_denominator
    return float(ppv), float(npv)


def _scenario_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    prevalence: float,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws = {"ppv": [], "npv": []}
    completed = 0
    while completed < repeats:
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_predictions = probabilities[indices] >= threshold
        true_positive = int(((sampled_labels == 1) & sampled_predictions).sum())
        false_negative = int(((sampled_labels == 1) & ~sampled_predictions).sum())
        true_negative = int(((sampled_labels == 0) & ~sampled_predictions).sum())
        false_positive = int(((sampled_labels == 0) & sampled_predictions).sum())
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        ppv, npv = _predictive_values(
            sensitivity,
            specificity,
            prevalence,
        )
        draws["ppv"].append(ppv)
        draws["npv"].append(npv)
        completed += 1
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in draws.items()
    }


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _new_recalibrator() -> LogisticRegression:
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )


def _cross_validated_external_recalibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict]:
    logits = _logit(probabilities)
    cv = StratifiedKFold(
        n_splits=RECALIBRATION_FOLDS,
        shuffle=True,
        random_state=LIGHT_SEED + 97,
    )
    recalibrated = np.full(len(labels), np.nan, dtype=float)
    coefficient_rows = []
    for fold, (fit_indices, validation_indices) in enumerate(
        cv.split(logits, labels), start=1
    ):
        model = _new_recalibrator()
        model.fit(logits[fit_indices], labels[fit_indices])
        recalibrated[validation_indices] = model.predict_proba(
            logits[validation_indices]
        )[:, 1]
        coefficient_rows.append(
            {
                "fit_scope": "external_10fold_training_fold",
                "fold": fold,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(validation_indices)),
                "intercept_a": float(model.intercept_[0]),
                "slope_b": float(model.coef_[0, 0]),
            }
        )
    if np.isnan(recalibrated).any():
        raise RuntimeError("External OOF recalibrated probabilities are incomplete.")
    full_model = _new_recalibrator()
    full_model.fit(logits, labels)
    full_fit = {
        "fit_scope": "full_external_for_future_local_use_not_evaluated_here",
        "fold": np.nan,
        "fit_rows": int(len(labels)),
        "validation_rows": 0,
        "intercept_a": float(full_model.intercept_[0]),
        "slope_b": float(full_model.coef_[0, 0]),
    }
    return recalibrated, coefficient_rows, full_fit


def _bootstrap_calibration_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    completed = 0
    while completed < repeats:
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        try:
            statistics = _calibration_statistics(
                sampled_labels,
                probabilities[indices],
            )
        except ValueError:
            continue
        for metric, value in statistics.items():
            if np.isfinite(value):
                draws.setdefault(metric, []).append(float(value))
        completed += 1
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in draws.items()
    }


def run_light_operating_points_recalibration(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
    external_education_path: Path | None = None,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(
        development_path,
        external_path,
        qc_output_dir,
        external_education_path=external_education_path,
    )
    development = split["development_eligible_in_memory"]
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    internal_test = development.iloc[split["test_relative_indices"]].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    registry = split["registry"]
    predictors = registry.loc[
        registry["role"].eq("predictor"), "canonical_name"
    ].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()

    threshold_rows = []
    operating_rows = []
    scenario_rows = []
    for model_path in sorted((light_output_dir / "models").glob("*.joblib")):
        bundle = joblib.load(model_path)
        train_labels, train_oof = _training_oof_probabilities(
            train,
            predictors,
            variable_types,
            bundle,
        )
        partition_predictions = {
            "internal_test_20_locked": _predict_bundle(bundle, internal_test),
            "external_validation": _predict_bundle(bundle, external),
        }
        for target in TARGET_SENSITIVITIES:
            threshold = _target_sensitivity_threshold(train_labels, train_oof, target)
            training_values = _metric_values(train_labels, train_oof, threshold)
            threshold_rows.append(
                {
                    "model_name": bundle["model_name"],
                    "target_sensitivity": target,
                    "threshold": threshold,
                    "threshold_source": "Development train_80 OOF only",
                    "train_oof_sensitivity": training_values["sensitivity"],
                    "train_oof_specificity": training_values["specificity"],
                    "train_oof_ppv": training_values["ppv"],
                    "train_oof_npv": training_values["npv"],
                }
            )
            for partition, (labels, probabilities) in partition_predictions.items():
                values = _metric_values(labels, probabilities, threshold)
                intervals = _bootstrap_intervals(
                    labels,
                    probabilities,
                    threshold,
                    LIGHT_BOOTSTRAP_REPEATS,
                    LIGHT_SEED
                    + sum(ord(char) for char in bundle["model_name"] + partition)
                    + int(target * 1000),
                )
                for metric in (
                    "sensitivity",
                    "specificity",
                    "ppv",
                    "npv",
                    "accuracy",
                    "balanced_accuracy",
                ):
                    lower, upper = intervals[metric]
                    operating_rows.append(
                        {
                            "partition": partition,
                            "model_name": bundle["model_name"],
                            "target_sensitivity_from_train_oof": target,
                            "threshold": threshold,
                            "metric": metric,
                            "estimate": values[metric],
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "ci_method": (
                                f"percentile bootstrap, {LIGHT_BOOTSTRAP_REPEATS} repeats"
                            ),
                        }
                    )
                for prevalence in SCENARIO_PREVALENCES:
                    ppv, npv = _predictive_values(
                        values["sensitivity"],
                        values["specificity"],
                        prevalence,
                    )
                    scenario_intervals = _scenario_bootstrap(
                        labels,
                        probabilities,
                        threshold,
                        prevalence,
                        LIGHT_BOOTSTRAP_REPEATS,
                        LIGHT_SEED
                        + sum(ord(char) for char in bundle["model_name"] + partition)
                        + int(target * 1000)
                        + int(prevalence * 10000),
                    )
                    for metric, estimate in (("ppv", ppv), ("npv", npv)):
                        lower, upper = scenario_intervals[metric]
                        scenario_rows.append(
                            {
                                "partition_accuracy_source": partition,
                                "model_name": bundle["model_name"],
                                "target_sensitivity_from_train_oof": target,
                                "threshold": threshold,
                                "assumed_prevalence": prevalence,
                                "metric": metric,
                                "estimate": estimate,
                                "ci_lower": lower,
                                "ci_upper": upper,
                                "assumption": (
                                    "partition sensitivity/specificity transport unchanged "
                                    "to the hypothetical prevalence"
                                ),
                            }
                        )

    recalibration_metric_rows = []
    recalibration_bin_rows = []
    recalibration_coefficient_rows = []
    for model_path in sorted(
        (light_output_dir / "calibrated_models").glob("*_platt.joblib")
    ):
        bundle = joblib.load(model_path)
        labels, locked_probabilities = _predict_bundle(bundle, external)
        recalibrated_probabilities, fold_coefficients, full_fit = (
            _cross_validated_external_recalibration(labels, locked_probabilities)
        )
        for row in fold_coefficients + [full_fit]:
            recalibration_coefficient_rows.append(
                {"model_name": bundle["model_name"], **row}
            )
        for analysis_name, probabilities in (
            ("development_locked_platt", locked_probabilities),
            (
                "external_10fold_oof_logistic_recalibration",
                recalibrated_probabilities,
            ),
        ):
            statistics = _calibration_statistics(labels, probabilities)
            intervals = _bootstrap_calibration_intervals(
                labels,
                probabilities,
                LIGHT_BOOTSTRAP_REPEATS,
                LIGHT_SEED
                + sum(ord(char) for char in bundle["model_name"] + analysis_name),
            )
            for metric, estimate in statistics.items():
                lower, upper = intervals[metric]
                recalibration_metric_rows.append(
                    {
                        "partition": "external_validation",
                        "model_name": bundle["model_name"],
                        "analysis": analysis_name,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_method": (
                            f"percentile bootstrap of predictions, "
                            f"{LIGHT_BOOTSTRAP_REPEATS} repeats"
                        ),
                    }
                )
            recalibration_bin_rows.extend(
                _calibration_bins(
                    "external_validation",
                    bundle["model_name"],
                    analysis_name,
                    labels,
                    probabilities,
                )
            )

    thresholds = pd.DataFrame(threshold_rows)
    operating_points = pd.DataFrame(operating_rows)
    prevalence_scenarios = pd.DataFrame(scenario_rows)
    recalibration_metrics = pd.DataFrame(recalibration_metric_rows)
    recalibration_bins = pd.DataFrame(recalibration_bin_rows)
    recalibration_coefficients = pd.DataFrame(recalibration_coefficient_rows)
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "target_sensitivities": list(TARGET_SENSITIVITIES),
        "threshold_source": "Development train_80 OOF predictions only",
        "internal_test_used_to_choose_thresholds": False,
        "external_used_to_choose_thresholds": False,
        "scenario_prevalences": list(SCENARIO_PREVALENCES),
        "scenario_warning": (
            "illustrative only; assumes sensitivity and specificity transport "
            "unchanged despite possible case-mix spectrum effects"
        ),
        "external_recalibration_role": (
            "secondary local model updating, not performance of the unchanged "
            "externally validated model"
        ),
        "external_recalibration_evaluation": (
            f"{RECALIBRATION_FOLDS}-fold out-of-fold logistic recalibration"
        ),
        "external_full_fit_role": (
            "parameters saved for possible future local use; no same-data "
            "post-recalibration performance claim"
        ),
        "bootstrap_repeats": LIGHT_BOOTSTRAP_REPEATS,
        "participant_level_predictions_written": False,
    }
    write_csv(
        thresholds,
        light_output_dir / "light_operating_point_thresholds.csv",
    )
    write_csv(
        operating_points,
        light_output_dir / "light_operating_point_metrics.csv",
    )
    write_csv(
        prevalence_scenarios,
        light_output_dir / "light_prevalence_scenario_predictive_values.csv",
    )
    write_csv(
        recalibration_metrics,
        light_output_dir / "light_external_recalibration_metrics.csv",
    )
    write_csv(
        recalibration_bins,
        light_output_dir / "light_external_recalibration_bins.csv",
    )
    write_csv(
        recalibration_coefficients,
        light_output_dir / "light_external_recalibration_coefficients.csv",
    )
    (light_output_dir / "light_operating_points_recalibration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "thresholds": thresholds,
        "operating_points": operating_points,
        "prevalence_scenarios": prevalence_scenarios,
        "recalibration_metrics": recalibration_metrics,
        "recalibration_bins": recalibration_bins,
        "recalibration_coefficients": recalibration_coefficients,
        "manifest": manifest,
    }
