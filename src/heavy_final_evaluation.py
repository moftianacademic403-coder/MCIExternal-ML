from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from heavy_nested_cv import (
    HEAVY_SEED,
    _candidate_list,
    _choose_candidate,
    _cuda_available,
    _evaluate_candidate,
    _fit_predict,
    _fit_predict_many,
    _inner_cache,
    _model_spaces,
    _run_mode,
    _validate_tabpfn_preflight,
)
from light_modeling import model_ready_frame
from mci_qc import write_csv
from mrmr_stability import rank_mrmr
from split_development import create_locked_split


TARGET_SENSITIVITIES = (0.80, 0.85, 0.90)
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.801, 0.01), 2)
LOCAL_FOLDS = 10


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _new_logistic_calibrator() -> LogisticRegression:
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=HEAVY_SEED,
    )


def _fit_calibrator(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> LogisticRegression:
    calibrator = _new_logistic_calibrator()
    calibrator.fit(_logit(probabilities).reshape(-1, 1), labels)
    return calibrator


def _apply_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


def _youden_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities
    )
    finite = np.isfinite(thresholds)
    index = int(
        np.argmax(true_positive_rate[finite] - false_positive_rate[finite])
    )
    return float(thresholds[finite][index])


def _target_sensitivity_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> float:
    positive_probabilities = np.sort(probabilities[labels == 1])
    false_negatives_allowed = int(
        np.floor((1 - target_sensitivity) * len(positive_probabilities))
    )
    index = min(false_negatives_allowed, len(positive_probabilities) - 1)
    threshold = float(positive_probabilities[index])
    while np.mean(probabilities[labels == 1] >= threshold) < target_sensitivity:
        threshold = float(np.nextafter(threshold, -np.inf))
    return threshold


def _decision_metrics(
    labels: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, float]:
    decisions = decisions.astype(bool)
    true_positive = int(np.sum((labels == 1) & decisions))
    false_negative = int(np.sum((labels == 1) & ~decisions))
    true_negative = int(np.sum((labels == 0) & ~decisions))
    false_positive = int(np.sum((labels == 0) & decisions))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    ppv_denominator = true_positive + false_positive
    npv_denominator = true_negative + false_negative
    ppv = true_positive / ppv_denominator if ppv_denominator else np.nan
    npv = true_negative / npv_denominator if npv_denominator else np.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((true_positive + true_negative) / len(labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def _stratified_bootstrap_indices(
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rows = []
    for _ in range(repeats):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        rng.shuffle(sampled)
        rows.append(sampled)
    return np.asarray(rows, dtype=int)


def _iid_bootstrap_indices(
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < repeats:
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size == 2:
            rows.append(indices)
    return np.asarray(rows, dtype=int)


def _calibration_statistics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | str]:
    logits = _logit(probabilities)
    calibrator = _fit_calibrator(labels, probabilities)
    intercept = float(calibrator.intercept_[0])
    slope = float(calibrator.coef_[0, 0])
    fitted = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    design = np.column_stack([np.ones(len(labels)), logits])
    weights = fitted * (1 - fitted)
    covariance = np.linalg.pinv(design.T @ (weights[:, None] * design))
    standard_errors = np.sqrt(np.diag(covariance))
    return {
        "calibration_intercept": intercept,
        "calibration_intercept_ci_lower": float(
            intercept - 1.959963984540054 * standard_errors[0]
        ),
        "calibration_intercept_ci_upper": float(
            intercept + 1.959963984540054 * standard_errors[0]
        ),
        "calibration_slope": slope,
        "calibration_slope_ci_lower": float(
            slope - 1.959963984540054 * standard_errors[1]
        ),
        "calibration_slope_ci_upper": float(
            slope + 1.959963984540054 * standard_errors[1]
        ),
        "calibration_ci_method": "model-based Wald interval",
    }


def _bootstrap_evaluation_rows(
    partition: str,
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    indices = _stratified_bootstrap_indices(labels, repeats, seed)
    rows = []
    discrimination = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    draws = {metric: [] for metric in discrimination}
    for sampled_indices in indices:
        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[sampled_indices]
        draws["roc_auc"].append(
            roc_auc_score(sampled_labels, sampled_probabilities)
        )
        draws["average_precision"].append(
            average_precision_score(sampled_labels, sampled_probabilities)
        )
        draws["brier_score"].append(
            brier_score_loss(sampled_labels, sampled_probabilities)
        )
    for metric, estimate in discrimination.items():
        rows.append(
            {
                "partition": partition,
                "model_name": model_name,
                "operating_point": "threshold_free",
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.quantile(draws[metric], 0.025)),
                "ci_upper": float(np.quantile(draws[metric], 0.975)),
                "ci_method": f"stratified percentile bootstrap, {repeats} repeats",
            }
        )
    calibration = _calibration_statistics(labels, probabilities)
    for metric in ("calibration_intercept", "calibration_slope"):
        rows.append(
            {
                "partition": partition,
                "model_name": model_name,
                "operating_point": "threshold_free",
                "metric": metric,
                "estimate": calibration[metric],
                "ci_lower": calibration[f"{metric}_ci_lower"],
                "ci_upper": calibration[f"{metric}_ci_upper"],
                "ci_method": calibration["calibration_ci_method"],
            }
        )

    for threshold_name, threshold in thresholds.items():
        decisions = probabilities >= threshold
        estimates = _decision_metrics(labels, decisions)
        metric_draws = {metric: [] for metric in estimates}
        for sampled_indices in indices:
            values = _decision_metrics(
                labels[sampled_indices], decisions[sampled_indices]
            )
            for metric, value in values.items():
                metric_draws[metric].append(value)
        for metric, estimate in estimates.items():
            rows.append(
                {
                    "partition": partition,
                    "model_name": model_name,
                    "operating_point": threshold_name,
                    "threshold": float(threshold),
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": float(
                        np.quantile(metric_draws[metric], 0.025)
                    ),
                    "ci_upper": float(
                        np.quantile(metric_draws[metric], 0.975)
                    ),
                    "ci_method": (
                        f"stratified percentile bootstrap, {repeats} repeats"
                    ),
                }
            )
    return rows


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def _reliability_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    analysis: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "observed": labels,
            "predicted": probabilities,
            "rank": pd.Series(probabilities).rank(method="first"),
        }
    )
    frame["bin"] = pd.qcut(
        frame["rank"], q=10, labels=False, duplicates="drop"
    )
    rows = []
    for bin_number, group in frame.groupby("bin", observed=True):
        lower, upper = _wilson_interval(
            int(group["observed"].sum()), len(group)
        )
        rows.append(
            {
                "model_name": model_name,
                "analysis": analysis,
                "bin": int(bin_number) + 1,
                "rows": int(len(group)),
                "mean_predicted": float(group["predicted"].mean()),
                "observed_rate": float(group["observed"].mean()),
                "observed_rate_ci_lower": lower,
                "observed_rate_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def _cross_fitted_local_update(
    labels: np.ndarray,
    locked_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cv = StratifiedKFold(
        n_splits=LOCAL_FOLDS,
        shuffle=True,
        random_state=HEAVY_SEED + 97,
    )
    logits = _logit(locked_probabilities)
    recalibrated = np.full(len(labels), np.nan, dtype=float)
    raw_decisions = np.zeros(len(labels), dtype=bool)
    recalibrated_decisions = np.zeros(len(labels), dtype=bool)
    fold_rows = []
    for fold, (fit_indices, validation_indices) in enumerate(
        cv.split(logits, labels), start=1
    ):
        calibrator = _new_logistic_calibrator()
        calibrator.fit(logits[fit_indices].reshape(-1, 1), labels[fit_indices])
        fit_recalibrated = calibrator.predict_proba(
            logits[fit_indices].reshape(-1, 1)
        )[:, 1]
        validation_recalibrated = calibrator.predict_proba(
            logits[validation_indices].reshape(-1, 1)
        )[:, 1]
        recalibrated[validation_indices] = validation_recalibrated
        raw_threshold = _target_sensitivity_threshold(
            labels[fit_indices], locked_probabilities[fit_indices], 0.85
        )
        recalibrated_threshold = _target_sensitivity_threshold(
            labels[fit_indices], fit_recalibrated, 0.85
        )
        raw_decisions[validation_indices] = (
            locked_probabilities[validation_indices] >= raw_threshold
        )
        recalibrated_decisions[validation_indices] = (
            validation_recalibrated >= recalibrated_threshold
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(validation_indices)),
                "recalibration_intercept": float(calibrator.intercept_[0]),
                "recalibration_slope": float(calibrator.coef_[0, 0]),
                "raw_local_threshold_85": raw_threshold,
                "recalibrated_local_threshold_85": recalibrated_threshold,
            }
        )
    if np.isnan(recalibrated).any():
        raise RuntimeError("Incomplete External OOF recalibration predictions.")
    return (
        recalibrated,
        raw_decisions,
        recalibrated_decisions,
        pd.DataFrame(fold_rows),
    )


def _dca_rows(
    labels: np.ndarray,
    strategies: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    bootstrap_indices = _iid_bootstrap_indices(labels, repeats, seed)
    bootstrap_counts = np.vstack(
        [np.bincount(indices, minlength=len(labels)) for indices in bootstrap_indices]
    ).astype(float)
    positive = labels == 1
    negative = labels == 0
    threshold_odds = DCA_THRESHOLDS / (1 - DCA_THRESHOLDS)
    decisions_by_strategy: dict[str, np.ndarray] = {}
    for strategy, values in strategies.items():
        if values.dtype == bool:
            decisions_by_strategy[strategy] = values
        else:
            decisions_by_strategy[strategy] = (
                values[:, None] >= DCA_THRESHOLDS[None, :]
            )
    decisions_by_strategy["screen_all"] = np.ones(len(labels), dtype=bool)
    decisions_by_strategy["screen_none"] = np.zeros(len(labels), dtype=bool)

    estimates: dict[str, np.ndarray] = {}
    draws: dict[str, np.ndarray] = {}
    for strategy, decisions in decisions_by_strategy.items():
        if decisions.ndim == 1:
            true_positive = np.repeat(
                np.sum(positive & decisions), len(DCA_THRESHOLDS)
            )
            false_positive = np.repeat(
                np.sum(negative & decisions), len(DCA_THRESHOLDS)
            )
            boot_tp = np.repeat(
                (bootstrap_counts @ (positive & decisions).astype(float))[:, None],
                len(DCA_THRESHOLDS),
                axis=1,
            )
            boot_fp = np.repeat(
                (bootstrap_counts @ (negative & decisions).astype(float))[:, None],
                len(DCA_THRESHOLDS),
                axis=1,
            )
        else:
            true_positive = np.sum(positive[:, None] & decisions, axis=0)
            false_positive = np.sum(negative[:, None] & decisions, axis=0)
            boot_tp = bootstrap_counts @ (positive[:, None] & decisions)
            boot_fp = bootstrap_counts @ (negative[:, None] & decisions)
        estimates[strategy] = (
            true_positive / len(labels)
            - false_positive / len(labels) * threshold_odds
        )
        draws[strategy] = (
            boot_tp / len(labels)
            - boot_fp / len(labels) * threshold_odds[None, :]
        )

    rows = []
    for threshold_index, threshold_probability in enumerate(DCA_THRESHOLDS):
        all_draws = draws["screen_all"][:, threshold_index]
        for strategy in decisions_by_strategy:
            current_draws = draws[strategy][:, threshold_index]
            difference_draws = current_draws - all_draws
            rows.append(
                {
                    "strategy": strategy,
                    "threshold_probability": threshold_probability,
                    "net_benefit": estimates[strategy][threshold_index],
                    "ci_lower": float(np.quantile(current_draws, 0.025)),
                    "ci_upper": float(np.quantile(current_draws, 0.975)),
                    "delta_vs_screen_all": float(
                        estimates[strategy][threshold_index]
                        - estimates["screen_all"][threshold_index]
                    ),
                    "delta_vs_screen_all_ci_lower": float(
                        np.quantile(difference_draws, 0.025)
                    ),
                    "delta_vs_screen_all_ci_upper": float(
                        np.quantile(difference_draws, 0.975)
                    ),
                    "ci_method": (
                        f"paired stratified bootstrap, {repeats} repeats"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _metric_row(
    model_name: str,
    layer: str,
    probability_source: str,
    threshold_source: str,
    labels: np.ndarray,
    decisions: np.ndarray,
    probabilities: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    estimates = _decision_metrics(labels, decisions)
    estimates["brier_score"] = float(
        brier_score_loss(labels, probabilities)
    )
    draws = {metric: [] for metric in estimates}
    indices = _iid_bootstrap_indices(labels, repeats, seed)
    for sampled_indices in indices:
        sampled = _decision_metrics(
            labels[sampled_indices], decisions[sampled_indices]
        )
        sampled["brier_score"] = float(
            brier_score_loss(
                labels[sampled_indices], probabilities[sampled_indices]
            )
        )
        for metric, value in sampled.items():
            draws[metric].append(value)
    row = {
        "model_name": model_name,
        "layer": layer,
        "probability_source": probability_source,
        "threshold_source": threshold_source,
    }
    for metric, estimate in estimates.items():
        row[metric] = estimate
        row[f"{metric}_ci_lower"] = float(
            np.quantile(draws[metric], 0.025)
        )
        row[f"{metric}_ci_upper"] = float(
            np.quantile(draws[metric], 0.975)
        )
    row["ci_method"] = f"participant percentile bootstrap, {repeats} repeats"
    return row


def run_heavy_final_evaluation(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    selection_output_dir: Path,
    final_output_dir: Path,
    smoke: bool = False,
    skip_tabpfn: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    run_mode = _run_mode(smoke)
    _validate_tabpfn_preflight(skip_tabpfn, smoke)
    selection = json.loads(
        (selection_output_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    if selection["external_used_for_selection"]:
        raise RuntimeError("External leakage detected in model selection input.")
    if smoke:
        nested_summary = pd.read_csv(
            selection_output_dir / "nested_model_summary.csv",
            encoding="utf-8-sig",
        )
        selected_model_name = str(nested_summary.iloc[0]["model_name"])
    else:
        selected_model_name = selection["selected_model_name"]
        if not selected_model_name:
            raise RuntimeError("The full nested-CV run did not select a model family.")

    split = create_locked_split(development_path, external_path, qc_output_dir)
    development = split["development_eligible_in_memory"]
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    registry = split["registry"]
    predictors = registry.loc[
        registry["role"].eq("predictor"), "canonical_name"
    ].tolist()
    variable_types = (
        registry.set_index("canonical_name")["variable_type"].to_dict()
    )
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    internal_test = development.iloc[
        split["test_relative_indices"]
    ].reset_index(drop=True)
    train_labels_text = train["mci"].reset_index(drop=True)
    train_labels = train_labels_text.map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    internal_labels = internal_test["mci"].map(
        {"no": 0, "yes": 1}
    ).to_numpy(dtype=int)
    external_valid = external["mci"].isin(["yes", "no"])
    external = external.loc[external_valid].reset_index(drop=True)
    external_labels = external["mci"].map(
        {"no": 0, "yes": 1}
    ).to_numpy(dtype=int)
    train_features = model_ready_frame(train, predictors, variable_types)
    internal_features = model_ready_frame(
        internal_test, predictors, variable_types
    )
    external_features = model_ready_frame(external, predictors, variable_types)

    spaces = _model_spaces()
    if skip_tabpfn:
        spaces.pop("tabpfn")
    if selected_model_name not in spaces:
        raise RuntimeError(
            f"Selected family {selected_model_name!r} is unavailable in this run."
        )

    full_inner_cache = _inner_cache(
        train,
        predictors,
        variable_types,
        run_mode.inner_folds,
        HEAVY_SEED + 500000,
    )
    full_ranking, _ = rank_mrmr(
        train_features,
        train_labels_text,
        {feature: variable_types[feature] for feature in predictors},
    )
    tuning_rows = []
    config_rows = []
    threshold_rows = []
    evaluation_rows = []
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    for model_offset, (model_name, spec) in enumerate(spaces.items(), start=1):
        print(f"Full Development tuning: {model_name}", flush=True)
        candidates = _candidate_list(
            model_name,
            spec["space"],
            run_mode,
            HEAVY_SEED + 510000 + model_offset,
        )
        model_tuning = []
        for candidate_index, params in enumerate(candidates, start=1):
            scores = _evaluate_candidate(
                model_name,
                params,
                train,
                predictors,
                variable_types,
                bool(spec["scale_numeric"]),
                full_inner_cache,
                HEAVY_SEED + 520000 + model_offset * 1000 + candidate_index * 10,
            )
            row = {
                "model_name": model_name,
                "family": spec["family"],
                "candidate_index": candidate_index,
                "k": int(params["k"]),
                "params_json": json.dumps(params, sort_keys=True, default=str),
                **scores,
            }
            tuning_rows.append(row)
            model_tuning.append(row)
        best = _choose_candidate(pd.DataFrame(model_tuning))
        best_params = json.loads(best["params_json"])
        k = int(best["k"])
        oof_raw = np.full(len(train), np.nan, dtype=float)
        for inner_fold, (fit_indices, validation_indices, ranking) in enumerate(
            full_inner_cache, start=1
        ):
            selected_fold = ranking[:k]
            oof_raw[validation_indices] = _fit_predict(
                model_name,
                best_params,
                train_features.iloc[fit_indices],
                train_labels[fit_indices],
                train_features.iloc[validation_indices],
                selected_fold,
                variable_types,
                bool(spec["scale_numeric"]),
                HEAVY_SEED + 530000 + model_offset * 100 + inner_fold,
            )
        if np.isnan(oof_raw).any():
            raise RuntimeError(f"Incomplete OOF predictions for {model_name}.")
        calibrator = _fit_calibrator(train_labels, oof_raw)
        oof_calibrated = _apply_calibrator(calibrator, oof_raw)
        selected_features = full_ranking[:k]
        internal_raw, external_raw = _fit_predict_many(
            model_name,
            best_params,
            train_features,
            train_labels,
            [internal_features, external_features],
            selected_features,
            variable_types,
            bool(spec["scale_numeric"]),
            HEAVY_SEED + 540000 + model_offset,
        )
        internal_probabilities = _apply_calibrator(calibrator, internal_raw)
        external_probabilities = _apply_calibrator(calibrator, external_raw)
        thresholds = {"youden": _youden_threshold(train_labels, oof_calibrated)}
        thresholds.update(
            {
                f"target_sensitivity_{int(target * 100)}": (
                    _target_sensitivity_threshold(
                        train_labels, oof_calibrated, target
                    )
                )
                for target in TARGET_SENSITIVITIES
            }
        )
        threshold_rows.extend(
            {
                "model_name": model_name,
                "operating_point": name,
                "threshold": threshold,
                "source": "Development train-80 cross-fitted calibrated probabilities",
            }
            for name, threshold in thresholds.items()
        )
        evaluation_rows.extend(
            _bootstrap_evaluation_rows(
                "internal_test_20_locked",
                model_name,
                internal_labels,
                internal_probabilities,
                thresholds,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + 550000 + model_offset,
            )
        )
        evaluation_rows.extend(
            _bootstrap_evaluation_rows(
                "external_validation_locked",
                model_name,
                external_labels,
                external_probabilities,
                thresholds,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + 560000 + model_offset,
            )
        )
        config_rows.append(
            {
                "model_name": model_name,
                "family": spec["family"],
                "selected_by_nested_cv_for_primary_analysis": (
                    model_name == selected_model_name
                ),
                "k": k,
                "params_json": best["params_json"],
                "selected_features": "|".join(selected_features),
                "calibration_intercept_a": float(calibrator.intercept_[0]),
                "calibration_slope_b": float(calibrator.coef_[0, 0]),
                "calibration_source": "Development train-80 OOF only",
            }
        )
        prediction_cache[model_name] = {
            "train_oof": oof_calibrated,
            "internal": internal_probabilities,
            "external": external_probabilities,
            "threshold_85": np.asarray(
                [thresholds["target_sensitivity_85"]], dtype=float
            ),
        }

    selected_predictions = prediction_cache[selected_model_name]
    locked_external_probabilities = selected_predictions["external"]
    development_threshold_85 = float(selected_predictions["threshold_85"][0])
    (
        local_recalibrated_probabilities,
        raw_local_decisions,
        recalibrated_local_decisions,
        local_folds,
    ) = _cross_fitted_local_update(
        external_labels,
        locked_external_probabilities,
    )
    development_decisions = (
        locked_external_probabilities >= development_threshold_85
    )
    three_layer = pd.DataFrame(
        [
            _metric_row(
                selected_model_name,
                "A_locked_model_plus_development_threshold_85",
                "Development-calibrated locked probability",
                "Development train-80 OOF",
                external_labels,
                development_decisions,
                locked_external_probabilities,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + 571001,
            ),
            _metric_row(
                selected_model_name,
                "B_locked_model_plus_local_threshold_85",
                "Development-calibrated locked probability",
                "External 10-fold training subsets",
                external_labels,
                raw_local_decisions,
                locked_external_probabilities,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + 571002,
            ),
            _metric_row(
                selected_model_name,
                "C_locked_model_plus_local_recalibration_and_threshold_85",
                "External 10-fold OOF logistic recalibration",
                "External 10-fold training subsets",
                external_labels,
                recalibrated_local_decisions,
                local_recalibrated_probabilities,
                run_mode.bootstrap_repeats,
                HEAVY_SEED + 571003,
            ),
        ]
    )
    reliability = pd.concat(
        [
            _reliability_bins(
                external_labels,
                locked_external_probabilities,
                selected_model_name,
                "locked_external_probability",
            ),
            _reliability_bins(
                external_labels,
                local_recalibrated_probabilities,
                selected_model_name,
                "external_10fold_oof_local_recalibration",
            ),
        ],
        ignore_index=True,
    )
    dca = _dca_rows(
        external_labels,
        {
            "locked_external_probability_model": locked_external_probabilities,
            "local_recalibrated_oof_probability_model": (
                local_recalibrated_probabilities
            ),
            "development_threshold_85_binary_rule": development_decisions,
            "local_threshold_85_oof_binary_rule": recalibrated_local_decisions,
        },
        run_mode.bootstrap_repeats,
        HEAVY_SEED + 570000,
    )
    dca.insert(0, "model_name", selected_model_name)
    dca.insert(0, "partition", "external_validation_and_local_update_as_labeled")

    final_tuning = pd.DataFrame(tuning_rows)
    final_configs = pd.DataFrame(config_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    status = (
        "smoke_test_only_not_for_manuscript"
        if smoke
        else "manuscript_grade_locked_evaluation_completed"
    )
    manifest = {
        "status": status,
        "selected_model_name": selected_model_name,
        "selection_source": str(selection_output_dir / "model_selection.json"),
        "selection_partition": "Development train-80 nested CV only",
        "final_tuning_partition": "Development train-80 only",
        "calibration_partition": "Development train-80 OOF only",
        "threshold_partition": "Development train-80 OOF only",
        "internal_test_role": "single locked evaluation",
        "external_role": (
            "primary locked validation followed by separately labeled 10-fold OOF "
            "local updating"
        ),
        "bootstrap_repeats": run_mode.bootstrap_repeats,
        "models_evaluated": list(spaces),
        "tabpfn_included": "tabpfn" in spaces,
        "cuda_available": _cuda_available(),
        "participant_level_predictions_written": False,
        "full_external_base_model_refit": False,
        "pending_after_this_stage": [
            "selected-model subgroup analyses with interaction tests",
            "MICE and complete-case sensitivity analyses",
            "selected-model SHAP or TabPFN-native interpretability",
        ],
    }
    final_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(final_tuning, final_output_dir / "final_training_tuning.csv")
    write_csv(final_configs, final_output_dir / "final_model_configs.csv")
    write_csv(thresholds_frame, final_output_dir / "development_thresholds.csv")
    write_csv(evaluation, final_output_dir / "locked_evaluation_metrics.csv")
    write_csv(local_folds, final_output_dir / "external_local_update_folds.csv")
    write_csv(three_layer, final_output_dir / "three_layer_summary.csv")
    write_csv(reliability, final_output_dir / "external_reliability_bins.csv")
    write_csv(dca, final_output_dir / "external_dca.csv")
    (final_output_dir / "final_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "final_tuning": final_tuning,
        "final_configs": final_configs,
        "thresholds": thresholds_frame,
        "evaluation": evaluation,
        "local_folds": local_folds,
        "three_layer": three_layer,
        "reliability": reliability,
        "dca": dca,
        "manifest": manifest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Development-trained MCI evaluation after nested selection."
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-tabpfn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_heavy_final_evaluation(
        args.development,
        args.external,
        args.qc_output,
        args.selection_output,
        args.output,
        smoke=args.smoke,
        skip_tabpfn=args.skip_tabpfn,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
