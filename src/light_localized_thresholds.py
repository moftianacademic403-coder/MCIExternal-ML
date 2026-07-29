from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from light_modeling import LIGHT_SEED
from light_operating_points_recalibration import (
    LIGHT_BOOTSTRAP_REPEATS,
    RECALIBRATION_FOLDS,
    SCENARIO_PREVALENCES,
    TARGET_SENSITIVITIES,
    _logit,
    _new_recalibrator,
    _predict_bundle,
    _predictive_values,
    _target_sensitivity_threshold,
)
from mci_qc import write_csv
from split_development import create_locked_split


def _decision_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    decisions = decisions.astype(bool)
    true_positive = int(((labels == 1) & decisions).sum())
    false_negative = int(((labels == 1) & ~decisions).sum())
    true_negative = int(((labels == 0) & ~decisions).sum())
    false_positive = int(((labels == 0) & decisions).sum())
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    ppv = true_positive / (true_positive + false_positive)
    npv = true_negative / (true_negative + false_negative)
    accuracy = (true_positive + true_negative) / len(labels)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def _bootstrap_decision_intervals(
    labels: np.ndarray,
    decisions: np.ndarray,
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
        values = _decision_metrics(sampled_labels, decisions[indices])
        for metric, value in values.items():
            draws.setdefault(metric, []).append(value)
        completed += 1
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in draws.items()
    }


def _bootstrap_scenario_intervals(
    labels: np.ndarray,
    decisions: np.ndarray,
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
        values = _decision_metrics(sampled_labels, decisions[indices])
        ppv, npv = _predictive_values(
            values["sensitivity"],
            values["specificity"],
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


def _brier_decomposition(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "observed": labels,
            "predicted": probabilities,
            "rank": pd.Series(probabilities).rank(method="first"),
        }
    )
    frame["bin"] = pd.qcut(
        frame["rank"],
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    grouped = frame.groupby("bin", observed=True).agg(
        rows=("observed", "size"),
        observed_rate=("observed", "mean"),
        mean_predicted=("predicted", "mean"),
    )
    weights = grouped["rows"].to_numpy(dtype=float) / len(frame)
    prevalence = float(np.mean(labels))
    reliability = float(
        np.sum(
            weights
            * (grouped["mean_predicted"] - grouped["observed_rate"]) ** 2
        )
    )
    resolution = float(
        np.sum(weights * (grouped["observed_rate"] - prevalence) ** 2)
    )
    uncertainty = float(prevalence * (1 - prevalence))
    refinement = float(uncertainty - resolution)
    reconstructed = float(reliability + refinement)
    brier = float(np.mean((probabilities - labels) ** 2))
    return {
        "brier_score": brier,
        "reliability_calibration_loss": reliability,
        "resolution_discrimination_gain": resolution,
        "uncertainty": uncertainty,
        "refinement_loss": refinement,
        "reconstructed_brier": reconstructed,
        "binning_residual": float(brier - reconstructed),
    }


def _bootstrap_brier_decomposition(
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
        values = _brier_decomposition(
            sampled_labels,
            probabilities[indices],
        )
        for metric, value in values.items():
            draws.setdefault(metric, []).append(value)
        completed += 1
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in draws.items()
    }


def _cross_fitted_localization(
    labels: np.ndarray,
    locked_probabilities: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[str, float], np.ndarray], list[dict]]:
    logits = _logit(locked_probabilities)
    cv = StratifiedKFold(
        n_splits=RECALIBRATION_FOLDS,
        shuffle=True,
        random_state=LIGHT_SEED + 97,
    )
    recalibrated_probabilities = np.full(len(labels), np.nan, dtype=float)
    decisions = {
        (source, target): np.full(len(labels), False, dtype=bool)
        for source in ("raw_local_threshold", "recalibrated_local_threshold")
        for target in TARGET_SENSITIVITIES
    }
    fold_rows = []
    for fold, (fit_indices, validation_indices) in enumerate(
        cv.split(logits, labels), start=1
    ):
        recalibrator = _new_recalibrator()
        recalibrator.fit(logits[fit_indices], labels[fit_indices])
        fit_recalibrated = recalibrator.predict_proba(logits[fit_indices])[:, 1]
        validation_recalibrated = recalibrator.predict_proba(
            logits[validation_indices]
        )[:, 1]
        recalibrated_probabilities[validation_indices] = validation_recalibrated
        for target in TARGET_SENSITIVITIES:
            source_pairs = (
                (
                    "raw_local_threshold",
                    locked_probabilities[fit_indices],
                    locked_probabilities[validation_indices],
                ),
                (
                    "recalibrated_local_threshold",
                    fit_recalibrated,
                    validation_recalibrated,
                ),
            )
            for source, fit_probabilities, validation_probabilities in source_pairs:
                threshold = _target_sensitivity_threshold(
                    labels[fit_indices],
                    fit_probabilities,
                    target,
                )
                fit_decisions = fit_probabilities >= threshold
                validation_decisions = validation_probabilities >= threshold
                decisions[(source, target)][validation_indices] = validation_decisions
                fit_metrics = _decision_metrics(labels[fit_indices], fit_decisions)
                validation_metrics = _decision_metrics(
                    labels[validation_indices],
                    validation_decisions,
                )
                fold_rows.append(
                    {
                        "fold": fold,
                        "probability_source": source,
                        "target_sensitivity": target,
                        "threshold": float(threshold),
                        "recalibration_intercept_a": float(
                            recalibrator.intercept_[0]
                        ),
                        "recalibration_slope_b": float(recalibrator.coef_[0, 0]),
                        "fit_rows": int(len(fit_indices)),
                        "validation_rows": int(len(validation_indices)),
                        "fit_sensitivity": fit_metrics["sensitivity"],
                        "fit_specificity": fit_metrics["specificity"],
                        "validation_sensitivity": validation_metrics["sensitivity"],
                        "validation_specificity": validation_metrics["specificity"],
                    }
                )
    if np.isnan(recalibrated_probabilities).any():
        raise RuntimeError("Cross-fitted recalibrated probabilities are incomplete.")
    return recalibrated_probabilities, decisions, fold_rows


def run_light_localized_thresholds(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)

    metric_rows = []
    fold_rows = []
    scenario_rows = []
    decomposition_rows = []
    agreement_rows = []
    for model_path in sorted(
        (light_output_dir / "calibrated_models").glob("*_platt.joblib")
    ):
        bundle = joblib.load(model_path)
        labels, locked_probabilities = _predict_bundle(bundle, external)
        recalibrated_probabilities, decisions, model_fold_rows = (
            _cross_fitted_localization(labels, locked_probabilities)
        )
        fold_rows.extend(
            {"model_name": bundle["model_name"], **row}
            for row in model_fold_rows
        )
        for probability_source in (
            "raw_local_threshold",
            "recalibrated_local_threshold",
        ):
            for target in TARGET_SENSITIVITIES:
                current_decisions = decisions[(probability_source, target)]
                values = _decision_metrics(labels, current_decisions)
                intervals = _bootstrap_decision_intervals(
                    labels,
                    current_decisions,
                    LIGHT_BOOTSTRAP_REPEATS,
                    LIGHT_SEED
                    + sum(ord(char) for char in bundle["model_name"] + probability_source)
                    + int(target * 1000),
                )
                for metric, estimate in values.items():
                    lower, upper = intervals[metric]
                    metric_rows.append(
                        {
                            "partition": "external_local_updating_10fold_oof",
                            "model_name": bundle["model_name"],
                            "probability_source": probability_source,
                            "target_sensitivity": target,
                            "metric": metric,
                            "estimate": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "ci_method": (
                                f"percentile bootstrap of OOF decisions, "
                                f"{LIGHT_BOOTSTRAP_REPEATS} repeats"
                            ),
                        }
                    )
                if probability_source == "recalibrated_local_threshold":
                    for prevalence in SCENARIO_PREVALENCES:
                        ppv, npv = _predictive_values(
                            values["sensitivity"],
                            values["specificity"],
                            prevalence,
                        )
                        scenario_intervals = _bootstrap_scenario_intervals(
                            labels,
                            current_decisions,
                            prevalence,
                            LIGHT_BOOTSTRAP_REPEATS,
                            LIGHT_SEED
                            + sum(ord(char) for char in bundle["model_name"])
                            + int(target * 1000)
                            + int(prevalence * 10000),
                        )
                        for metric, estimate in (("ppv", ppv), ("npv", npv)):
                            lower, upper = scenario_intervals[metric]
                            scenario_rows.append(
                                {
                                    "model_name": bundle["model_name"],
                                    "target_sensitivity": target,
                                    "assumed_prevalence": prevalence,
                                    "metric": metric,
                                    "estimate": estimate,
                                    "ci_lower": lower,
                                    "ci_upper": upper,
                                    "source": (
                                        "External 10-fold OOF local recalibration "
                                        "and local threshold updating"
                                    ),
                                }
                            )
        for target in TARGET_SENSITIVITIES:
            raw_decisions = decisions[("raw_local_threshold", target)]
            recalibrated_decisions = decisions[
                ("recalibrated_local_threshold", target)
            ]
            agreement_rows.append(
                {
                    "model_name": bundle["model_name"],
                    "target_sensitivity": target,
                    "decision_agreement": float(
                        np.mean(raw_decisions == recalibrated_decisions)
                    ),
                    "different_decisions_n": int(
                        np.sum(raw_decisions != recalibrated_decisions)
                    ),
                }
            )

        for analysis, probabilities in (
            ("development_locked_platt", locked_probabilities),
            (
                "external_10fold_oof_logistic_recalibration",
                recalibrated_probabilities,
            ),
        ):
            values = _brier_decomposition(labels, probabilities)
            intervals = _bootstrap_brier_decomposition(
                labels,
                probabilities,
                LIGHT_BOOTSTRAP_REPEATS,
                LIGHT_SEED
                + sum(ord(char) for char in bundle["model_name"] + analysis),
            )
            for component, estimate in values.items():
                lower, upper = intervals[component]
                decomposition_rows.append(
                    {
                        "partition": "external_validation",
                        "model_name": bundle["model_name"],
                        "analysis": analysis,
                        "component": component,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "method": "Murphy decomposition with 10 equal-frequency bins",
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    prevalence_scenarios = pd.DataFrame(scenario_rows)
    brier_decomposition = pd.DataFrame(decomposition_rows)
    decision_agreement = pd.DataFrame(agreement_rows)
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "external_role": "secondary local updating analysis only",
        "cross_fitting": (
            f"{RECALIBRATION_FOLDS}-fold; recalibrator and target-sensitivity "
            "threshold fitted on each fold's training subset and evaluated on "
            "its held-out subset"
        ),
        "target_sensitivities": list(TARGET_SENSITIVITIES),
        "thresholds_vary_by_fold": True,
        "raw_vs_recalibrated_local_threshold_comparison": (
            "tests whether monotonic recalibration itself changes classification "
            "after reselecting the same target-sensitivity threshold"
        ),
        "scenario_prevalences": list(SCENARIO_PREVALENCES),
        "brier_decomposition": (
            "Murphy reliability-resolution-uncertainty decomposition with 10 "
            "equal-frequency bins; approximate for continuous probabilities"
        ),
        "bootstrap_repeats": LIGHT_BOOTSTRAP_REPEATS,
        "participant_level_predictions_written": False,
        "final_model_selected_from_external": False,
    }
    write_csv(
        metrics,
        light_output_dir / "light_external_localized_threshold_metrics.csv",
    )
    write_csv(
        folds,
        light_output_dir / "light_external_localized_threshold_folds.csv",
    )
    write_csv(
        prevalence_scenarios,
        light_output_dir / "light_external_localized_prevalence_scenarios.csv",
    )
    write_csv(
        brier_decomposition,
        light_output_dir / "light_external_brier_decomposition.csv",
    )
    write_csv(
        decision_agreement,
        light_output_dir / "light_external_raw_recalibrated_decision_agreement.csv",
    )
    (
        light_output_dir / "light_external_localized_threshold_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "folds": folds,
        "prevalence_scenarios": prevalence_scenarios,
        "brier_decomposition": brier_decomposition,
        "decision_agreement": decision_agreement,
        "manifest": manifest,
    }

