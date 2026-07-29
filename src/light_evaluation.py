from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from light_modeling import LIGHT_SEED, model_ready_frame
from mci_qc import write_csv
from split_development import create_locked_split


LIGHT_BOOTSTRAP_REPEATS = 300


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
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
        values = _metric_values(sampled_labels, probabilities[indices], threshold)
        for name, value in values.items():
            if np.isfinite(value):
                draws.setdefault(name, []).append(value)
        completed += 1
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in draws.items()
    }


def _evaluate_partition(
    partition_name: str,
    frame: pd.DataFrame,
    model_dir: Path,
) -> tuple[list[dict], list[dict]]:
    valid_outcome = frame["mci"].isin(["yes", "no"])
    analysis = frame.loc[valid_outcome].reset_index(drop=True)
    labels = analysis["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    metric_rows = []
    summary_rows = []
    for model_path in sorted(model_dir.glob("*.joblib")):
        bundle = joblib.load(model_path)
        selected = bundle["selected_features"]
        features = model_ready_frame(analysis, selected, bundle["variable_types"])
        probabilities = bundle["pipeline"].predict_proba(features[selected])[:, 1]
        threshold = float(bundle["threshold_from_training_oof_youden"])
        estimates = _metric_values(labels, probabilities, threshold)
        intervals = _bootstrap_intervals(
            labels,
            probabilities,
            threshold,
            LIGHT_BOOTSTRAP_REPEATS,
            LIGHT_SEED + sum(ord(character) for character in bundle["model_name"] + partition_name),
        )
        for metric, estimate in estimates.items():
            lower, upper = intervals[metric]
            metric_rows.append(
                {
                    "partition": partition_name,
                    "model_name": bundle["model_name"],
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_method": f"percentile bootstrap, {LIGHT_BOOTSTRAP_REPEATS} repeats",
                    "threshold": threshold if metric in {
                        "sensitivity", "specificity", "ppv", "npv",
                        "accuracy", "balanced_accuracy",
                    } else np.nan,
                }
            )
        summary_rows.append(
            {
                "partition": partition_name,
                "model_name": bundle["model_name"],
                "rows": int(len(analysis)),
                "mci_yes_n": int(labels.sum()),
                "mci_no_n": int((1 - labels).sum()),
                "prevalence": float(labels.mean()),
                "k": int(bundle["k"]),
                "threshold_from_training_oof_youden": threshold,
            }
        )
    return metric_rows, summary_rows


def run_light_evaluation(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    development = split["development_eligible_in_memory"]
    internal_test = development.iloc[split["test_relative_indices"]].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    model_dir = light_output_dir / "models"
    metric_rows = []
    summary_rows = []
    for partition_name, frame in (
        ("internal_test_20_locked", internal_test),
        ("external_validation", external),
    ):
        partition_metrics, partition_summary = _evaluate_partition(
            partition_name, frame, model_dir
        )
        metric_rows.extend(partition_metrics)
        summary_rows.extend(partition_summary)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["partition", "metric", "estimate"], ascending=[True, True, False]
    )
    summary = pd.DataFrame(summary_rows)
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "models_locked_before_evaluation": True,
        "threshold_source": "training-only out-of-fold Youden index",
        "internal_test_used_for_tuning": False,
        "external_used_for_tuning": False,
        "bootstrap_repeats": LIGHT_BOOTSTRAP_REPEATS,
        "bootstrap_ci": "percentile 95% CI",
        "participant_level_predictions_written": False,
        "post_evaluation_retuning_allowed": False,
    }
    write_csv(metrics, light_output_dir / "light_evaluation_metrics.csv")
    write_csv(summary, light_output_dir / "light_evaluation_summary.csv")
    (light_output_dir / "light_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "summary": summary, "manifest": manifest}

