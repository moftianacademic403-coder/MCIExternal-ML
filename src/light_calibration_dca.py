from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from light_modeling import INNER_FOLDS, LIGHT_SEED, model_ready_frame
from mci_qc import write_csv
from split_development import create_locked_split


DCA_THRESHOLDS = tuple(np.round(np.arange(0.05, 0.81, 0.05), 2))


def _calibration_statistics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    recalibration = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    recalibration.fit(logits, labels)
    bins = pd.qcut(
        pd.Series(probabilities).rank(method="first"),
        q=min(10, len(probabilities)),
        labels=False,
        duplicates="drop",
    )
    bin_frame = pd.DataFrame(
        {"bin": bins, "observed": labels, "predicted": probabilities}
    ).groupby("bin", observed=True).agg(
        rows=("observed", "size"),
        observed_rate=("observed", "mean"),
        mean_predicted=("predicted", "mean"),
    )
    ece = float(
        np.average(
            np.abs(bin_frame["observed_rate"] - bin_frame["mean_predicted"]),
            weights=bin_frame["rows"],
        )
    )
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "calibration_intercept": float(recalibration.intercept_[0]),
        "calibration_slope": float(recalibration.coef_[0, 0]),
        "mean_predicted_risk": float(np.mean(probabilities)),
        "observed_prevalence": float(np.mean(labels)),
        "expected_calibration_error_10_bins": ece,
    }


def _calibration_bins(
    partition: str,
    model_name: str,
    calibration: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    bin_ids = pd.qcut(
        pd.Series(probabilities).rank(method="first"),
        q=min(10, len(probabilities)),
        labels=False,
        duplicates="drop",
    )
    frame = pd.DataFrame(
        {"bin": bin_ids, "observed": labels, "predicted": probabilities}
    ).groupby("bin", observed=True).agg(
        rows=("observed", "size"),
        observed_rate=("observed", "mean"),
        mean_predicted=("predicted", "mean"),
        min_predicted=("predicted", "min"),
        max_predicted=("predicted", "max"),
    ).reset_index()
    return [
        {
            "partition": partition,
            "model_name": model_name,
            "calibration": calibration,
            **row._asdict(),
        }
        for row in frame.itertuples(index=False)
    ]


def _dca_rows(
    partition: str,
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    rows = []
    sample_size = len(labels)
    prevalence = float(labels.mean())
    for threshold in DCA_THRESHOLDS:
        predicted_positive = probabilities >= threshold
        true_positive = int((predicted_positive & (labels == 1)).sum())
        false_positive = int((predicted_positive & (labels == 0)).sum())
        odds = threshold / (1 - threshold)
        net_benefit = true_positive / sample_size - false_positive / sample_size * odds
        treat_all = prevalence - (1 - prevalence) * odds
        rows.extend(
            [
                {
                    "partition": partition,
                    "model_name": model_name,
                    "strategy": "platt_calibrated_model",
                    "threshold_probability": threshold,
                    "net_benefit": float(net_benefit),
                },
                {
                    "partition": partition,
                    "model_name": model_name,
                    "strategy": "screen_all",
                    "threshold_probability": threshold,
                    "net_benefit": float(treat_all),
                },
                {
                    "partition": partition,
                    "model_name": model_name,
                    "strategy": "screen_none",
                    "threshold_probability": threshold,
                    "net_benefit": 0.0,
                },
            ]
        )
    return rows


def run_light_calibration_dca(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    development = split["development_eligible_in_memory"]
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    internal_test = development.iloc[split["test_relative_indices"]].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    train_labels = train["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)

    calibration_rows = []
    bin_rows = []
    dca_rows = []
    calibrated_dir = light_output_dir / "calibrated_models"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    for model_path in sorted((light_output_dir / "models").glob("*.joblib")):
        bundle = joblib.load(model_path)
        selected = bundle["selected_features"]
        train_features = model_ready_frame(train, selected, bundle["variable_types"])
        calibration_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=LIGHT_SEED
        )
        calibrated = CalibratedClassifierCV(
            estimator=clone(bundle["pipeline"]),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=False,
        )
        calibrated.fit(train_features[selected], train_labels)
        calibrated_bundle = {
            **{key: value for key, value in bundle.items() if key != "pipeline"},
            "pipeline": calibrated,
            "calibration_method": "sigmoid_platt",
            "calibration_source": "Development train_80 only, 3-fold CV",
        }
        joblib.dump(
            calibrated_bundle,
            calibrated_dir / f"{bundle['model_name']}_platt.joblib",
        )

        for partition_name, frame in (
            ("internal_test_20_locked", internal_test),
            ("external_validation", external),
        ):
            valid = frame["mci"].isin(["yes", "no"])
            analysis = frame.loc[valid].reset_index(drop=True)
            labels = analysis["mci"].map({"no": 0, "yes": 1}).to_numpy(dtype=int)
            features = model_ready_frame(analysis, selected, bundle["variable_types"])
            uncalibrated_probabilities = bundle["pipeline"].predict_proba(
                features[selected]
            )[:, 1]
            calibrated_probabilities = calibrated.predict_proba(features[selected])[:, 1]
            for calibration_name, probabilities in (
                ("uncalibrated", uncalibrated_probabilities),
                ("sigmoid_platt", calibrated_probabilities),
            ):
                statistics = _calibration_statistics(labels, probabilities)
                calibration_rows.extend(
                    {
                        "partition": partition_name,
                        "model_name": bundle["model_name"],
                        "calibration": calibration_name,
                        "metric": metric,
                        "estimate": estimate,
                    }
                    for metric, estimate in statistics.items()
                )
                bin_rows.extend(
                    _calibration_bins(
                        partition_name,
                        bundle["model_name"],
                        calibration_name,
                        labels,
                        probabilities,
                    )
                )
            dca_rows.extend(
                _dca_rows(
                    partition_name,
                    bundle["model_name"],
                    labels,
                    calibrated_probabilities,
                )
            )

    calibration_metrics = pd.DataFrame(calibration_rows)
    calibration_bins = pd.DataFrame(bin_rows)
    dca = pd.DataFrame(dca_rows)
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "calibration_method": "sigmoid (Platt scaling), prespecified",
        "calibration_fit_partition": "Development train_80 only",
        "calibration_cv": f"{INNER_FOLDS}-fold StratifiedKFold",
        "internal_test_used_to_fit_calibration": False,
        "external_used_to_fit_calibration": False,
        "dca_threshold_probabilities": list(DCA_THRESHOLDS),
        "dca_prediction_source": "Development-trained Platt calibrated models",
        "participant_level_predictions_written": False,
    }
    write_csv(
        calibration_metrics,
        light_output_dir / "light_calibration_metrics.csv",
    )
    write_csv(calibration_bins, light_output_dir / "light_calibration_bins.csv")
    write_csv(dca, light_output_dir / "light_dca.csv")
    (light_output_dir / "light_calibration_dca_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "calibration_metrics": calibration_metrics,
        "calibration_bins": calibration_bins,
        "dca": dca,
        "manifest": manifest,
    }

