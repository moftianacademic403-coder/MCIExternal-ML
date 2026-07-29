from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mci_qc import write_csv
from split_development import create_locked_split


EXTREME_IQR_MULTIPLIER = 3.0
ROBUST_Z_THRESHOLD = 3.5


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _training_thresholds(series: pd.Series) -> dict[str, float | int | str]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return {
            "nonmissing_n": 0,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "extreme_iqr_lower": np.nan,
            "extreme_iqr_upper": np.nan,
            "mad_raw": np.nan,
            "mad_scaled": np.nan,
            "p01": np.nan,
            "p99": np.nan,
            "min": np.nan,
            "max": np.nan,
            "sample_skewness": np.nan,
            "upper_tail_gap_ratio": np.nan,
            "lower_tail_gap_ratio": np.nan,
        }
    q1 = float(values.quantile(0.25))
    median = float(values.median())
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    mad_raw = float((values - median).abs().median())
    mad_scaled = 1.4826 * mad_raw
    p01 = float(values.quantile(0.01))
    p99 = float(values.quantile(0.99))
    minimum = float(values.min())
    maximum = float(values.max())
    central_98_width = p99 - p01
    return {
        "nonmissing_n": int(len(values)),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "extreme_iqr_lower": q1 - EXTREME_IQR_MULTIPLIER * iqr,
        "extreme_iqr_upper": q3 + EXTREME_IQR_MULTIPLIER * iqr,
        "mad_raw": mad_raw,
        "mad_scaled": mad_scaled,
        "p01": p01,
        "p99": p99,
        "min": minimum,
        "max": maximum,
        "sample_skewness": float(values.skew()),
        "upper_tail_gap_ratio": _safe_ratio(maximum - p99, central_98_width),
        "lower_tail_gap_ratio": _safe_ratio(p01 - minimum, central_98_width),
    }


def _apply_thresholds(
    series: pd.Series,
    thresholds: dict[str, float | int | str],
) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce")
    observed = values.dropna().astype(float)
    if observed.empty:
        return {
            "rows": int(len(values)),
            "missing_n": int(values.isna().sum()),
            "nonmissing_n": 0,
            "below_extreme_iqr_n": 0,
            "above_extreme_iqr_n": 0,
            "robust_z_flag_n": 0,
            "outside_train_range_n": 0,
            "zero_n": 0,
            "negative_n": 0,
            "extreme_iqr_flag_pct_nonmissing": np.nan,
            "robust_z_flag_pct_nonmissing": np.nan,
        }
    lower = float(thresholds["extreme_iqr_lower"])
    upper = float(thresholds["extreme_iqr_upper"])
    below = int((observed < lower).sum()) if np.isfinite(lower) else 0
    above = int((observed > upper).sum()) if np.isfinite(upper) else 0
    mad_scaled = float(thresholds["mad_scaled"])
    if np.isfinite(mad_scaled) and mad_scaled > 0:
        robust_z = (observed - float(thresholds["median"])).abs() / mad_scaled
        robust_n = int((robust_z > ROBUST_Z_THRESHOLD).sum())
    else:
        robust_n = 0
    train_min = float(thresholds["min"])
    train_max = float(thresholds["max"])
    outside_train = int(((observed < train_min) | (observed > train_max)).sum())
    return {
        "rows": int(len(values)),
        "missing_n": int(values.isna().sum()),
        "nonmissing_n": int(len(observed)),
        "below_extreme_iqr_n": below,
        "above_extreme_iqr_n": above,
        "robust_z_flag_n": robust_n,
        "outside_train_range_n": outside_train,
        "zero_n": int(observed.eq(0).sum()),
        "negative_n": int(observed.lt(0).sum()),
        "extreme_iqr_flag_pct_nonmissing": float(100 * (below + above) / len(observed)),
        "robust_z_flag_pct_nonmissing": float(100 * robust_n / len(observed)),
    }


def run_outlier_audit(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    registry = split["registry"]
    numeric_features = registry.loc[
        (registry["role"] == "predictor") & (registry["variable_type"] == "numeric"),
        "canonical_name",
    ].tolist()
    development = split["development_eligible_in_memory"]
    train = development.iloc[split["train_relative_indices"]].reset_index(drop=True)
    internal_test = development.iloc[split["test_relative_indices"]].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)

    threshold_rows: list[dict] = []
    partition_rows: list[dict] = []
    for feature in numeric_features:
        thresholds = {"canonical_name": feature, **_training_thresholds(train[feature])}
        train_flags = _apply_thresholds(train[feature], thresholds)
        thresholds.update(
            {
                "train_extreme_iqr_flag_n": train_flags["below_extreme_iqr_n"]
                + train_flags["above_extreme_iqr_n"],
                "train_extreme_iqr_flag_pct_nonmissing": train_flags[
                    "extreme_iqr_flag_pct_nonmissing"
                ],
                "train_robust_z_flag_n": train_flags["robust_z_flag_n"],
                "train_robust_z_flag_pct_nonmissing": train_flags[
                    "robust_z_flag_pct_nonmissing"
                ],
                "automatic_action": "none_review_only",
            }
        )
        threshold_rows.append(thresholds)
        for partition_name, frame in (
            ("train_80", train),
            ("internal_test_20_locked", internal_test),
            ("external_reserved_predictors_only", external),
        ):
            partition_rows.append(
                {
                    "partition": partition_name,
                    "canonical_name": feature,
                    **_apply_thresholds(frame[feature], thresholds),
                }
            )

    thresholds_frame = pd.DataFrame(threshold_rows).sort_values(
        ["upper_tail_gap_ratio", "train_robust_z_flag_pct_nonmissing"],
        ascending=[False, False],
        na_position="last",
    )
    partition_frame = pd.DataFrame(partition_rows)

    policy = {
        "status": "lightweight_preliminary",
        "source_partition_for_thresholds": "train_80_only",
        "internal_test_used_to_define_rules": False,
        "external_used_to_define_rules": False,
        "participant_level_outputs_written": False,
        "outlier_audit": {
            "extreme_iqr_multiplier": EXTREME_IQR_MULTIPLIER,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "purpose": "flag_for_review_only",
            "automatic_deletion": False,
            "automatic_winsorization": False,
            "confirmed_invalid_values": "correct_from_source_or_set_missing",
            "clinically_plausible_extreme_values": "retain",
        },
        "primary_missing_data": {
            "numeric": "training-fold median imputation plus missing indicator when present",
            "categorical": "explicit missing/unknown category",
            "mice": "sensitivity analysis, not primary lightweight pipeline",
        },
        "primary_scaling": {
            "scale_sensitive_models": "training-fold RobustScaler",
            "tree_models": "no scaling",
        },
        "planned_sensitivity": [
            "fold-fitted MICE",
            "fold-fitted percentile winsorization",
            "exclude predictors with severe availability limitations",
        ],
    }

    light_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(thresholds_frame, light_output_dir / "outlier_train_thresholds.csv")
    write_csv(partition_frame, light_output_dir / "outlier_partition_flags.csv")
    (light_output_dir / "preprocessing_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "thresholds": thresholds_frame,
        "partition_flags": partition_frame,
        "policy": policy,
    }

