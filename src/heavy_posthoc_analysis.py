from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.special import expit
from scipy.stats import chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from heavy_final_evaluation import (
    _bootstrap_evaluation_rows,
    _calibration_statistics,
    _cross_fitted_local_update,
    _decision_metrics,
    _fit_calibrator,
    _logit,
    _reliability_bins,
    _stratified_bootstrap_indices,
    _target_sensitivity_threshold,
)
from heavy_nested_cv import (
    HEAVY_SEED,
    _classical_pipeline,
    _fit_predict,
    _fit_predict_many,
    _model_spaces,
    _tabpfn_frame,
)
from light_modeling import model_ready_frame
from mci_qc import write_csv
from mrmr_stability import run_mrmr_stability
from split_development import create_locked_split
from transportability_audit import run_transportability_audit


POSTHOC_SEED = 20260730
INNER_FOLDS = 4
STABILITY_REPEATS = 30


def _saved_calibration(
    raw_probabilities: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return expit(intercept + slope * _logit(raw_probabilities))


def _load_locked_configuration(
    prior_output_dir: Path,
) -> tuple[str, dict[str, Any], list[str], float, float, float]:
    manifest_path = (
        prior_output_dir / "final_evaluation" / "final_evaluation_manifest.json"
    )
    prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prior_manifest.get("status") != "manuscript_grade_locked_evaluation_completed":
        raise RuntimeError(
            "Post-hoc analysis requires a manuscript-grade heavy run; received "
            f"{prior_manifest.get('status')!r}."
        )
    if (
        prior_manifest.get("education_harmonization_mode")
        != "four_level_code_matched_auxiliary_source"
    ):
        raise RuntimeError("Prior heavy run did not use four-level education.")
    if prior_manifest.get("participant_level_predictions_written") is not False:
        raise RuntimeError("Prior heavy-run privacy guard failed.")
    configs = pd.read_csv(
        prior_output_dir / "final_evaluation" / "final_model_configs.csv",
        encoding="utf-8-sig",
    )
    selected_flag = configs["selected_by_nested_cv_for_primary_analysis"]
    if selected_flag.dtype != bool:
        selected_flag = selected_flag.astype("string").str.casefold().eq("true")
    selected = configs.loc[selected_flag]
    if len(selected) != 1:
        raise RuntimeError("Expected exactly one locked primary configuration.")
    row = selected.iloc[0]
    model_name = str(row["model_name"])
    params = json.loads(row["params_json"])
    features = str(row["selected_features"]).split("|")
    thresholds = pd.read_csv(
        prior_output_dir / "final_evaluation" / "development_thresholds.csv",
        encoding="utf-8-sig",
    )
    threshold_row = thresholds.loc[
        thresholds["model_name"].eq(model_name)
        & thresholds["operating_point"].eq("target_sensitivity_85")
    ]
    if len(threshold_row) != 1:
        raise RuntimeError("Locked Development target-sensitivity threshold is missing.")
    return (
        model_name,
        params,
        features,
        float(row["calibration_intercept_a"]),
        float(row["calibration_slope_b"]),
        float(threshold_row.iloc[0]["threshold"]),
    )


def _encoded_feature_owner(
    encoded_name: str,
    predictors: list[str],
) -> str | None:
    suffix = encoded_name.split("__", 1)[-1]
    for feature in sorted(predictors, key=len, reverse=True):
        if suffix == feature or suffix == f"missingindicator_{feature}":
            return feature
        if suffix.startswith(f"{feature}_"):
            return feature
    return None


def _elastic_net_stability(
    train: pd.DataFrame,
    labels: np.ndarray,
    predictors: list[str],
    variable_types: dict[str, str],
) -> pd.DataFrame:
    numeric = [name for name in predictors if variable_types[name] == "numeric"]
    categorical = [
        name for name in predictors if variable_types[name] == "categorical"
    ]
    sampler = StratifiedShuffleSplit(
        n_splits=STABILITY_REPEATS,
        train_size=0.5,
        random_state=POSTHOC_SEED,
    )
    selected_counts = {feature: 0 for feature in predictors}
    magnitudes = {feature: [] for feature in predictors}
    for repeat, (indices, _) in enumerate(sampler.split(train, labels), start=1):
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
                                    strategy="constant", fill_value="__missing__"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        estimator = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegressionCV(
                        Cs=[0.01, 0.03, 0.1, 0.3, 1.0],
                        cv=3,
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=[0.25, 0.5, 0.75],
                        scoring="roc_auc",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=POSTHOC_SEED + repeat,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        estimator.fit(train.iloc[indices], labels[indices])
        encoded_names = estimator.named_steps["preprocessor"].get_feature_names_out()
        coefficients = np.abs(estimator.named_steps["model"].coef_[0])
        grouped = {feature: 0.0 for feature in predictors}
        for encoded_name, coefficient in zip(encoded_names, coefficients):
            owner = _encoded_feature_owner(str(encoded_name), predictors)
            if owner is not None:
                grouped[owner] = max(grouped[owner], float(coefficient))
        for feature, magnitude in grouped.items():
            magnitudes[feature].append(magnitude)
            if magnitude > 1e-8:
                selected_counts[feature] += 1
    rows = [
        {
            "canonical_name": feature,
            "selection_frequency": selected_counts[feature] / STABILITY_REPEATS,
            "median_max_abs_coefficient": float(np.median(magnitudes[feature])),
            "mean_max_abs_coefficient": float(np.mean(magnitudes[feature])),
            "repeats": STABILITY_REPEATS,
            "subsample_fraction": 0.5,
        }
        for feature in predictors
    ]
    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "median_max_abs_coefficient", "canonical_name"],
        ascending=[False, False, True],
    )


def _mrmr_consensus(rank_stability: pd.DataFrame, k: int = 30) -> list[str]:
    frequency_column = f"selection_frequency_top_{k}"
    return (
        rank_stability.sort_values(
            [frequency_column, "median_rank", "mean_rank", "canonical_name"],
            ascending=[False, True, True, True],
        )
        .head(k)["canonical_name"]
        .tolist()
    )


def _winsorize_from_fit(
    fit: pd.DataFrame,
    others: list[pd.DataFrame],
    features: list[str],
    variable_types: dict[str, str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    fit_transformed = fit.copy()
    other_transformed = [frame.copy() for frame in others]
    for feature in features:
        if variable_types[feature] != "numeric":
            continue
        values = pd.to_numeric(fit[feature], errors="coerce")
        lower = float(values.quantile(0.01))
        upper = float(values.quantile(0.99))
        fit_transformed[feature] = values.clip(lower, upper)
        for frame in other_transformed:
            frame[feature] = pd.to_numeric(
                frame[feature], errors="coerce"
            ).clip(lower, upper)
    return fit_transformed, other_transformed


def _evaluate_feature_scenario(
    scenario: str,
    model_name: str,
    train: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    train_labels: np.ndarray,
    internal_labels: np.ndarray,
    external_labels: np.ndarray,
    selected_features: list[str],
    variable_types: dict[str, str],
    params: dict[str, Any],
    scale_numeric: bool,
    bootstrap_repeats: int,
    winsorize: bool = False,
    complete_case: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_current = train.reset_index(drop=True)
    internal_current = internal.reset_index(drop=True)
    external_current = external.reset_index(drop=True)
    train_y = np.asarray(train_labels)
    internal_y = np.asarray(internal_labels)
    external_y = np.asarray(external_labels)
    if complete_case:
        train_mask = train_current[selected_features].notna().all(axis=1).to_numpy()
        internal_mask = internal_current[selected_features].notna().all(axis=1).to_numpy()
        external_mask = external_current[selected_features].notna().all(axis=1).to_numpy()
        train_current, train_y = train_current.loc[train_mask].reset_index(drop=True), train_y[train_mask]
        internal_current, internal_y = internal_current.loc[internal_mask].reset_index(drop=True), internal_y[internal_mask]
        external_current, external_y = external_current.loc[external_mask].reset_index(drop=True), external_y[external_mask]
    if min(np.bincount(train_y, minlength=2)) < INNER_FOLDS:
        raise RuntimeError(f"Too few complete cases for {scenario} cross-fitting.")

    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=POSTHOC_SEED + len(scenario),
    )
    oof_raw = np.full(len(train_current), np.nan, dtype=float)
    for fold, (fit_indices, validation_indices) in enumerate(
        cv.split(train_current, train_y), start=1
    ):
        fit_frame = train_current.iloc[fit_indices]
        validation_frame = train_current.iloc[validation_indices]
        if winsorize:
            fit_frame, [validation_frame] = _winsorize_from_fit(
                fit_frame,
                [validation_frame],
                selected_features,
                variable_types,
            )
        oof_raw[validation_indices] = _fit_predict(
            model_name,
            params,
            fit_frame,
            train_y[fit_indices],
            validation_frame,
            selected_features,
            variable_types,
            scale_numeric,
            POSTHOC_SEED + fold * 100 + len(scenario),
        )
    calibrator = _fit_calibrator(train_y, oof_raw)
    oof_probability = calibrator.predict_proba(
        _logit(oof_raw).reshape(-1, 1)
    )[:, 1]
    threshold = _target_sensitivity_threshold(train_y, oof_probability, 0.85)

    fit_frame = train_current
    validation_frames = [internal_current, external_current]
    if winsorize:
        fit_frame, validation_frames = _winsorize_from_fit(
            fit_frame,
            validation_frames,
            selected_features,
            variable_types,
        )
    internal_raw, external_raw = _fit_predict_many(
        model_name,
        params,
        fit_frame,
        train_y,
        validation_frames,
        selected_features,
        variable_types,
        scale_numeric,
        POSTHOC_SEED + 9000 + len(scenario),
    )
    internal_probability = calibrator.predict_proba(
        _logit(internal_raw).reshape(-1, 1)
    )[:, 1]
    external_probability = calibrator.predict_proba(
        _logit(external_raw).reshape(-1, 1)
    )[:, 1]
    rows = []
    for partition, labels, probabilities in (
        ("internal_test_20_locked", internal_y, internal_probability),
        ("external_posthoc_sensitivity", external_y, external_probability),
    ):
        rows.extend(
            _bootstrap_evaluation_rows(
                partition,
                scenario,
                labels,
                probabilities,
                {"target_sensitivity_85": threshold},
                bootstrap_repeats,
                POSTHOC_SEED + len(rows) + len(scenario),
            )
        )
    metadata = {
        "scenario": scenario,
        "model_name": model_name,
        "features": "|".join(selected_features),
        "feature_count": len(selected_features),
        "development_train_rows": int(len(train_current)),
        "internal_rows": int(len(internal_current)),
        "external_rows": int(len(external_current)),
        "winsorized_training_p01_p99": winsorize,
        "complete_case": complete_case,
        "development_oof_threshold_85": float(threshold),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
    }
    return pd.DataFrame(rows), metadata


def _brier_decomposition(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"label": labels, "probability": probabilities})
    ranked = frame["probability"].rank(method="first")
    frame["bin"] = pd.qcut(ranked, q=min(10, len(frame)), labels=False)
    prevalence = float(frame["label"].mean())
    reliability = 0.0
    resolution = 0.0
    for _, group in frame.groupby("bin", observed=True):
        weight = len(group) / len(frame)
        mean_probability = float(group["probability"].mean())
        observed = float(group["label"].mean())
        reliability += weight * (mean_probability - observed) ** 2
        resolution += weight * (observed - prevalence) ** 2
    uncertainty = prevalence * (1 - prevalence)
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_reconstructed_brier": reliability - resolution + uncertainty,
    }


def _local_calibration_bootstrap(
    labels: np.ndarray,
    locked_probabilities: np.ndarray,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    local_probability, raw_decisions, local_decisions, folds = (
        _cross_fitted_local_update(labels, locked_probabilities)
    )
    estimates = {
        "locked": {
            **_calibration_statistics(labels, locked_probabilities),
            **_brier_decomposition(labels, locked_probabilities),
        },
        "local_oof_recalibrated": {
            **_calibration_statistics(labels, local_probability),
            **_brier_decomposition(labels, local_probability),
        },
    }
    metrics = [
        "calibration_intercept",
        "calibration_slope",
        "brier_score",
        "reliability",
        "resolution",
        "uncertainty",
        "decomposition_reconstructed_brier",
    ]
    draws = {
        analysis: {metric: [] for metric in metrics}
        for analysis in estimates
    }
    bootstrap_indices = _stratified_bootstrap_indices(
        labels, repeats, POSTHOC_SEED + 41000
    )
    for sampled_indices in bootstrap_indices:
        sampled_labels = labels[sampled_indices]
        sampled_locked = locked_probabilities[sampled_indices]
        sampled_local, _, _, _ = _cross_fitted_local_update(
            sampled_labels, sampled_locked
        )
        for analysis, probabilities in (
            ("locked", sampled_locked),
            ("local_oof_recalibrated", sampled_local),
        ):
            current = {
                **_calibration_statistics(sampled_labels, probabilities),
                **_brier_decomposition(sampled_labels, probabilities),
            }
            for metric in metrics:
                draws[analysis][metric].append(float(current[metric]))
    rows = []
    for analysis in estimates:
        for metric in metrics:
            rows.append(
                {
                    "analysis": analysis,
                    "metric": metric,
                    "estimate": float(estimates[analysis][metric]),
                    "ci_lower": float(np.quantile(draws[analysis][metric], 0.025)),
                    "ci_upper": float(np.quantile(draws[analysis][metric], 0.975)),
                    "ci_method": (
                        f"stratified bootstrap with full 10-fold local update, {repeats} repeats"
                        if analysis == "local_oof_recalibrated"
                        else f"stratified participant bootstrap, {repeats} repeats"
                    ),
                }
            )
    return pd.DataFrame(rows), folds, local_probability, local_decisions


def _subgroup_assignments(frame: pd.DataFrame, selected_features: list[str]) -> dict[str, pd.Series]:
    age = pd.to_numeric(frame["age"], errors="coerce")
    missing_count = frame[selected_features].isna().sum(axis=1)
    return {
        "sex": frame["sex"].astype("string").fillna("missing"),
        "age_group": pd.cut(
            age,
            bins=[-np.inf, 65, 75, np.inf],
            right=False,
            labels=["under_65", "65_to_74", "75_plus"],
        ).astype("string").fillna("missing"),
        "selected_predictor_missingness": pd.Series(
            np.where(missing_count.eq(0), "none", "one_or_more"),
            index=frame.index,
            dtype="string",
        ),
    }


def _safe_metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    decisions = probabilities >= threshold
    values = {
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else np.nan,
        "average_precision": float(average_precision_score(labels, probabilities))
        if np.unique(labels).size == 2
        else np.nan,
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    values.update(_decision_metrics(labels, decisions))
    return values


def _interaction_p_value(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: pd.Series,
) -> tuple[float, int]:
    group_dummies = pd.get_dummies(groups.astype("string"), drop_first=True, dtype=float)
    if group_dummies.shape[1] == 0:
        return np.nan, 0
    logit_probability = _logit(probabilities)
    reduced = pd.DataFrame({"prediction_logit": logit_probability})
    reduced = pd.concat([reduced, group_dummies.reset_index(drop=True)], axis=1)
    full = reduced.copy()
    for column in group_dummies.columns:
        full[f"prediction_x_{column}"] = logit_probability * group_dummies[column].to_numpy()
    try:
        reduced_fit = sm.Logit(
            labels, sm.add_constant(reduced, has_constant="add")
        ).fit(disp=False)
        full_fit = sm.Logit(
            labels, sm.add_constant(full, has_constant="add")
        ).fit(disp=False)
    except (np.linalg.LinAlgError, PerfectSeparationError, ValueError):
        return np.nan, full.shape[1] - reduced.shape[1]
    degrees_of_freedom = full.shape[1] - reduced.shape[1]
    statistic = max(0.0, 2.0 * (full_fit.llf - reduced_fit.llf))
    return float(chi2.sf(statistic, degrees_of_freedom)), degrees_of_freedom


def _subgroup_analysis(
    partition: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selected_features: list[str],
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_rows = []
    interaction_rows = []
    for subgroup_name, assignments in _subgroup_assignments(
        frame, selected_features
    ).items():
        interaction_p, interaction_df = _interaction_p_value(
            labels, probabilities, assignments.reset_index(drop=True)
        )
        interaction_rows.append(
            {
                "partition": partition,
                "subgroup": subgroup_name,
                "interaction_test": "likelihood-ratio test for prediction_logit-by-subgroup interaction",
                "interaction_p_value": interaction_p,
                "degrees_of_freedom": interaction_df,
            }
        )
        for level in sorted(assignments.dropna().unique()):
            mask = assignments.eq(level).to_numpy()
            current_labels = labels[mask]
            current_probabilities = probabilities[mask]
            estimates = _safe_metric_values(
                current_labels, current_probabilities, threshold
            )
            bootstrap = _stratified_bootstrap_indices(
                current_labels,
                repeats,
                POSTHOC_SEED + len(performance_rows) + 60000,
            ) if np.unique(current_labels).size == 2 else np.empty((0, 0), dtype=int)
            metric_draws = {metric: [] for metric in estimates}
            for indices in bootstrap:
                values = _safe_metric_values(
                    current_labels[indices],
                    current_probabilities[indices],
                    threshold,
                )
                for metric, value in values.items():
                    metric_draws[metric].append(value)
            for metric, estimate in estimates.items():
                performance_rows.append(
                    {
                        "partition": partition,
                        "subgroup": subgroup_name,
                        "level": str(level),
                        "rows": int(mask.sum()),
                        "events": int(current_labels.sum()),
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": float(np.quantile(metric_draws[metric], 0.025))
                        if metric_draws[metric]
                        else np.nan,
                        "ci_upper": float(np.quantile(metric_draws[metric], 0.975))
                        if metric_draws[metric]
                        else np.nan,
                        "ci_method": f"stratified participant bootstrap, {repeats} repeats",
                    }
                )
    return pd.DataFrame(performance_rows), pd.DataFrame(interaction_rows)


def _plot_calibration(
    labels: np.ndarray,
    locked_probabilities: np.ndarray,
    local_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    locked_bins = _reliability_bins(labels, locked_probabilities, "locked")
    local_bins = _reliability_bins(labels, local_probabilities, "local_oof")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    axis.plot(
        locked_bins["mean_predicted"],
        locked_bins["observed_rate"],
        marker="o",
        label="Locked model",
    )
    axis.plot(
        local_bins["mean_predicted"],
        local_bins["observed_rate"],
        marker="o",
        label="10-fold OOF local recalibration",
    )
    axis.set(xlabel="Predicted probability", ylabel="Observed MCI rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_dca(prior_output_dir: Path, output_path: Path) -> None:
    dca = pd.read_csv(
        prior_output_dir / "final_evaluation" / "external_dca.csv",
        encoding="utf-8-sig",
    )
    strategies = [
        "locked_external_probability_model",
        "local_recalibrated_oof_probability_model",
        "screen_all",
        "screen_none",
    ]
    figure, axis = plt.subplots(figsize=(8, 6))
    for strategy in strategies:
        current = dca.loc[dca["strategy"].eq(strategy)]
        axis.plot(
            current["threshold_probability"],
            current["net_benefit"],
            label=strategy.replace("_", " "),
        )
        if strategy not in {"screen_all", "screen_none"}:
            axis.fill_between(
                current["threshold_probability"],
                current["ci_lower"],
                current["ci_upper"],
                alpha=0.12,
            )
    axis.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.80))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _tabpfn_shap(
    train: pd.DataFrame,
    train_labels: np.ndarray,
    internal: pd.DataFrame,
    internal_labels: np.ndarray,
    selected_features: list[str],
    variable_types: dict[str, str],
    params: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq
    from tabpfn_extensions.interpretability import shapiq_to_shap_explanation
    import shap

    train_prepared = _tabpfn_frame(train, selected_features, variable_types)
    internal_prepared = _tabpfn_frame(internal, selected_features, variable_types)
    categorical_indices = [
        index
        for index, feature in enumerate(selected_features)
        if variable_types[feature] == "categorical"
    ]
    classifier = TabPFNClassifier(
        n_estimators=int(params["n_estimators"]),
        categorical_features_indices=categorical_indices,
        device="cuda",
        random_state=POSTHOC_SEED,
        show_progress_bar=False,
        fit_mode="fit_with_cache",
    )
    classifier.fit(train_prepared, train_labels)
    rng = np.random.default_rng(POSTHOC_SEED)
    explain_indices = np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(internal_labels == label), size=25, replace=False)
                for label in (0, 1)
            ]
        )
    )
    explain_frame = internal_prepared.iloc[explain_indices]
    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=classifier,
        data=train_prepared,
        index="SV",
        max_order=1,
    )
    explanation = shapiq_to_shap_explanation(
        explainer,
        explain_frame,
        budget=256,
        feature_names=selected_features,
    )
    values = np.asarray(explanation.values, dtype=float)
    importance = pd.DataFrame(
        {
            "canonical_name": selected_features,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
            "explained_internal_rows": len(explain_indices),
            "budget_per_row": 256,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    write_csv(importance, output_dir / "selected_model_shap_global_importance.csv")
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_model_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    del classifier, explainer, explanation
    gc.collect()
    return {
        "status": "completed",
        "model_name": "tabpfn",
        "method": "TabPFN shapiq imputation explainer converted to shap.Explanation",
        "scope": "50 stratified rows from the locked internal test only",
        "budget_per_row": 256,
        "warning": "Experimental TabPFN extension; SHAP values describe associations, not causality.",
    }


def _normalize_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP value shape: {array.shape}")
    return array


def _classical_shap(
    model_name: str,
    train: pd.DataFrame,
    train_labels: np.ndarray,
    internal: pd.DataFrame,
    internal_labels: np.ndarray,
    selected_features: list[str],
    variable_types: dict[str, str],
    params: dict[str, Any],
    scale_numeric: bool,
    output_dir: Path,
) -> dict[str, Any]:
    import shap

    pipeline = _classical_pipeline(
        model_name,
        params,
        selected_features,
        variable_types,
        scale_numeric,
        POSTHOC_SEED,
    )
    pipeline.fit(train[selected_features], train_labels)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    encoded_train = np.asarray(preprocessor.transform(train[selected_features]), dtype=float)
    encoded_internal = np.asarray(preprocessor.transform(internal[selected_features]), dtype=float)
    encoded_names = [str(name) for name in preprocessor.get_feature_names_out()]
    rng = np.random.default_rng(POSTHOC_SEED)
    explain_indices = np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(internal_labels == label),
                    size=min(25, int(np.sum(internal_labels == label))),
                    replace=False,
                )
                for label in (0, 1)
            ]
        )
    )
    background_indices = rng.choice(
        np.arange(len(encoded_train)),
        size=min(100, len(encoded_train)),
        replace=False,
    )
    background = encoded_train[background_indices]
    explain_matrix = encoded_internal[explain_indices]
    if model_name in {"random_forest", "xgboost"}:
        explainer = shap.TreeExplainer(estimator, data=background)
        values = _normalize_shap_values(explainer.shap_values(explain_matrix))
        method = "TreeSHAP on the fitted selected-model estimator"
    elif model_name == "elastic_net_logistic":
        explainer = shap.LinearExplainer(estimator, background)
        values = _normalize_shap_values(explainer.shap_values(explain_matrix))
        method = "Linear SHAP on the fitted elastic-net logistic estimator"
    else:
        background_summary = shap.kmeans(background, min(25, len(background)))
        explainer = shap.KernelExplainer(
            lambda matrix: estimator.predict_proba(matrix)[:, 1],
            background_summary,
        )
        values = _normalize_shap_values(
            explainer.shap_values(explain_matrix, nsamples=256)
        )
        method = "Kernel SHAP on the fitted RBF-SVM estimator"

    owner_indices: dict[str, list[int]] = {feature: [] for feature in selected_features}
    for index, encoded_name in enumerate(encoded_names):
        owner = _encoded_feature_owner(encoded_name, selected_features)
        if owner is not None:
            owner_indices[owner].append(index)
    aggregated = np.column_stack(
        [
            values[:, indices].sum(axis=1) if indices else np.zeros(len(values))
            for feature, indices in owner_indices.items()
        ]
    )
    importance = pd.DataFrame(
        {
            "canonical_name": selected_features,
            "mean_abs_shap": np.mean(np.abs(aggregated), axis=0),
            "mean_shap": np.mean(aggregated, axis=0),
            "explained_internal_rows": len(explain_indices),
            "budget_per_row": 256 if model_name == "svm_rbf" else np.nan,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    write_csv(importance, output_dir / "selected_model_shap_global_importance.csv")
    top = importance.head(20).sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    axis.barh(top["canonical_name"], top["mean_abs_shap"], color="#2F5D7C")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title(f"Selected-model SHAP importance: {model_name}")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "selected_model_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "status": "completed",
        "model_name": model_name,
        "method": method,
        "scope": f"{len(explain_indices)} stratified rows from the locked internal test only",
        "warning": "SHAP values describe model associations, not causality.",
    }


def run_posthoc_analysis(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    prior_output_dir: Path,
    output_dir: Path,
    external_education_path: Path | None = None,
    bootstrap_repeats: int = 2000,
    skip_shap: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = create_locked_split(
        development_path,
        external_path,
        qc_output_dir,
        external_education_path=external_education_path,
    )
    development = split["development_eligible_in_memory"]
    train_indices = split["train_relative_indices"]
    test_indices = split["test_relative_indices"]
    train = development.iloc[train_indices].reset_index(drop=True)
    internal = development.iloc[test_indices].reset_index(drop=True)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    registry = split["registry"]
    predictors = registry.loc[registry["role"].eq("predictor"), "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train = model_ready_frame(train, predictors, variable_types).assign(
        mci=development.iloc[train_indices]["mci"].reset_index(drop=True)
    )
    internal = model_ready_frame(internal, predictors, variable_types).assign(
        mci=development.iloc[test_indices]["mci"].reset_index(drop=True)
    )
    external_labels_text = split["external_harmonized_in_memory"]["mci"].reset_index(drop=True)
    external = model_ready_frame(external, predictors, variable_types).assign(mci=external_labels_text)
    train_labels = train.pop("mci").map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    internal_labels = internal.pop("mci").map({"no": 0, "yes": 1}).to_numpy(dtype=int)
    external_labels = external.pop("mci").map({"no": 0, "yes": 1}).to_numpy(dtype=int)

    model_name, params, frozen_features, calibration_a, calibration_b, locked_threshold = (
        _load_locked_configuration(prior_output_dir)
    )
    spaces = _model_spaces()
    if model_name not in spaces:
        raise RuntimeError(f"Unsupported selected model family: {model_name}")
    scale_numeric = bool(spaces[model_name]["scale_numeric"])
    internal_raw, external_raw = _fit_predict_many(
        model_name,
        params,
        train,
        train_labels,
        [internal, external],
        frozen_features,
        variable_types,
        scale_numeric,
        POSTHOC_SEED,
    )
    internal_locked = _saved_calibration(internal_raw, calibration_a, calibration_b)
    external_locked = _saved_calibration(external_raw, calibration_a, calibration_b)

    stability_dir = output_dir / "feature_stability"
    stability = run_mrmr_stability(
        development_path,
        external_path,
        stability_dir,
        external_education_path=external_education_path,
    )
    elastic_stability = _elastic_net_stability(
        train, train_labels, predictors, variable_types
    )
    write_csv(
        elastic_stability,
        stability_dir / "elastic_net_stability_selection.csv",
    )
    mrmr_top30 = _mrmr_consensus(stability["rank_stability"], 30)
    elastic_top30 = elastic_stability.head(30)["canonical_name"].tolist()

    transport_dir = output_dir / "transportability"
    transport = run_transportability_audit(
        development_path,
        external_path,
        qc_output_dir,
        transport_dir,
        external_education_path=external_education_path,
    )
    drift = transport["drift"]
    eligible_transport = set(
        drift.loc[
            drift["external_missing_pct"].le(20)
            & drift["drift_priority_score"].le(0.5),
            "canonical_name",
        ]
    )
    transport_top30 = [
        feature for feature in mrmr_top30 if feature in eligible_transport
    ][:30]
    if len(transport_top30) < 10:
        raise RuntimeError("Transportability sensitivity retained fewer than 10 features.")

    scenarios = [
        ("all_43_features", predictors, False, False),
        ("mrmr_30repeat_consensus_top30", mrmr_top30, False, False),
        ("elastic_net_stability_top30", elastic_top30, False, False),
        (
            "exclude_adl_iadl_from_frozen_features",
            [feature for feature in frozen_features if feature not in {"adl", "iadl"}],
            False,
            False,
        ),
        (
            "posthoc_transportable_core",
            transport_top30,
            False,
            False,
        ),
        ("frozen_features_winsorized", frozen_features, True, False),
        ("frozen_features_complete_case", frozen_features, False, True),
    ]
    sensitivity_rows = []
    sensitivity_metadata = []
    for scenario, features, winsorize, complete_case in scenarios:
        print(f"Post-hoc sensitivity: {scenario}", flush=True)
        metrics, metadata = _evaluate_feature_scenario(
            scenario,
            model_name,
            train,
            internal,
            external,
            train_labels,
            internal_labels,
            external_labels,
            features,
            variable_types,
            params,
            scale_numeric,
            bootstrap_repeats,
            winsorize=winsorize,
            complete_case=complete_case,
        )
        sensitivity_rows.append(metrics)
        sensitivity_metadata.append(metadata)
    sensitivity_metrics = pd.concat(sensitivity_rows, ignore_index=True)
    write_csv(sensitivity_metrics, output_dir / "posthoc_sensitivity_metrics.csv")
    write_csv(pd.DataFrame(sensitivity_metadata), output_dir / "posthoc_sensitivity_scenarios.csv")

    calibration_summary, local_folds, external_local, external_local_decisions = (
        _local_calibration_bootstrap(
            external_labels, external_locked, bootstrap_repeats
        )
    )
    write_csv(calibration_summary, output_dir / "local_calibration_and_brier_decomposition.csv")
    write_csv(local_folds, output_dir / "local_update_folds.csv")
    _plot_calibration(
        external_labels,
        external_locked,
        external_local,
        output_dir / "external_calibration_before_after.png",
    )
    _plot_dca(prior_output_dir, output_dir / "external_dca_with_ci.png")

    subgroup_tables = []
    interaction_tables = []
    for partition, frame, labels, probabilities in (
        ("internal_test_20_locked", internal, internal_labels, internal_locked),
        ("external_descriptive", external, external_labels, external_locked),
    ):
        performance, interactions = _subgroup_analysis(
            partition,
            frame,
            labels,
            probabilities,
            locked_threshold,
            frozen_features,
            bootstrap_repeats,
        )
        subgroup_tables.append(performance)
        interaction_tables.append(interactions)
    write_csv(pd.concat(subgroup_tables, ignore_index=True), output_dir / "subgroup_performance.csv")
    write_csv(pd.concat(interaction_tables, ignore_index=True), output_dir / "subgroup_interaction_tests.csv")

    shap_manifest: dict[str, Any]
    if skip_shap:
        shap_manifest = {"status": "skipped_by_flag"}
    elif model_name == "tabpfn":
        shap_manifest = _tabpfn_shap(
            train,
            train_labels,
            internal,
            internal_labels,
            frozen_features,
            variable_types,
            params,
            output_dir,
        )
    else:
        shap_manifest = _classical_shap(
            model_name,
            train,
            train_labels,
            internal,
            internal_labels,
            frozen_features,
            variable_types,
            params,
            scale_numeric,
            output_dir,
        )
    manifest = {
        "status": "posthoc_sensitivity_transportability_and_interpretability_completed",
        "primary_model": model_name,
        "primary_model_changed": False,
        "primary_model_source": "Development train-80 repeated nested CV",
        "external_outcome_used_for_primary_selection": False,
        "external_outcome_used_for_posthoc_reporting": True,
        "transportability_core_uses_external_predictors_but_not_external_outcome": True,
        "transportability_core_role": (
            "post-hoc diagnostic sensitivity only; requires a new external cohort "
            "before any confirmatory claim"
        ),
        "mice_performed": False,
        "mice_reason": "excluded at investigator request",
        "mrmr_stability_repeats": STABILITY_REPEATS,
        "elastic_net_stability_repeats": STABILITY_REPEATS,
        "bootstrap_repeats": bootstrap_repeats,
        "shap": shap_manifest,
        "participant_level_outputs_written": False,
        "education_harmonization_mode": split["manifest"][
            "education_harmonization_mode"
        ],
    }
    (output_dir / "posthoc_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc sensitivity, transportability, subgroup, calibration and SHAP analyses."
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-education", type=Path)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_posthoc_analysis(
        args.development,
        args.external,
        args.qc_output,
        args.prior_output,
        args.output,
        external_education_path=args.external_education,
        bootstrap_repeats=args.bootstrap_repeats,
        skip_shap=args.skip_shap,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
