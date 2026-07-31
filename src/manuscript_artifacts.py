from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mci_qc import write_csv
from split_development import create_locked_split


BLUE = "#2F5D7C"
ORANGE = "#D98B45"
GOLD = "#C5A13A"
OLIVE = "#788C5D"
PINK = "#B66A7A"
INK = "#263238"
GREY = "#8A9499"
LIGHT_GREY = "#D8DEE2"
MODEL_COLORS = {
    "elastic_net_logistic": BLUE,
    "svm_rbf": ORANGE,
    "random_forest": OLIVE,
    "xgboost": PINK,
    "tabpfn": GOLD,
}
MODEL_LABELS = {
    "elastic_net_logistic": "Elastic-net logistic",
    "svm_rbf": "RBF-SVM",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN",
}
CORE_TABLE1_FEATURES = [
    "age",
    "sex",
    "education",
    "employment_status",
    "housing_status",
    "household_income",
    "adl",
    "iadl",
    "depression_status",
    "sleep_quality",
    "max_handgrip",
    "hypertension",
    "diabetes",
    "current_smoker",
]


def _save_figure(figure: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    tiff = stem.with_suffix(".tiff")
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        tiff,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return [png.name, tiff.name]


def _n_pct(count: int, denominator: int) -> str:
    return f"{count} ({100 * count / denominator:.1f}%)" if denominator else "0 (NA)"


def _numeric_shift(left: pd.Series, right: pd.Series) -> float:
    left_values = pd.to_numeric(left, errors="coerce").dropna()
    right_values = pd.to_numeric(right, errors="coerce").dropna()
    if len(left_values) < 2 or len(right_values) < 2:
        return np.nan
    pooled = np.sqrt((left_values.var(ddof=1) + right_values.var(ddof=1)) / 2)
    return float((right_values.mean() - left_values.mean()) / pooled) if pooled else 0.0


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "site": ["development_train"] * len(left) + ["external"] * len(right),
            "value": pd.concat([left, right], ignore_index=True)
            .astype("string")
            .fillna("missing"),
        }
    )
    table = pd.crosstab(frame["site"], frame["value"])
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0 or min(observed.shape) <= 1:
        return 0.0
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / total
    valid = expected > 0
    chi_square = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    denominator = total * min(observed.shape[0] - 1, observed.shape[1] - 1)
    return float(np.sqrt(chi_square / denominator)) if denominator else 0.0


def _characteristics_table(
    partitions: dict[str, pd.DataFrame],
    features: list[str],
    variable_types: dict[str, str],
) -> pd.DataFrame:
    train = partitions["development_train"]
    external = partitions["external"]
    rows: list[dict[str, Any]] = []
    outcome_levels = ["no", "yes"]
    for level in outcome_levels:
        row: dict[str, Any] = {
            "variable": "mci",
            "level_or_statistic": level,
            "variable_type": "outcome",
            "shift_measure": "not_applicable",
            "shift_value": np.nan,
        }
        for partition_name, frame in partitions.items():
            row[partition_name] = _n_pct(int(frame["mci"].eq(level).sum()), len(frame))
            row[f"{partition_name}_missing"] = _n_pct(int(frame["mci"].isna().sum()), len(frame))
        rows.append(row)

    for feature in features:
        variable_type = variable_types[feature]
        if variable_type == "numeric":
            row = {
                "variable": feature,
                "level_or_statistic": "mean (SD); median [Q1, Q3]",
                "variable_type": variable_type,
                "shift_measure": "standardized_mean_difference_external_minus_train",
                "shift_value": _numeric_shift(train[feature], external[feature]),
            }
            for partition_name, frame in partitions.items():
                values = pd.to_numeric(frame[feature], errors="coerce")
                nonmissing = values.dropna()
                if nonmissing.empty:
                    row[partition_name] = "NA"
                else:
                    row[partition_name] = (
                        f"{nonmissing.mean():.2f} ({nonmissing.std(ddof=1):.2f}); "
                        f"{nonmissing.median():.2f} "
                        f"[{nonmissing.quantile(0.25):.2f}, {nonmissing.quantile(0.75):.2f}]"
                    )
                row[f"{partition_name}_missing"] = _n_pct(int(values.isna().sum()), len(values))
            rows.append(row)
        else:
            levels = sorted(
                set(train[feature].astype("string").fillna("missing"))
                | set(external[feature].astype("string").fillna("missing"))
            )
            shift = _cramers_v(train[feature], external[feature])
            for level in levels:
                row = {
                    "variable": feature,
                    "level_or_statistic": str(level),
                    "variable_type": variable_type,
                    "shift_measure": "cramers_v_train_vs_external",
                    "shift_value": shift,
                }
                for partition_name, frame in partitions.items():
                    values = frame[feature].astype("string").fillna("missing")
                    row[partition_name] = _n_pct(int(values.eq(level).sum()), len(values))
                    row[f"{partition_name}_missing"] = _n_pct(
                        int(frame[feature].isna().sum()), len(frame)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _participant_flow(split: dict[str, Any], output_stem: Path) -> list[str]:
    summary = split["summary"].set_index("partition")
    eligible = int(split["manifest"]["eligible_development_rows"])
    excluded = int(split["manifest"]["excluded_missing_or_invalid_mci_rows"])
    train_rows = int(summary.loc["development_train_80", "rows"])
    internal_rows = int(summary.loc["development_internal_test_20", "rows"])
    external_rows = int(summary.loc["external_reserved", "rows"])
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.axis("off")
    boxes = [
        (0.5, 0.88, f"Development cohort assessed\nN = {eligible + excluded:,}"),
        (0.5, 0.68, f"Excluded: invalid/missing MCI label\nn = {excluded:,}"),
        (0.30, 0.43, f"Development training set (80%)\nn = {train_rows:,}"),
        (0.70, 0.43, f"Locked internal test set (20%)\nn = {internal_rows:,}"),
        (0.5, 0.15, f"Independent External validation cohort\nn = {external_rows:,}"),
    ]
    for x, y, label in boxes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor=BLUE, linewidth=1.5),
        )
    arrows = [((0.5, 0.82), (0.5, 0.74)), ((0.5, 0.61), (0.3, 0.50)), ((0.5, 0.61), (0.7, 0.50))]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    axis.plot([0.20, 0.80], [0.29, 0.29], color=LIGHT_GREY, lw=1)
    axis.annotate("", xy=(0.5, 0.22), xytext=(0.5, 0.29), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    axis.set_title("Participant flow and analysis partitions", fontsize=14, color=INK, pad=15)
    return _save_figure(figure, output_stem)


def _nested_model_figure(summary: pd.DataFrame, output_stem: Path) -> list[str]:
    ordered = summary.sort_values("mean_roc_auc")
    labels = [MODEL_LABELS.get(name, name) for name in ordered["model_name"]]
    y = np.arange(len(ordered))
    estimate = ordered["mean_roc_auc"].to_numpy(float)
    lower = ordered["mean_roc_auc_ci_lower"].to_numpy(float)
    upper = ordered["mean_roc_auc_ci_upper"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for index, (_, row) in enumerate(ordered.iterrows()):
        axis.errorbar(
            float(row["mean_roc_auc"]),
            index,
            xerr=[[float(row["mean_roc_auc"]) - float(row["mean_roc_auc_ci_lower"])], [float(row["mean_roc_auc_ci_upper"]) - float(row["mean_roc_auc"])]],
            fmt="o",
            color=MODEL_COLORS.get(str(row["model_name"]), BLUE),
            ecolor=GREY,
            capsize=3,
            markersize=7,
        )
    axis.set_yticks(y, labels)
    axis.set_xlabel("Mean outer-fold AUROC")
    axis.set_title("Repeated nested cross-validation model comparison")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    margin = max(0.01, (upper.max() - lower.min()) * 0.25)
    axis.set_xlim(lower.min() - margin, upper.max() + margin)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _curve_figure(
    curves: pd.DataFrame,
    metric: str,
    output_stem: Path,
) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True)
    partitions = ["internal_test_20_locked", "external_validation_locked"]
    partition_labels = ["Locked internal test", "External validation"]
    for axis, partition, partition_label in zip(axes, partitions, partition_labels):
        current = curves.loc[curves["partition"].eq(partition)]
        for model_name, group in current.groupby("model_name", sort=False):
            if metric == "roc":
                axis.plot(
                    group["false_positive_rate"],
                    group["true_positive_rate"],
                    label=MODEL_LABELS.get(model_name, model_name),
                    color=MODEL_COLORS.get(model_name, BLUE),
                    lw=1.8,
                )
            else:
                axis.plot(
                    group["recall"],
                    group["precision"],
                    label=MODEL_LABELS.get(model_name, model_name),
                    color=MODEL_COLORS.get(model_name, BLUE),
                    lw=1.8,
                )
        if metric == "roc":
            axis.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
            axis.set_xlabel("1 − specificity")
            axis.set_ylabel("Sensitivity")
        else:
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
        axis.set_title(partition_label)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(color=LIGHT_GREY, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="lower left", frameon=False, fontsize=8)
    figure.suptitle("ROC curves" if metric == "roc" else "Precision–recall curves", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _mrmr_stability_figure(stability: pd.DataFrame, output_stem: Path) -> list[str]:
    top = stability.sort_values(["selection_frequency_top_30", "median_rank"], ascending=[False, True]).head(20)
    top = top.sort_values("selection_frequency_top_30")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["canonical_name"], top["selection_frequency_top_30"], color=BLUE, edgecolor=INK, linewidth=0.4)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Selection frequency in top 30 across 30 resamples")
    axis.set_title("mRMR feature-selection stability")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _dca_figure(dca: pd.DataFrame, output_stem: Path) -> list[str]:
    labels = {
        "locked_external_probability_model": "Locked model",
        "local_recalibrated_oof_probability_model": "OOF locally recalibrated",
        "screen_all": "Screen all",
        "screen_none": "Screen none",
    }
    colors = {
        "locked_external_probability_model": BLUE,
        "local_recalibrated_oof_probability_model": ORANGE,
        "screen_all": GREY,
        "screen_none": INK,
    }
    figure, axis = plt.subplots(figsize=(8.5, 6))
    for strategy in labels:
        group = dca.loc[dca["strategy"].eq(strategy)].sort_values("threshold_probability")
        if group.empty:
            continue
        linestyle = "--" if strategy.startswith("screen_") else "-"
        axis.plot(group["threshold_probability"], group["net_benefit"], label=labels[strategy], color=colors[strategy], lw=1.8, linestyle=linestyle)
        if not strategy.startswith("screen_"):
            axis.fill_between(group["threshold_probability"], group["ci_lower"], group["ci_upper"], color=colors[strategy], alpha=0.12)
    axis.axhline(0, color=INK, lw=0.8)
    model_rows = dca.loc[~dca["strategy"].str.startswith("screen_")]
    upper_limit = float(model_rows["ci_upper"].max()) + 0.04
    axis.set_ylim(-0.05, max(0.10, upper_limit))
    axis.set_xlabel("Threshold probability")
    axis.set_ylabel("Net benefit")
    axis.set_title("Decision curve analysis in the External cohort")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(color=LIGHT_GREY, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _subgroup_figure(subgroups: pd.DataFrame, output_stem: Path) -> list[str]:
    current = subgroups.loc[
        subgroups["partition"].eq("external_descriptive")
        & subgroups["metric"].eq("roc_auc")
    ].copy()
    current["label"] = current["subgroup"] + ": " + current["level"]
    current = current.sort_values(["subgroup", "level"], ascending=[False, False])
    y = np.arange(len(current))
    figure, axis = plt.subplots(figsize=(9, max(5, 0.45 * len(current) + 1.5)))
    axis.errorbar(
        current["estimate"],
        y,
        xerr=np.vstack([current["estimate"] - current["ci_lower"], current["ci_upper"] - current["estimate"]]),
        fmt="o",
        color=BLUE,
        ecolor=GREY,
        capsize=3,
    )
    axis.set_yticks(y, current["label"])
    axis.set_xlim(0.45, 1.0)
    axis.set_xlabel("AUROC (95% CI)")
    axis.set_title("External subgroup discrimination")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _sensitivity_figure(sensitivity: pd.DataFrame, output_stem: Path) -> list[str]:
    current = sensitivity.loc[
        sensitivity["partition"].eq("external_posthoc_sensitivity")
        & sensitivity["metric"].eq("roc_auc")
        & sensitivity["operating_point"].eq("threshold_free")
    ].copy()
    current = current.sort_values("estimate")
    y = np.arange(len(current))
    figure, axis = plt.subplots(figsize=(9, max(5, 0.45 * len(current) + 1.5)))
    axis.errorbar(
        current["estimate"],
        y,
        xerr=np.vstack([current["estimate"] - current["ci_lower"], current["ci_upper"] - current["estimate"]]),
        fmt="o",
        color=ORANGE,
        ecolor=GREY,
        capsize=3,
    )
    axis.set_yticks(y, current["model_name"])
    axis.set_xlabel("External AUROC (95% CI)")
    axis.set_title("Post-hoc feature-set sensitivity analyses")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _transport_figure(drift: pd.DataFrame, output_stem: Path) -> list[str]:
    top = drift.sort_values("drift_priority_score", ascending=False).head(15).sort_values("drift_priority_score")
    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(top["canonical_name"], top["drift_priority_score"], color=PINK, edgecolor=INK, linewidth=0.4)
    axis.set_xlabel("Distribution-shift priority score")
    axis.set_title("Largest predictor distribution shifts between sites")
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _prevalence_scenarios(three_layer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, layer in three_layer.iterrows():
        sensitivity = float(layer["sensitivity"])
        specificity = float(layer["specificity"])
        for prevalence in (0.10, 0.20, 0.30):
            ppv = sensitivity * prevalence / (
                sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
            )
            npv = specificity * (1 - prevalence) / (
                (1 - sensitivity) * prevalence + specificity * (1 - prevalence)
            )
            rows.append(
                {
                    "model_name": layer["model_name"],
                    "layer": layer["layer"],
                    "assumed_prevalence": prevalence,
                    "sensitivity_assumed_transportable": sensitivity,
                    "specificity_assumed_transportable": specificity,
                    "projected_ppv": ppv,
                    "projected_npv": npv,
                    "warning": "Illustrative projection; assumes sensitivity and specificity transport to the target prevalence.",
                }
            )
    return pd.DataFrame(rows)


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_required_figure(source: Path, destination_stem: Path) -> list[str]:
    if not source.exists():
        raise FileNotFoundError(source)
    png = destination_stem.with_suffix(".png")
    tiff = destination_stem.with_suffix(".tiff")
    png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgb.save(png, format="PNG", dpi=(300, 300))
        rgb.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    return [png.name, tiff.name]


def build_manuscript_artifacts(
    development_path: Path,
    external_path: Path,
    external_education_path: Path,
    heavy_output_dir: Path,
    posthoc_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    qc_dir = output_dir / "qc_rebuild"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    heavy_manifest_path = heavy_output_dir / "final_evaluation" / "final_evaluation_manifest.json"
    posthoc_manifest_path = posthoc_output_dir / "posthoc_manifest.json"
    heavy_manifest = json.loads(heavy_manifest_path.read_text(encoding="utf-8"))
    posthoc_manifest = json.loads(posthoc_manifest_path.read_text(encoding="utf-8"))
    if heavy_manifest["status"] != "manuscript_grade_locked_evaluation_completed":
        raise RuntimeError(f"Unexpected heavy status: {heavy_manifest['status']}")
    if posthoc_manifest["status"] != "posthoc_sensitivity_transportability_and_interpretability_completed":
        raise RuntimeError(f"Unexpected post-hoc status: {posthoc_manifest['status']}")
    for manifest in (heavy_manifest, posthoc_manifest):
        if manifest.get("education_harmonization_mode") != "four_level_code_matched_auxiliary_source":
            raise RuntimeError("A final analysis did not use four-level education.")
    if posthoc_manifest.get("mice_performed"):
        raise RuntimeError("MICE was unexpectedly performed.")

    split = create_locked_split(
        development_path,
        external_path,
        qc_dir,
        external_education_path=external_education_path,
    )
    development = split["development_eligible_in_memory"]
    external = split["external_harmonized_in_memory"]
    partitions = {
        "development_train": development.iloc[split["train_relative_indices"]].reset_index(drop=True),
        "internal_test": development.iloc[split["test_relative_indices"]].reset_index(drop=True),
        "external": external.loc[external["mci"].isin(["yes", "no"])].reset_index(drop=True),
    }
    registry = split["registry"]
    predictors = registry.loc[registry["role"].eq("predictor"), "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    core_features = [feature for feature in CORE_TABLE1_FEATURES if feature in predictors]
    table1 = _characteristics_table(partitions, core_features, variable_types)
    table_s1 = _characteristics_table(partitions, predictors, variable_types)
    write_csv(table1, tables_dir / "table1_core_cohort_characteristics.csv")
    write_csv(table_s1, tables_dir / "table_s1_all_predictor_characteristics.csv")

    nested_summary = pd.read_csv(heavy_output_dir / "selection" / "nested_model_summary.csv", encoding="utf-8-sig")
    nested_outer = pd.read_csv(heavy_output_dir / "selection" / "nested_outer_metrics.csv", encoding="utf-8-sig")
    final_configs = pd.read_csv(heavy_output_dir / "final_evaluation" / "final_model_configs.csv", encoding="utf-8-sig")
    locked_metrics = pd.read_csv(heavy_output_dir / "final_evaluation" / "locked_evaluation_metrics.csv", encoding="utf-8-sig")
    thresholds = pd.read_csv(heavy_output_dir / "final_evaluation" / "development_thresholds.csv", encoding="utf-8-sig")
    three_layer = pd.read_csv(heavy_output_dir / "final_evaluation" / "three_layer_summary.csv", encoding="utf-8-sig")
    reliability = pd.read_csv(heavy_output_dir / "final_evaluation" / "external_reliability_bins.csv", encoding="utf-8-sig")
    dca = pd.read_csv(heavy_output_dir / "final_evaluation" / "external_dca.csv", encoding="utf-8-sig")
    roc_curves = pd.read_csv(heavy_output_dir / "final_evaluation" / "roc_curves.csv", encoding="utf-8-sig")
    pr_curves = pd.read_csv(heavy_output_dir / "final_evaluation" / "precision_recall_curves.csv", encoding="utf-8-sig")
    sensitivity = pd.read_csv(posthoc_output_dir / "posthoc_sensitivity_metrics.csv", encoding="utf-8-sig")
    sensitivity_scenarios = pd.read_csv(posthoc_output_dir / "posthoc_sensitivity_scenarios.csv", encoding="utf-8-sig")
    calibration = pd.read_csv(posthoc_output_dir / "local_calibration_and_brier_decomposition.csv", encoding="utf-8-sig")
    subgroups = pd.read_csv(posthoc_output_dir / "subgroup_performance.csv", encoding="utf-8-sig")
    interactions = pd.read_csv(posthoc_output_dir / "subgroup_interaction_tests.csv", encoding="utf-8-sig")
    mrmr_stability = pd.read_csv(posthoc_output_dir / "feature_stability" / "mrmr_rank_stability.csv", encoding="utf-8-sig")
    mrmr_set_stability = pd.read_csv(posthoc_output_dir / "feature_stability" / "mrmr_set_stability.csv", encoding="utf-8-sig")
    elastic_stability = pd.read_csv(posthoc_output_dir / "feature_stability" / "elastic_net_stability_selection.csv", encoding="utf-8-sig")
    transport_drift = pd.read_csv(posthoc_output_dir / "transportability" / "featurewise_transport_drift.csv", encoding="utf-8-sig")
    prevalence = _prevalence_scenarios(three_layer)

    table_map = {
        "table2_nested_cv_model_performance.csv": nested_summary,
        "table_s2_nested_outer_fold_metrics.csv": nested_outer,
        "table3_final_model_configurations.csv": final_configs,
        "table4_locked_internal_external_performance.csv": locked_metrics,
        "table5_development_operating_thresholds.csv": thresholds,
        "table6_three_layer_external_analysis.csv": three_layer,
        "table_s3_external_reliability_bins.csv": reliability,
        "table_s4_posthoc_sensitivity_metrics.csv": sensitivity,
        "table_s5_posthoc_sensitivity_scenarios.csv": sensitivity_scenarios,
        "table_s6_local_calibration_brier_decomposition.csv": calibration,
        "table_s7_subgroup_performance.csv": subgroups,
        "table_s8_subgroup_interaction_tests.csv": interactions,
        "table_s9_mrmr_rank_stability.csv": mrmr_stability,
        "table_s10_mrmr_set_stability.csv": mrmr_set_stability,
        "table_s11_elastic_net_stability.csv": elastic_stability,
        "table_s12_transportability_drift.csv": transport_drift,
        "table_s13_prevalence_scenario_ppv_npv.csv": prevalence,
    }
    for name, frame in table_map.items():
        write_csv(frame, tables_dir / name)

    figure_files: list[dict[str, str]] = []
    for figure_id, files in (
        ("figure1_participant_flow", _participant_flow(split, figures_dir / "figure1_participant_flow")),
        ("figure2_nested_model_comparison", _nested_model_figure(nested_summary, figures_dir / "figure2_nested_model_comparison")),
        ("figure3_roc_curves", _curve_figure(roc_curves, "roc", figures_dir / "figure3_roc_curves")),
        ("figure_s1_precision_recall_curves", _curve_figure(pr_curves, "pr", figures_dir / "figure_s1_precision_recall_curves")),
        ("figure4_mrmr_stability", _mrmr_stability_figure(mrmr_stability, figures_dir / "figure4_mrmr_stability")),
        ("figure6_dca", _dca_figure(dca, figures_dir / "figure6_dca")),
        ("figure7_subgroup_forest", _subgroup_figure(subgroups, figures_dir / "figure7_subgroup_forest")),
        ("figure_s2_sensitivity_forest", _sensitivity_figure(sensitivity, figures_dir / "figure_s2_sensitivity_forest")),
        ("figure_s3_transportability_shift", _transport_figure(transport_drift, figures_dir / "figure_s3_transportability_shift")),
    ):
        figure_files.extend({"figure_id": figure_id, "file": file} for file in files)

    for source_name, destination_stem in (
        ("external_calibration_before_after.png", "figure5_external_calibration_before_after"),
        ("selected_model_shap_summary.png", "figure8_selected_model_shap_summary"),
    ):
        copied = _copy_required_figure(
            posthoc_output_dir / source_name,
            figures_dir / destination_stem,
        )
        figure_files.extend(
            {"figure_id": destination_stem, "file": file} for file in copied
        )
    write_csv(pd.DataFrame(figure_files), output_dir / "figures_manifest.csv")

    required_files = [
        tables_dir / "table1_core_cohort_characteristics.csv",
        tables_dir / "table4_locked_internal_external_performance.csv",
        tables_dir / "table_s6_local_calibration_brier_decomposition.csv",
        figures_dir / "figure3_roc_curves.png",
        figures_dir / "figure5_external_calibration_before_after.png",
        figures_dir / "figure6_dca.png",
        figures_dir / "figure8_selected_model_shap_summary.png",
    ]
    missing = [str(path) for path in required_files if not path.exists() or path.stat().st_size == 0]
    manifest = {
        "status": "ready_for_manuscript_writing" if not missing else "incomplete",
        "heavy_status": heavy_manifest["status"],
        "posthoc_status": posthoc_manifest["status"],
        "education_harmonization_mode": heavy_manifest["education_harmonization_mode"],
        "mice_performed": False,
        "tables_written": sorted(path.name for path in tables_dir.glob("*.csv")),
        "figures_written": sorted(path.name for path in figures_dir.iterdir() if path.is_file()),
        "participant_level_outputs_written": False,
        "required_missing": missing,
        "interpretation_boundary": (
            "Locked External validation is confirmatory; local recalibration, local threshold updating, "
            "transportable-core features and alternative feature selectors are secondary analyses."
        ),
    }
    (output_dir / "manuscript_readiness_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing:
        raise RuntimeError(f"Manuscript package incomplete: {missing}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggregate-only manuscript tables and publication figures.")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-education", type=Path, required=True)
    parser.add_argument("--heavy-output", type=Path, required=True)
    parser.add_argument("--posthoc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manuscript_artifacts(
        args.development,
        args.external,
        args.external_education,
        args.heavy_output,
        args.posthoc_output,
        args.output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
