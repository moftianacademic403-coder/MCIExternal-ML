from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from heavy_nested_cv import _tabpfn_frame
from heavy_posthoc_analysis import (
    POSTHOC_SEED,
    _load_locked_configuration,
    _saved_calibration,
)
from light_modeling import model_ready_frame
from split_development import create_locked_split


NAVY = "#17324D"
BLUE = "#2C6E9B"
TEAL = "#168C86"
ORANGE = "#D97742"
GOLD = "#C99A2E"
PLUM = "#8C5A7A"
INK = "#1F2933"
MID = "#65727C"
LIGHT = "#D8E0E5"
PALE = "#EEF3F5"
WHITE = "#FFFFFF"
MODEL_COLORS = {
    "elastic_net_logistic": BLUE,
    "svm_rbf": ORANGE,
    "random_forest": TEAL,
    "xgboost": PLUM,
    "tabpfn": GOLD,
}
MODEL_LABELS = {
    "elastic_net_logistic": "Elastic-net logistic",
    "svm_rbf": "RBF-SVM",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN",
}
FEATURE_LABELS = {
    "iadl": "Instrumental ADL",
    "household_income": "Household income",
    "visual_acuity_both": "Binocular visual acuity",
    "sleep_quality": "Sleep quality",
    "employment_status": "Employment status",
    "whisper_test_left": "Left whisper test",
    "osteoporosis_status": "Osteoporosis",
    "education": "Education",
    "adl": "Basic ADL",
    "age": "Age",
    "current_smoker": "Current smoking",
    "depression_status": "Depression",
    "housing_status": "Housing status",
    "waist_hip_ratio": "Waist-to-hip ratio",
    "hypertension": "Hypertension",
    "max_handgrip": "Maximum handgrip",
    "history_heart_failure": "History of heart failure",
    "history_stroke": "History of stroke",
    "rbc": "Red blood cell count",
    "ldl_cholesterol": "LDL cholesterol",
    "sex": "Sex",
    "age_group": "Age group",
    "selected_predictor_missingness": "Selected-predictor missingness",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.fontsize": 7.5,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(axis: plt.Axes, grid: str | None = None) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT, linewidth=0.55, alpha=0.9)
        axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.7)


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.09,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NAVY,
    )


def _save(figure: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, kwargs in (
        (".png", {"dpi": 400}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05, **kwargs)
        outputs.append(path.name)
    plt.close(figure)
    return outputs


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _metric(
    metrics: pd.DataFrame,
    partition: str,
    model: str,
    metric: str,
    operating_point: str = "threshold_free",
) -> pd.Series:
    rows = metrics.loc[
        metrics["partition"].eq(partition)
        & metrics["model_name"].eq(model)
        & metrics["operating_point"].eq(operating_point)
        & metrics["metric"].eq(metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one metric row for {partition}/{model}/{operating_point}/{metric}; got {len(rows)}"
        )
    return rows.iloc[0]


def _study_design(split: dict[str, Any], stem: Path) -> list[str]:
    summary = split["summary"].set_index("partition")
    eligible = int(split["manifest"]["eligible_development_rows"])
    excluded = int(split["manifest"]["excluded_missing_or_invalid_mci_rows"])
    train_n = int(summary.loc["train_80", "rows"])
    internal_n = int(summary.loc["internal_test_20_locked", "rows"])
    external_n = int(summary.loc["external_reserved", "rows"])
    figure, axis = plt.subplots(figsize=(7.1, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, detail: str, color: str) -> None:
        patch = mpl.patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.25,
        )
        axis.add_patch(patch)
        axis.text(x, y + 0.025, title, ha="center", va="center", fontsize=9, fontweight="semibold")
        axis.text(x, y - 0.03, detail, ha="center", va="center", fontsize=8, color=MID)

    box(0.25, 0.83, 0.38, 0.15, "Development cohort", f"Assessed N = {eligible + excluded:,}; eligible n = {eligible:,}", NAVY)
    box(0.75, 0.83, 0.38, 0.15, "External validation cohort", f"Independent sample n = {external_n:,}", TEAL)
    box(0.25, 0.58, 0.38, 0.15, "Locked development split", "Stratified 80:20 split before modeling", NAVY)
    box(0.13, 0.31, 0.22, 0.15, "Training set", f"n = {train_n:,}\nModel selection + tuning", BLUE)
    box(0.38, 0.31, 0.22, 0.15, "Internal test", f"n = {internal_n:,}\nOne locked evaluation", GOLD)
    box(0.75, 0.42, 0.38, 0.19, "External evaluation", "Locked validation first\nLocal updating reported separately", TEAL)
    box(0.75, 0.14, 0.38, 0.14, "Interpretability + robustness", "Aggregate SHAP, subgroups, sensitivity", PLUM)

    arrows = [
        ((0.25, 0.75), (0.25, 0.67)),
        ((0.25, 0.50), (0.13, 0.40)),
        ((0.25, 0.50), (0.38, 0.40)),
        ((0.75, 0.75), (0.75, 0.53)),
        ((0.75, 0.32), (0.75, 0.22)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": MID})
    axis.text(0.25, 0.97, "Study design and analysis partitions", ha="center", fontsize=11, fontweight="semibold", color=NAVY)
    axis.text(0.75, 0.97, "Independent validation and secondary analyses", ha="center", fontsize=11, fontweight="semibold", color=NAVY)
    axis.text(0.25, 0.685, f"Excluded for invalid/missing outcome: n = {excluded:,}", ha="center", fontsize=7.5, color=MID)
    figure.tight_layout()
    return _save(figure, stem)


def _nested_model_comparison(summary: pd.DataFrame, stem: Path) -> list[str]:
    ordered = summary.sort_values("mean_roc_auc", ascending=True).reset_index(drop=True)
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 3.35), gridspec_kw={"wspace": 0.34})
    specs = [
        ("mean_roc_auc", "mean_roc_auc_ci_lower", "mean_roc_auc_ci_upper", "AUROC"),
        ("mean_average_precision", "mean_average_precision_ci_lower", "mean_average_precision_ci_upper", "Average precision"),
        ("mean_brier_score", "mean_brier_score_ci_lower", "mean_brier_score_ci_upper", "Brier score ↓"),
    ]
    y = np.arange(len(ordered))
    for idx, (axis, (est, low, high, title)) in enumerate(zip(axes, specs)):
        for row_index, row in ordered.iterrows():
            color = MODEL_COLORS[str(row["model_name"])]
            axis.errorbar(
                float(row[est]), row_index,
                xerr=[[float(row[est]) - float(row[low])], [float(row[high]) - float(row[est])]],
                fmt="o", color=color, ecolor=MID, elinewidth=1.0, capsize=2.2,
                markersize=5.5, markeredgecolor=WHITE, markeredgewidth=0.5,
            )
        axis.set_title(title, pad=7)
        axis.set_yticks(y)
        if idx == 0:
            axis.set_yticklabels([MODEL_LABELS.get(v, v) for v in ordered["model_name"]])
        else:
            axis.set_yticklabels([])
        _clean_axis(axis, "x")
        values = np.concatenate([ordered[low].to_numpy(float), ordered[high].to_numpy(float)])
        pad = max(0.003, np.ptp(values) * 0.22)
        axis.set_xlim(values.min() - pad, values.max() + pad)
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        _panel(axis, chr(65 + idx))
    figure.suptitle("Repeated nested cross-validation", y=1.01, fontsize=11, fontweight="semibold", color=NAVY)
    figure.text(0.5, -0.02, "15 outer-fold evaluations per model; intervals are descriptive because folds are dependent", ha="center", fontsize=7.5, color=MID)
    return _save(figure, stem)


def _discrimination(
    roc: pd.DataFrame,
    pr: pd.DataFrame,
    metrics: pd.DataFrame,
    stem: Path,
) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.2), sharex="col", sharey="row")
    partitions = [
        ("internal_test_20_locked", "Locked internal test", 400),
        ("external_validation_locked", "External validation", 1345),
    ]
    for col, (partition, title, n) in enumerate(partitions):
        for model_name, group in roc.loc[roc["partition"].eq(partition)].groupby("model_name", sort=False):
            selected = model_name == "tabpfn"
            axes[0, col].plot(
                group["false_positive_rate"], group["true_positive_rate"],
                color=MODEL_COLORS[model_name] if selected else LIGHT,
                lw=2.3 if selected else 0.9,
                alpha=1 if selected else 0.85,
                zorder=3 if selected else 1,
            )
        axes[0, col].plot([0, 1], [0, 1], ls=(0, (4, 3)), color=MID, lw=0.9)
        selected_auc = _metric(metrics, partition, "tabpfn", "roc_auc")
        axes[0, col].text(
            0.97, 0.06,
            f"TabPFN AUROC {selected_auc['estimate']:.3f}\n95% CI {selected_auc['ci_lower']:.3f}–{selected_auc['ci_upper']:.3f}",
            transform=axes[0, col].transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.3"},
        )
        axes[0, col].set_title(f"{title}\nN = {n:,}", pad=7)
        for model_name, group in pr.loc[pr["partition"].eq(partition)].groupby("model_name", sort=False):
            selected = model_name == "tabpfn"
            axes[1, col].plot(
                group["recall"], group["precision"],
                color=MODEL_COLORS[model_name] if selected else LIGHT,
                lw=2.3 if selected else 0.9,
                alpha=1 if selected else 0.85,
                zorder=3 if selected else 1,
            )
        selected_ap = _metric(metrics, partition, "tabpfn", "average_precision")
        prevalence = 0.4725 if partition.startswith("internal") else 711 / 1345
        axes[1, col].axhline(prevalence, ls=(0, (4, 3)), color=MID, lw=0.9)
        axes[1, col].text(
            0.97, 0.06,
            f"TabPFN AP {selected_ap['estimate']:.3f}\n95% CI {selected_ap['ci_lower']:.3f}–{selected_ap['ci_upper']:.3f}",
            transform=axes[1, col].transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.3"},
        )
    for axis in axes.ravel():
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        _clean_axis(axis, "both")
    axes[0, 0].set_ylabel("Sensitivity")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 1].set_xlabel("Recall")
    axes[0, 0].set_xlabel("1 − specificity")
    axes[0, 1].set_xlabel("1 − specificity")
    for label, axis in zip("ABCD", axes.ravel()):
        _panel(axis, label)
    legend = [
        Line2D([0], [0], color=GOLD, lw=2.4, label="Selected model (TabPFN)"),
        Line2D([0], [0], color=LIGHT, lw=1.2, label="Other candidate models"),
        Line2D([0], [0], color=MID, lw=0.9, ls=(0, (4, 3)), label="No-skill reference"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Locked discrimination performance", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.subplots_adjust(top=0.90, bottom=0.11, left=0.10, right=0.98, hspace=0.32, wspace=0.20)
    return _save(figure, stem)


def _external_performance(
    reliability: pd.DataFrame,
    dca: pd.DataFrame,
    three_layer: pd.DataFrame,
    metrics: pd.DataFrame,
    stem: Path,
) -> list[str]:
    figure = plt.figure(figsize=(7.1, 6.25))
    gs = figure.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.38, wspace=0.28)
    ax_cal = figure.add_subplot(gs[0, 0])
    ax_ops = figure.add_subplot(gs[0, 1])
    ax_dca = figure.add_subplot(gs[1, :])

    cal_specs = [
        ("locked_external_probability", "Locked model", BLUE, "o"),
        ("external_10fold_oof_local_recalibration", "OOF local recalibration", ORANGE, "s"),
    ]
    ax_cal.plot([0, 1], [0, 1], color=INK, lw=1.0, ls=(0, (4, 3)), label="Ideal")
    for analysis, label, color, marker in cal_specs:
        group = reliability.loc[reliability["analysis"].eq(analysis)].sort_values("mean_predicted")
        ax_cal.errorbar(
            group["mean_predicted"], group["observed_rate"],
            yerr=[group["observed_rate"] - group["observed_rate_ci_lower"], group["observed_rate_ci_upper"] - group["observed_rate"]],
            color=color, marker=marker, ms=4.2, lw=1.45, capsize=2.0, label=label,
        )
    ax_cal.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed MCI proportion", title="External calibration")
    ax_cal.legend(frameon=False, loc="upper left")
    _clean_axis(ax_cal, "both")
    _panel(ax_cal, "A")

    layer_labels = {
        "A_locked_model_plus_development_threshold_85": "Locked threshold",
        "B_locked_model_plus_local_threshold_85": "Local threshold",
        "C_locked_model_plus_local_recalibration_and_threshold_85": "Local recalibration",
    }
    measures = ["sensitivity", "specificity", "ppv", "npv"]
    measure_labels = ["Sensitivity", "Specificity", "PPV", "NPV"]
    x = np.arange(len(measures))
    width = 0.23
    for i, (_, row) in enumerate(three_layer.iterrows()):
        values = [float(row[m]) for m in measures]
        ax_ops.bar(x + (i - 1) * width, values, width=width, color=[BLUE, GOLD, ORANGE][i], label=layer_labels[str(row["layer"])], edgecolor=WHITE, linewidth=0.45)
    ax_ops.set_xticks(x, measure_labels, rotation=18, ha="right")
    ax_ops.set_ylim(0, 1.05)
    ax_ops.set_ylabel("Proportion")
    ax_ops.set_title("External operating characteristics", pad=24)
    ax_ops.legend(
        frameon=False, fontsize=6.4, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 1.005), columnspacing=0.8, handlelength=1.6,
    )
    _clean_axis(ax_ops, "y")
    _panel(ax_ops, "B")

    dca_specs = [
        ("locked_external_probability_model", "Locked model", BLUE, "-"),
        ("local_recalibrated_oof_probability_model", "OOF local recalibration", ORANGE, "-"),
        ("screen_all", "Screen all", MID, (0, (4, 3))),
        ("screen_none", "Screen none", INK, (0, (1, 2))),
    ]
    for strategy, label, color, linestyle in dca_specs:
        group = dca.loc[dca["strategy"].eq(strategy)].sort_values("threshold_probability")
        if group.empty:
            continue
        ax_dca.plot(group["threshold_probability"], group["net_benefit"], color=color, lw=1.8 if "model" in strategy else 1.0, ls=linestyle, label=label)
        if "model" in strategy:
            ax_dca.fill_between(group["threshold_probability"], group["ci_lower"], group["ci_upper"], color=color, alpha=0.10, linewidth=0)
    ax_dca.axvline(0.329254, color=GOLD, lw=1.0, ls=(0, (3, 3)))
    ax_dca.text(0.337, ax_dca.get_ylim()[1] if ax_dca.get_ylim()[1] else 0.1, "Locked 85% sensitivity threshold", fontsize=7, color=GOLD, va="top")
    ax_dca.set(xlim=(0.05, 0.35), xlabel="Threshold probability", ylabel="Net benefit", title="Decision-curve analysis in the external cohort")
    ax_dca.legend(frameon=False, ncol=4, loc="upper right")
    _clean_axis(ax_dca, "both")
    _panel(ax_dca, "C")
    figure.suptitle("External validation, calibration, and clinical utility", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.subplots_adjust(top=0.84, bottom=0.09, left=0.10, right=0.98)
    return _save(figure, stem)


def _bootstrap_importance(values: np.ndarray, repeats: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(POSTHOC_SEED + 33)
    draws = np.empty((repeats, values.shape[1]), dtype=float)
    for i in range(repeats):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        draws[i] = np.mean(np.abs(values[indices]), axis=0)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _jitter(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values))
    centered = ((ranks % 7) - 3) / 3
    return centered * 0.095 + rng.normal(0, 0.008, size=len(values))


def _global_shap(
    values: np.ndarray,
    selected_features: list[str],
    low: np.ndarray,
    high: np.ndarray,
    stem: Path,
) -> list[str]:
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    display = order[::-1]
    y = np.arange(len(display))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 6.1), gridspec_kw={"width_ratios": [0.92, 1.55], "wspace": 0.32})
    axes[0].barh(y, mean_abs[display], color=NAVY, edgecolor=WHITE, linewidth=0.5, height=0.66)
    axes[0].errorbar(
        mean_abs[display], y,
        xerr=[mean_abs[display] - low[display], high[display] - mean_abs[display]],
        fmt="none", ecolor=MID, elinewidth=0.9, capsize=2,
    )
    axes[0].set_yticks(y, [FEATURE_LABELS.get(selected_features[i], selected_features[i]) for i in display])
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("Global importance with 95% bootstrap CI", pad=7)
    _clean_axis(axes[0], "x")
    _panel(axes[0], "A")

    max_abs = float(np.quantile(np.abs(values[:, display]), 0.995))
    for row, feature_index in enumerate(display):
        current = values[:, feature_index]
        jitter = _jitter(current, POSTHOC_SEED + feature_index)
        colors = np.where(current >= 0, ORANGE, BLUE)
        axes[1].scatter(current, row + jitter, c=colors, s=13, alpha=0.82, edgecolors=WHITE, linewidths=0.25, rasterized=True)
    axes[1].axvline(0, color=INK, lw=0.8)
    axes[1].set_yticks(y, [])
    axes[1].set_xlim(-max_abs * 1.08, max_abs * 1.08)
    axes[1].set_xlabel("SHAP value (change in class-1 model logit)")
    axes[1].set_title("Direction and distribution of contributions", pad=7)
    _clean_axis(axes[1], "x")
    _panel(axes[1], "B")
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=WHITE, label="Increases class-1 logit"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=WHITE, label="Decreases class-1 logit"),
        ], frameon=False, loc="lower right", fontsize=7,
    )
    figure.suptitle("TabPFN global SHAP analysis", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.text(0.5, 0.012, "50 outcome-stratified observations from the locked internal test; bootstrap intervals quantify sampling variability within the explained set", ha="center", fontsize=7.2, color=MID)
    figure.subplots_adjust(top=0.92, bottom=0.10, left=0.25, right=0.98)
    return _save(figure, stem)


def _format_value(feature: str, value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if feature in {"age"}:
        return f"{float(value):.0f}"
    if feature in {"waist_hip_ratio", "visual_acuity_both", "rbc"}:
        return f"{float(value):.2f}"
    if feature in {"max_handgrip", "ldl_cholesterol"}:
        return f"{float(value):.1f}"
    text = str(value).replace("_", " ")
    return text if len(text) <= 18 else text[:17] + "…"


def _local_waterfalls(
    local_values: np.ndarray,
    baselines: np.ndarray,
    predictions: np.ndarray,
    prototypes: pd.DataFrame,
    selected_features: list[str],
    stem: Path,
) -> list[str]:
    labels = ["Low-score synthetic profile", "Mid-score synthetic profile", "High-score synthetic profile"]
    colors = [BLUE, GOLD, ORANGE]
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 5.7), sharex=False)
    for panel_index, axis in enumerate(axes):
        contributions = local_values[panel_index]
        top_indices = np.argsort(np.abs(contributions))[::-1][:8]
        other = float(contributions.sum() - contributions[top_indices].sum())
        plot_values = np.r_[contributions[top_indices], other]
        plot_labels = [
            f"{FEATURE_LABELS.get(selected_features[i], selected_features[i])}\n= {_format_value(selected_features[i], prototypes.iloc[panel_index, i])}"
            for i in top_indices
        ] + ["Other features"]
        order = np.argsort(np.abs(plot_values))
        y = np.arange(len(plot_values))
        ordered_values = plot_values[order]
        ordered_labels = [plot_labels[i] for i in order]
        bar_colors = [ORANGE if v >= 0 else BLUE for v in ordered_values]
        axis.barh(y, ordered_values, color=bar_colors, edgecolor=WHITE, linewidth=0.45, height=0.66)
        axis.axvline(0, color=INK, lw=0.8)
        axis.set_yticks(y, ordered_labels)
        axis.tick_params(axis="y", labelsize=6.6)
        axis.set_title(f"{labels[panel_index]}\nRaw probability = {predictions[panel_index]:.3f}", color=colors[panel_index], pad=7)
        axis.set_xlabel("SHAP contribution to class-1 logit")
        _clean_axis(axis, "x")
        axis.text(-0.18, 1.035, chr(65 + panel_index), transform=axis.transAxes, fontsize=11, fontweight="bold", color=NAVY, va="bottom")
        axis.text(0.02, 0.02, f"Baseline = {baselines[panel_index]:.3f}", transform=axis.transAxes, fontsize=6.8, color=MID)
    max_limit = max(max(abs(axis.get_xlim()[0]), abs(axis.get_xlim()[1])) for axis in axes)
    for axis in axes:
        axis.set_xlim(-max_limit, max_limit)
    figure.suptitle("Local SHAP explanations for privacy-preserving synthetic profiles", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.text(0.5, 0.012, "Profiles are feature-wise medians/modes within internal-test score tertiles and do not represent identifiable participants", ha="center", fontsize=7.2, color=MID)
    figure.subplots_adjust(top=0.82, bottom=0.10, left=0.15, right=0.99, wspace=0.53)
    return _save(figure, stem)


def _subgroup_figure(subgroups: pd.DataFrame, interactions: pd.DataFrame, stem: Path) -> list[str]:
    external = subgroups.loc[
        subgroups["partition"].eq("external_descriptive") & subgroups["metric"].eq("roc_auc")
    ].copy()
    group_order = ["sex", "age_group", "selected_predictor_missingness"]
    external["group_order"] = external["subgroup"].map({v: i for i, v in enumerate(group_order)})
    external = external.sort_values(["group_order", "level"]).reset_index(drop=True)
    label_map = {
        "female": "Female", "male": "Male", "65_to_74": "65–74 years", "75_plus": "≥75 years",
        "none": "No missing selected predictors", "one_or_more": "≥1 missing selected predictor",
    }
    labels = [label_map.get(str(v), str(v).replace("_", " ")) for v in external["level"]]
    y = np.arange(len(external))[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 4.35), gridspec_kw={"width_ratios": [1.75, 0.85], "wspace": 0.32})
    for i, row in external.iterrows():
        axes[0].errorbar(
            float(row["estimate"]), y[i],
            xerr=[[float(row["estimate"]) - float(row["ci_lower"])], [float(row["ci_upper"]) - float(row["estimate"])]],
            fmt="o", color=NAVY, ecolor=MID, elinewidth=1.1, capsize=2.5, markersize=5.5,
            markerfacecolor=WHITE, markeredgewidth=1.2,
        )
        axes[0].text(float(row["ci_upper"]) + 0.007, y[i], f"{row['estimate']:.2f}", va="center", fontsize=7, color=MID)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0.68, 0.93)
    axes[0].set_xlabel("AUROC (95% CI)")
    axes[0].set_title("External discrimination by subgroup", pad=7)
    _clean_axis(axes[0], "x")
    _panel(axes[0], "A")

    ext_interactions = interactions.loc[interactions["partition"].eq("external_descriptive")].copy()
    ext_interactions["order"] = ext_interactions["subgroup"].map({v: i for i, v in enumerate(group_order)})
    ext_interactions = ext_interactions.sort_values("order")
    yy = np.arange(len(ext_interactions))[::-1]
    pvals = ext_interactions["interaction_p_value"].to_numpy(float)
    axes[1].scatter(pvals, yy, s=38, color=TEAL, edgecolor=WHITE, linewidth=0.5)
    axes[1].axvline(0.05, color=ORANGE, lw=1.0, ls=(0, (4, 3)))
    axes[1].set_yticks(yy, [FEATURE_LABELS.get(v, v) for v in ext_interactions["subgroup"]])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Interaction P value")
    axes[1].set_title("Heterogeneity tests", pad=7)
    _clean_axis(axes[1], "x")
    _panel(axes[1], "B")
    figure.suptitle("External subgroup performance", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.text(0.5, 0.012, "Subgroup analyses are exploratory; interaction tests are unadjusted for multiplicity", ha="center", fontsize=7.2, color=MID)
    figure.subplots_adjust(top=0.87, bottom=0.13, left=0.27, right=0.98)
    return _save(figure, stem)


def _stability_transport(stability: pd.DataFrame, drift: pd.DataFrame, selected: list[str], stem: Path) -> list[str]:
    top = stability.sort_values(["selection_frequency_top_20", "median_rank"], ascending=[False, True]).head(15).copy()
    top = top.sort_values("median_rank", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 5.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    y = np.arange(len(top))
    axes[0].hlines(y, top["rank_q1"], top["rank_q3"], color=LIGHT, lw=5)
    axes[0].scatter(top["median_rank"], y, color=NAVY, s=24, zorder=3)
    axes[0].set_yticks(y, [FEATURE_LABELS.get(v, v) for v in top["canonical_name"]])
    axes[0].set_xlabel("mRMR rank (lower is better)")
    axes[0].set_title("Feature-selection stability", pad=7)
    _clean_axis(axes[0], "x")
    _panel(axes[0], "A")

    selected_drift = drift.loc[drift["canonical_name"].isin(selected)].copy()
    score_col = "drift_priority_score"
    if score_col not in selected_drift.columns:
        score_col = "standardized_mean_difference" if "standardized_mean_difference" in selected_drift.columns else "drift_magnitude"
    if score_col not in selected_drift.columns:
        numeric = [c for c in selected_drift.columns if c not in {"canonical_name", "variable_type"} and pd.api.types.is_numeric_dtype(selected_drift[c])]
        score_col = numeric[0]
    selected_drift[score_col] = pd.to_numeric(selected_drift[score_col], errors="coerce").abs()
    selected_drift = selected_drift.nlargest(15, score_col).sort_values(score_col)
    yy = np.arange(len(selected_drift))
    axes[1].barh(yy, selected_drift[score_col], color=TEAL, edgecolor=WHITE, linewidth=0.45)
    axes[1].set_yticks(yy, [FEATURE_LABELS.get(v, v) for v in selected_drift["canonical_name"]])
    axes[1].set_xlabel("Absolute transport drift")
    axes[1].set_title("Development-to-external shift", pad=7)
    _clean_axis(axes[1], "x")
    _panel(axes[1], "B")
    figure.suptitle("Feature stability and transportability", y=0.995, fontsize=11, fontweight="semibold", color=NAVY)
    figure.subplots_adjust(top=0.90, bottom=0.09, left=0.28, right=0.98)
    return _save(figure, stem)


def _synthetic_profiles(frame: pd.DataFrame, scores: np.ndarray, features: list[str], variable_types: dict[str, str]) -> pd.DataFrame:
    quantiles = np.quantile(scores, [0, 1 / 3, 2 / 3, 1])
    profiles: list[dict[str, Any]] = []
    for i in range(3):
        if i < 2:
            mask = (scores >= quantiles[i]) & (scores < quantiles[i + 1])
        else:
            mask = (scores >= quantiles[i]) & (scores <= quantiles[i + 1])
        group = frame.loc[mask, features]
        profile: dict[str, Any] = {}
        for feature in features:
            if variable_types[feature] == "numeric":
                profile[feature] = pd.to_numeric(group[feature], errors="coerce").median()
            else:
                mode = group[feature].dropna().mode()
                profile[feature] = mode.iloc[0] if len(mode) else pd.NA
        profiles.append(profile)
    return pd.DataFrame(profiles, columns=features)


def _run_shap(
    train: pd.DataFrame,
    train_labels: np.ndarray,
    internal: pd.DataFrame,
    internal_labels: np.ndarray,
    selected_features: list[str],
    variable_types: dict[str, str],
    params: dict[str, Any],
) -> dict[str, Any]:
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

    train_prepared = _tabpfn_frame(train, selected_features, variable_types)
    internal_prepared = _tabpfn_frame(internal, selected_features, variable_types)
    categorical_indices = [i for i, f in enumerate(selected_features) if variable_types[f] == "categorical"]
    classifier = TabPFNClassifier(
        n_estimators=int(params["n_estimators"]), categorical_features_indices=categorical_indices,
        device="cuda", random_state=POSTHOC_SEED, show_progress_bar=False, fit_mode="fit_with_cache",
    )
    classifier.fit(train_prepared, train_labels)
    internal_scores = classifier.predict_proba(internal_prepared)[:, 1]
    rng = np.random.default_rng(POSTHOC_SEED)
    explain_indices = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(internal_labels == label), size=25, replace=False) for label in (0, 1)
    ]))
    explain_frame = internal_prepared.iloc[explain_indices]
    profiles = _synthetic_profiles(internal_prepared, internal_scores, selected_features, variable_types)
    profiles_prepared = _tabpfn_frame(profiles, selected_features, variable_types)
    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(model=classifier, data=train_prepared, index="SV", max_order=1)
    interaction_values = explainer.explain_X(
        pd.concat([explain_frame, profiles_prepared], ignore_index=True).to_numpy(),
        budget=256, random_state=POSTHOC_SEED, verbose=False,
    )
    values = np.asarray([[current[(j,)] for j in range(len(selected_features))] for current in interaction_values], dtype=float)
    baselines = np.asarray([current.baseline_value for current in interaction_values], dtype=float)
    combined = pd.concat([explain_frame, profiles_prepared], ignore_index=True)
    reconstructed = baselines + values.sum(axis=1)
    direct_logits = np.asarray(classifier.predict_logits(combined), dtype=float)
    direct_score = direct_logits[:, 1] if direct_logits.ndim == 2 else direct_logits
    direct_probability = classifier.predict_proba(combined)[:, 1]
    additivity_error = np.abs(reconstructed - direct_score)
    result = {
        "global_values": values[: len(explain_frame)],
        "local_values": values[len(explain_frame) :],
        "local_baselines": baselines[len(explain_frame) :],
        "local_predictions": direct_probability[len(explain_frame) :],
        "profiles": profiles_prepared,
        "additivity_max_abs_error": float(additivity_error.max()),
        "additivity_mean_abs_error": float(additivity_error.mean()),
        "explained_rows": int(len(explain_frame)),
        "budget_per_row": 256,
    }
    del classifier, explainer, interaction_values
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def _validate_saved_outputs(
    selection_manifest: dict[str, Any],
    final_manifest: dict[str, Any],
    posthoc_manifest: dict[str, Any],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    three_layer: pd.DataFrame,
    reliability: pd.DataFrame,
    dca: pd.DataFrame,
    shap_additivity: dict[str, float],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str, severity: str = "high") -> None:
        checks.append({"check": name, "passed": bool(passed), "severity_if_failed": severity, "evidence": evidence})

    add("Nested-CV completion status", selection_manifest.get("status") == "development_only_nested_cv_model_family_selected", str(selection_manifest.get("status")))
    add("Locked evaluation completion status", final_manifest.get("status") == "manuscript_grade_locked_evaluation_completed", str(final_manifest.get("status")))
    add("Post-hoc completion status", posthoc_manifest.get("status") == "posthoc_sensitivity_transportability_and_interpretability_completed", str(posthoc_manifest.get("status")))
    add("Participant-level predictions not written", final_manifest.get("participant_level_predictions_written") is False, str(final_manifest.get("participant_level_predictions_written")))
    metric_values = metrics[["estimate", "ci_lower", "ci_upper"]].apply(pd.to_numeric, errors="coerce")
    add("Metric estimates and intervals are finite", np.isfinite(metric_values.to_numpy()).all(), f"rows={len(metrics)}")
    add("Confidence intervals contain estimates", bool(((metric_values["ci_lower"] <= metric_values["estimate"]) & (metric_values["estimate"] <= metric_values["ci_upper"])).all()), "all aggregate metric rows")
    bounded = metrics["metric"].isin(["roc_auc", "average_precision", "brier_score", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy"])
    add("Bounded metrics lie in [0,1]", bool(metrics.loc[bounded, "estimate"].between(0, 1).all()), f"bounded rows={int(bounded.sum())}")
    nested_winner = str(summary.sort_values("mean_roc_auc", ascending=False).iloc[0]["model_name"])
    add("Selected model is TabPFN", nested_winner == "tabpfn" and final_manifest.get("selected_model_name") == "tabpfn", f"nested_winner={nested_winner}; final={final_manifest.get('selected_model_name')}")
    add("All five candidate models have 15 outer evaluations", set(summary["outer_fold_evaluations"].astype(int)) == {15} and len(summary) == 5, f"models={len(summary)}")
    add("Paired model differences use matched outer folds", bool((paired["outer_fold_pairs"].astype(int) == 15).all()), f"comparisons={len(paired)}")
    add("External reliability bins cover all participants per analysis", all(group["rows"].sum() == 1345 for _, group in reliability.groupby("analysis")), str(reliability.groupby("analysis")["rows"].sum().to_dict()))
    add("DCA coordinates are finite", np.isfinite(dca[["threshold_probability", "net_benefit"]].to_numpy(float)).all(), f"rows={len(dca)}")
    add("Three-layer external summaries have valid rates", bool(three_layer[["sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "brier_score"]].apply(pd.to_numeric).apply(lambda s: s.between(0, 1)).all().all()), f"layers={len(three_layer)}")
    add("SHAP additivity is numerically adequate", shap_additivity["max_abs_error"] <= 0.03, f"max_abs_error={shap_additivity['max_abs_error']:.6f}; mean_abs_error={shap_additivity['mean_abs_error']:.6f}", "medium")
    failures = [row for row in checks if not row["passed"]]
    rating = "needs_revision" if any(row["severity_if_failed"] == "high" for row in failures) else ("share_with_caveats" if failures else "share_with_caveats")
    return {"overall_rating": rating, "checks": checks, "failed_checks": failures}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _style()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = args.prior_output.resolve()
    posthoc = args.posthoc_output.resolve()
    selection_dir = prior / "selection"
    final_dir = prior / "final_evaluation"
    analysis_dir = posthoc / "analysis"

    split = create_locked_split(args.development, args.external, args.qc_output, external_education_path=args.external_education)
    development = split["development_eligible_in_memory"]
    train_indices = split["train_relative_indices"]
    test_indices = split["test_relative_indices"]
    train_raw = development.iloc[train_indices].reset_index(drop=True)
    internal_raw = development.iloc[test_indices].reset_index(drop=True)
    registry = split["registry"]
    predictors = registry.loc[registry["role"].eq("predictor"), "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train = model_ready_frame(train_raw, predictors, variable_types)
    internal = model_ready_frame(internal_raw, predictors, variable_types)
    train_labels = train_raw["mci"].map({"no": 0, "yes": 1}).to_numpy(int)
    internal_labels = internal_raw["mci"].map({"no": 0, "yes": 1}).to_numpy(int)
    model_name, params, selected_features, _, _, _ = _load_locked_configuration(prior)
    if model_name != "tabpfn":
        raise RuntimeError(f"Publication SHAP workflow expects selected TabPFN model, got {model_name}")

    summary = _read(selection_dir / "nested_model_summary.csv")
    paired = _read(selection_dir / "nested_paired_auc_differences.csv")
    metrics = _read(final_dir / "locked_evaluation_metrics.csv")
    roc = _read(final_dir / "roc_curves.csv")
    pr = _read(final_dir / "precision_recall_curves.csv")
    reliability = _read(final_dir / "external_reliability_bins.csv")
    dca = _read(final_dir / "external_dca.csv")
    three_layer = _read(final_dir / "three_layer_summary.csv")
    subgroups = _read(analysis_dir / "subgroup_performance.csv")
    interactions = _read(analysis_dir / "subgroup_interaction_tests.csv")
    stability = _read(analysis_dir / "feature_stability" / "mrmr_rank_stability.csv")
    drift = _read(analysis_dir / "transportability" / "featurewise_transport_drift.csv")

    shap_result = _run_shap(train, train_labels, internal, internal_labels, selected_features, variable_types, params)
    low, high = _bootstrap_importance(shap_result["global_values"])
    importance = pd.DataFrame({
        "canonical_name": selected_features,
        "display_name": [FEATURE_LABELS.get(v, v) for v in selected_features],
        "mean_abs_shap": np.mean(np.abs(shap_result["global_values"]), axis=0),
        "mean_shap": np.mean(shap_result["global_values"], axis=0),
        "bootstrap_ci_lower": low,
        "bootstrap_ci_upper": high,
        "explained_rows": shap_result["explained_rows"],
        "budget_per_row": shap_result["budget_per_row"],
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output / "global_shap_importance_with_ci.csv", index=False, encoding="utf-8-sig")

    figures: dict[str, list[str]] = {}
    figures["figure1"] = _study_design(split, output / "figure1_study_design")
    figures["figure2"] = _nested_model_comparison(summary, output / "figure2_nested_model_comparison")
    figures["figure3"] = _discrimination(roc, pr, metrics, output / "figure3_locked_discrimination")
    figures["figure4"] = _external_performance(reliability, dca, three_layer, metrics, output / "figure4_external_validation_utility")
    figures["figure5"] = _global_shap(shap_result["global_values"], selected_features, low, high, output / "figure5_global_shap")
    figures["figure6"] = _local_waterfalls(
        shap_result["local_values"], shap_result["local_baselines"], shap_result["local_predictions"],
        shap_result["profiles"], selected_features, output / "figure6_local_shap_synthetic_profiles",
    )
    figures["figure7"] = _subgroup_figure(subgroups, interactions, output / "figure7_subgroup_validation")
    figures["figureS1"] = _stability_transport(stability, drift, selected_features, output / "figureS1_stability_transportability")

    selection_manifest = json.loads((selection_dir / "nested_cv_manifest.json").read_text(encoding="utf-8"))
    final_manifest = json.loads((final_dir / "final_evaluation_manifest.json").read_text(encoding="utf-8"))
    posthoc_manifest = json.loads((analysis_dir / "posthoc_manifest.json").read_text(encoding="utf-8"))
    validation = _validate_saved_outputs(
        selection_manifest, final_manifest, posthoc_manifest, summary, paired, metrics, three_layer,
        reliability, dca,
        {"max_abs_error": shap_result["additivity_max_abs_error"], "mean_abs_error": shap_result["additivity_mean_abs_error"]},
    )
    manifest = {
        "status": "publication_figure_pack_completed",
        "selected_model": model_name,
        "formats": ["png_400_dpi", "tiff_600_dpi_lzw", "pdf_vector", "svg_vector"],
        "figures": figures,
        "shap": {
            "method": "TabPFN shapiq imputation explainer; Shapley Value index; first-order effects",
            "scope": "50 outcome-stratified locked internal-test observations for global summaries",
            "local_scope": "three synthetic median/mode profiles constructed within internal-test score tertiles",
            "privacy": "No participant identifiers or participant-level explanation tables were written",
            "budget_per_row": shap_result["budget_per_row"],
            "additivity_max_abs_error": shap_result["additivity_max_abs_error"],
            "additivity_mean_abs_error": shap_result["additivity_mean_abs_error"],
            "warning": "Model association/attribution only; not causal feature effects",
        },
        "validation": validation,
    }
    (output / "publication_figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-grade MCI figures and validate saved aggregate results.")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-education", type=Path, required=True)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--posthoc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
