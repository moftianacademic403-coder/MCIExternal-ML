from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from light_localized_thresholds import _cross_fitted_localization
from light_modeling import LIGHT_SEED
from light_operating_points_recalibration import _predict_bundle
from mci_qc import write_csv
from split_development import create_locked_split


LIGHT_DCA_BOOTSTRAP_REPEATS = 300
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.801, 0.01), 2)
TARGET_SENSITIVITY = 0.85


MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svm_rbf": "RBF SVM",
    "xgboost": "XGBoost",
}


def _net_benefit(
    labels: np.ndarray,
    decisions: np.ndarray,
    threshold_probability: float,
) -> float:
    n = len(labels)
    true_positive = np.sum((labels == 1) & decisions)
    false_positive = np.sum((labels == 0) & decisions)
    odds = threshold_probability / (1.0 - threshold_probability)
    return float(true_positive / n - false_positive / n * odds)


def _bootstrap_indices(labels: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < repeats:
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size == 2:
            rows.append(indices)
    return np.asarray(rows, dtype=int)


def _brier_with_bootstrap_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    indices = _bootstrap_indices(labels, repeats, seed)
    draws = np.mean((probabilities[indices] - labels[indices]) ** 2, axis=1)
    return (
        float(np.mean((probabilities - labels) ** 2)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _bootstrap_net_benefit(
    labels: np.ndarray,
    decisions: np.ndarray,
    threshold_probability: float,
    bootstrap_indices: np.ndarray,
) -> np.ndarray:
    sampled_labels = labels[bootstrap_indices]
    sampled_decisions = decisions[bootstrap_indices]
    true_positive = np.sum((sampled_labels == 1) & sampled_decisions, axis=1)
    false_positive = np.sum((sampled_labels == 0) & sampled_decisions, axis=1)
    odds = threshold_probability / (1.0 - threshold_probability)
    return true_positive / labels.size - false_positive / labels.size * odds


def _dca_for_model(
    model_name: str,
    labels: np.ndarray,
    locked_probabilities: np.ndarray,
    local_recalibrated_probabilities: np.ndarray,
    development_rule: np.ndarray,
    local_rule: np.ndarray,
) -> pd.DataFrame:
    bootstrap_indices = _bootstrap_indices(
        labels,
        LIGHT_DCA_BOOTSTRAP_REPEATS,
        LIGHT_SEED + sum(ord(char) for char in model_name) + 811,
    )
    bootstrap_counts = np.vstack(
        [np.bincount(indices, minlength=len(labels)) for indices in bootstrap_indices]
    ).astype(float)
    threshold_odds = DCA_THRESHOLDS / (1.0 - DCA_THRESHOLDS)
    strategy_decisions = {
        "locked_development_probability_model": (
            locked_probabilities[:, None] >= DCA_THRESHOLDS[None, :]
        ),
        "local_recalibrated_oof_probability_model": (
            local_recalibrated_probabilities[:, None] >= DCA_THRESHOLDS[None, :]
        ),
        "locked_development_threshold_85_binary_rule": development_rule,
        "local_threshold_85_oof_binary_rule": local_rule,
        "screen_all": np.ones(len(labels), dtype=bool),
        "screen_none": np.zeros(len(labels), dtype=bool),
    }
    estimates: dict[str, np.ndarray] = {}
    draws: dict[str, np.ndarray] = {}
    positive = labels == 1
    negative = labels == 0
    for strategy, decisions in strategy_decisions.items():
        if decisions.ndim == 2:
            true_positive = np.sum(positive[:, None] & decisions, axis=0)
            false_positive = np.sum(negative[:, None] & decisions, axis=0)
            bootstrap_true_positive = bootstrap_counts @ (
                positive[:, None] & decisions
            )
            bootstrap_false_positive = bootstrap_counts @ (
                negative[:, None] & decisions
            )
        else:
            true_positive = np.repeat(np.sum(positive & decisions), len(DCA_THRESHOLDS))
            false_positive = np.repeat(np.sum(negative & decisions), len(DCA_THRESHOLDS))
            bootstrap_true_positive = np.repeat(
                (bootstrap_counts @ (positive & decisions).astype(float))[:, None],
                len(DCA_THRESHOLDS),
                axis=1,
            )
            bootstrap_false_positive = np.repeat(
                (bootstrap_counts @ (negative & decisions).astype(float))[:, None],
                len(DCA_THRESHOLDS),
                axis=1,
            )
        estimates[strategy] = (
            true_positive / len(labels)
            - false_positive / len(labels) * threshold_odds
        )
        draws[strategy] = (
            bootstrap_true_positive / len(labels)
            - bootstrap_false_positive / len(labels) * threshold_odds[None, :]
        )

    rows = []
    for threshold_index, threshold_probability in enumerate(DCA_THRESHOLDS):
        all_draws = draws["screen_all"][:, threshold_index]
        none_draws = draws["screen_none"][:, threshold_index]
        all_estimate = estimates["screen_all"][threshold_index]
        for strategy in strategy_decisions:
            estimate = estimates[strategy][threshold_index]
            current_draws = draws[strategy][:, threshold_index]
            delta_all = current_draws - all_draws
            delta_none = current_draws - none_draws
            rows.append(
                {
                    "partition": "external_validation_or_local_updating_as_labeled",
                    "model_name": model_name,
                    "strategy": strategy,
                    "threshold_probability": threshold_probability,
                    "net_benefit": estimate,
                    "ci_lower": float(np.quantile(current_draws, 0.025)),
                    "ci_upper": float(np.quantile(current_draws, 0.975)),
                    "delta_vs_screen_all": estimate - all_estimate,
                    "delta_vs_screen_all_ci_lower": float(
                        np.quantile(delta_all, 0.025)
                    ),
                    "delta_vs_screen_all_ci_upper": float(
                        np.quantile(delta_all, 0.975)
                    ),
                    "delta_vs_screen_none": estimate,
                    "delta_vs_screen_none_ci_lower": float(
                        np.quantile(delta_none, 0.025)
                    ),
                    "delta_vs_screen_none_ci_upper": float(
                        np.quantile(delta_none, 0.975)
                    ),
                    "ci_method": (
                        "paired participant bootstrap, "
                        f"{LIGHT_DCA_BOOTSTRAP_REPEATS} repeats"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
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
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "observed": labels,
            "predicted": probabilities,
            "rank": pd.Series(probabilities).rank(method="first"),
        }
    )
    frame["bin"] = pd.qcut(
        frame["rank"],
        q=bins,
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_number, group in frame.groupby("bin", observed=True):
        successes = int(group["observed"].sum())
        lower, upper = _wilson_interval(successes, len(group))
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
                "binning": "10 equal-frequency bins",
            }
        )
    return pd.DataFrame(rows)


def _metric_lookup(
    frame: pd.DataFrame,
    filters: dict[str, object],
) -> dict[str, tuple[float, float, float]]:
    current = frame.copy()
    for column, value in filters.items():
        current = current.loc[current[column].eq(value)]
    return {
        row.metric: (float(row.estimate), float(row.ci_lower), float(row.ci_upper))
        for row in current.itertuples()
    }


def _three_layer_rows(
    model_name: str,
    operating_metrics: pd.DataFrame,
    localized_metrics: pd.DataFrame,
    brier_decomposition: pd.DataFrame,
    locked_uncalibrated_brier: tuple[float, float, float],
) -> list[dict]:
    development = _metric_lookup(
        operating_metrics,
        {
            "partition": "external_validation",
            "model_name": model_name,
            "target_sensitivity_from_train_oof": TARGET_SENSITIVITY,
        },
    )
    raw_local = _metric_lookup(
        localized_metrics,
        {
            "model_name": model_name,
            "probability_source": "raw_local_threshold",
            "target_sensitivity": TARGET_SENSITIVITY,
        },
    )
    recalibrated_local = _metric_lookup(
        localized_metrics,
        {
            "model_name": model_name,
            "probability_source": "recalibrated_local_threshold",
            "target_sensitivity": TARGET_SENSITIVITY,
        },
    )

    def brier(analysis: str) -> tuple[float, float, float]:
        current = brier_decomposition.loc[
            brier_decomposition["model_name"].eq(model_name)
            & brier_decomposition["analysis"].eq(analysis)
            & brier_decomposition["component"].eq("brier_score")
        ].iloc[0]
        return float(current.estimate), float(current.ci_lower), float(current.ci_upper)

    locked_platt_brier = brier("development_locked_platt")
    local_recalibrated_brier = brier(
        "external_10fold_oof_logistic_recalibration"
    )
    specifications = (
        {
            "layer": "A_locked_model_plus_development_threshold_85",
            "probability_source": "locked uncalibrated Development model",
            "threshold_source": "Development train-80 OOF",
            "evaluation_design": "unchanged locked external validation",
            "metrics": development,
            "brier": locked_uncalibrated_brier,
            "interpretation": (
                "Primary transport evaluation; no External adaptation."
            ),
        },
        {
            "layer": "B_locked_model_plus_local_threshold_85",
            "probability_source": "locked Development Platt probabilities",
            "threshold_source": "External 10-fold training subsets",
            "evaluation_design": "10-fold OOF local threshold updating",
            "metrics": raw_local,
            "brier": locked_platt_brier,
            "interpretation": (
                "Secondary local threshold update; model coefficients unchanged."
            ),
        },
        {
            "layer": "C_locked_model_plus_local_recalibration_and_threshold_85",
            "probability_source": "External 10-fold OOF logistic recalibration",
            "threshold_source": "External 10-fold training subsets",
            "evaluation_design": "10-fold OOF recalibration plus threshold updating",
            "metrics": recalibrated_local,
            "brier": local_recalibrated_brier,
            "interpretation": (
                "Secondary local probability recalibration; not a full base-model refit."
            ),
        },
    )
    rows = []
    for specification in specifications:
        row = {
            "model_name": model_name,
            "layer": specification["layer"],
            "probability_source": specification["probability_source"],
            "threshold_source": specification["threshold_source"],
            "evaluation_design": specification["evaluation_design"],
            "interpretation": specification["interpretation"],
        }
        for metric in ("sensitivity", "specificity", "ppv", "npv"):
            estimate, lower, upper = specification["metrics"][metric]
            row[metric] = estimate
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        estimate, lower, upper = specification["brier"]
        row["brier_score"] = estimate
        row["brier_ci_lower"] = lower
        row["brier_ci_upper"] = upper
        rows.append(row)
    return rows


def _plot_dca(dca: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    styles = {
        "locked_development_probability_model": ("#2563eb", "-", 2.2),
        "local_recalibrated_oof_probability_model": ("#ea580c", "-", 2.2),
        "locked_development_threshold_85_binary_rule": ("#2563eb", ":", 1.5),
        "local_threshold_85_oof_binary_rule": ("#ea580c", ":", 1.5),
        "screen_all": ("#525252", "--", 1.5),
        "screen_none": ("#000000", "-", 1.0),
    }
    labels = {
        "locked_development_probability_model": "Locked probability model",
        "local_recalibrated_oof_probability_model": "Local recalibration (OOF)",
        "locked_development_threshold_85_binary_rule": "Development 85% rule",
        "local_threshold_85_oof_binary_rule": "Local 85% rule (OOF)",
        "screen_all": "Screen all",
        "screen_none": "Screen none",
    }
    display = dca.loc[dca["threshold_probability"].le(0.50)]
    for axis, model_name in zip(axes, MODEL_LABELS):
        current = display.loc[display["model_name"].eq(model_name)]
        for strategy, (colour, linestyle, width) in styles.items():
            line = current.loc[current["strategy"].eq(strategy)].sort_values(
                "threshold_probability"
            )
            x = line["threshold_probability"].to_numpy(dtype=float)
            y = line["net_benefit"].to_numpy(dtype=float)
            axis.plot(
                x,
                y,
                color=colour,
                linestyle=linestyle,
                linewidth=width,
                label=labels[strategy],
            )
            if strategy in {
                "locked_development_probability_model",
                "local_recalibrated_oof_probability_model",
            }:
                axis.fill_between(
                    x,
                    line["ci_lower"].to_numpy(dtype=float),
                    line["ci_upper"].to_numpy(dtype=float),
                    color=colour,
                    alpha=0.10,
                    linewidth=0,
                )
        axis.set_title(MODEL_LABELS[model_name], fontweight="semibold")
        axis.grid(alpha=0.20)
        axis.set_xlim(0.05, 0.50)
        axis.set_ylim(-0.05, 0.53)
    for axis in axes[2:]:
        axis.set_xlabel("Threshold probability")
    for axis in axes[::2]:
        axis.set_ylabel("Net benefit")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "External decision curves: locked model and local updating",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Shaded bands: preliminary 95% paired-bootstrap CIs; displayed range 0.05–0.50",
        ha="center",
        fontsize=10,
        color="#404040",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(bins: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    styles = {
        "development_locked_platt": ("#2563eb", "o", "Locked Development model"),
        "external_10fold_oof_logistic_recalibration": (
            "#ea580c",
            "s",
            "Local recalibration (OOF)",
        ),
    }
    for axis, model_name in zip(axes, MODEL_LABELS):
        axis.plot([0, 1], [0, 1], color="#525252", linestyle="--", linewidth=1.2)
        for analysis, (colour, marker, label) in styles.items():
            current = bins.loc[
                bins["model_name"].eq(model_name)
                & bins["analysis"].eq(analysis)
            ].sort_values("mean_predicted")
            x = current["mean_predicted"].to_numpy(dtype=float)
            y = current["observed_rate"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    y - current["observed_rate_ci_lower"].to_numpy(dtype=float),
                    current["observed_rate_ci_upper"].to_numpy(dtype=float) - y,
                ]
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                color=colour,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                capsize=2,
                label=label,
            )
        axis.set_title(MODEL_LABELS[model_name], fontweight="semibold")
        axis.grid(alpha=0.20)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal", adjustable="box")
    for axis in axes[2:]:
        axis.set_xlabel("Mean predicted probability")
    for axis in axes[::2]:
        axis.set_ylabel("Observed MCI proportion")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "External reliability diagrams before and after local recalibration",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Ten equal-frequency bins; vertical bars are Wilson 95% CIs",
        ha="center",
        fontsize=10,
        color="#404040",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_light_dca_reliability_summary(
    development_path: Path,
    external_path: Path,
    qc_output_dir: Path,
    light_output_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    split = create_locked_split(development_path, external_path, qc_output_dir)
    external = split["external_harmonized_in_memory"].reset_index(drop=True)
    thresholds = pd.read_csv(
        light_output_dir / "light_operating_point_thresholds.csv",
        encoding="utf-8-sig",
    )
    operating_metrics = pd.read_csv(
        light_output_dir / "light_operating_point_metrics.csv",
        encoding="utf-8-sig",
    )
    localized_metrics = pd.read_csv(
        light_output_dir / "light_external_localized_threshold_metrics.csv",
        encoding="utf-8-sig",
    )
    brier_decomposition = pd.read_csv(
        light_output_dir / "light_external_brier_decomposition.csv",
        encoding="utf-8-sig",
    )

    dca_frames = []
    reliability_frames = []
    summary_rows = []
    for calibrated_path in sorted(
        (light_output_dir / "calibrated_models").glob("*_platt.joblib")
    ):
        calibrated_bundle = joblib.load(calibrated_path)
        model_name = calibrated_bundle["model_name"]
        labels, locked_probabilities = _predict_bundle(calibrated_bundle, external)
        local_probabilities, local_decisions, _ = _cross_fitted_localization(
            labels,
            locked_probabilities,
        )

        uncalibrated_bundle = joblib.load(
            light_output_dir / "models" / f"{model_name}.joblib"
        )
        uncalibrated_labels, uncalibrated_probabilities = _predict_bundle(
            uncalibrated_bundle,
            external,
        )
        if not np.array_equal(labels, uncalibrated_labels):
            raise RuntimeError(f"Label order differs for {model_name} bundles.")
        development_threshold = float(
            thresholds.loc[
                thresholds["model_name"].eq(model_name)
                & thresholds["target_sensitivity"].eq(TARGET_SENSITIVITY),
                "threshold",
            ].iloc[0]
        )
        development_rule = uncalibrated_probabilities >= development_threshold
        local_rule = local_decisions[
            ("recalibrated_local_threshold", TARGET_SENSITIVITY)
        ]
        dca_frames.append(
            _dca_for_model(
                model_name,
                labels,
                locked_probabilities,
                local_probabilities,
                development_rule,
                local_rule,
            )
        )
        reliability_frames.extend(
            [
                _reliability_bins(
                    labels,
                    locked_probabilities,
                    model_name,
                    "development_locked_platt",
                ),
                _reliability_bins(
                    labels,
                    local_probabilities,
                    model_name,
                    "external_10fold_oof_logistic_recalibration",
                ),
            ]
        )
        summary_rows.extend(
            _three_layer_rows(
                model_name,
                operating_metrics,
                localized_metrics,
                brier_decomposition,
                _brier_with_bootstrap_interval(
                    labels,
                    uncalibrated_probabilities,
                    LIGHT_DCA_BOOTSTRAP_REPEATS,
                    LIGHT_SEED
                    + sum(ord(char) for char in model_name)
                    + 1229,
                ),
            )
        )

    dca = pd.concat(dca_frames, ignore_index=True)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    three_layer_summary = pd.DataFrame(summary_rows)
    selected_models = pd.read_csv(
        light_output_dir / "light_selected_models.csv",
        encoding="utf-8-sig",
    )
    model_selection_status = selected_models[
        ["model_name", "family", "mean_inner_auc", "sd_inner_auc"]
    ].copy()
    model_selection_status["current_evidence"] = (
        "single lightweight Development inner-CV screen"
    )
    model_selection_status["final_selection_allowed"] = False
    model_selection_status["required_next_evidence"] = (
        "Development-only repeated nested CV including TabPFN"
    )
    model_selection_status["external_used_for_selection"] = False

    figures_dir = light_output_dir / "figures"
    dca_figure = figures_dir / "dca_updated_external.png"
    reliability_figure = figures_dir / "reliability_before_after_external.png"
    _plot_dca(dca, dca_figure)
    _plot_reliability(reliability, reliability_figure)

    write_csv(dca, light_output_dir / "light_updated_external_dca.csv")
    write_csv(
        reliability,
        light_output_dir / "light_updated_external_reliability_bins.csv",
    )
    write_csv(
        three_layer_summary,
        light_output_dir / "light_three_layer_model_summary.csv",
    )
    write_csv(
        model_selection_status,
        light_output_dir / "light_development_model_selection_status.csv",
    )
    manifest = {
        "status": "lightweight_preliminary_not_for_manuscript_results",
        "dca": {
            "threshold_probability_grid": [
                float(DCA_THRESHOLDS.min()),
                float(DCA_THRESHOLDS.max()),
                0.01,
            ],
            "figure_display_range": [0.05, 0.50],
            "bootstrap_repeats": LIGHT_DCA_BOOTSTRAP_REPEATS,
            "ci_interpretation": (
                "descriptive uncertainty; model usefulness is not conditioned "
                "on statistical significance of net-benefit contrasts"
            ),
            "important_distinction": (
                "DCA threshold probability encodes clinical harm-benefit tradeoff; "
                "it is not the score cutoff chosen to target 85% sensitivity"
            ),
        },
        "reliability_diagram": {
            "bins": "10 equal-frequency",
            "observed_rate_interval": "Wilson 95% CI",
            "local_probabilities": "10-fold OOF logistic recalibration",
        },
        "three_layer_table": {
            "A": "locked model plus Development threshold",
            "B": "locked model plus locally updated threshold",
            "C": "locked model plus local logistic recalibration and local threshold",
            "full_local_refit_performed": False,
        },
        "model_selection": {
            "final_model_selected": False,
            "reason": (
                "requires Development-only heavy repeated nested CV including TabPFN"
            ),
            "external_used_for_selection": False,
        },
        "participant_level_predictions_written": False,
        "figures": [str(dca_figure), str(reliability_figure)],
    }
    manifest_path = light_output_dir / "light_dca_reliability_summary_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "dca": dca,
        "reliability": reliability,
        "three_layer_summary": three_layer_summary,
        "model_selection_status": model_selection_status,
        "manifest": manifest,
    }
