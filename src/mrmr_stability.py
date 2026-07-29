from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedShuffleSplit

from mci_qc import sha256_file, write_csv
from split_development import create_locked_split


MRMR_SEED = 20260729
N_REPEATS = 30
SUBSAMPLE_FRACTION = 0.80
N_NUMERIC_BINS = 10
K_VALUES = (5, 10, 15, 20, 25, 30)


def _discretize_numeric(series: pd.Series, bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(-1, index=series.index, dtype="int64")
    nonmissing = values.notna()
    if int(nonmissing.sum()) == 0:
        return result.to_numpy()
    unique_count = int(values[nonmissing].nunique())
    if unique_count == 1:
        result.loc[nonmissing] = 0
        return result.to_numpy()
    effective_bins = min(bins, unique_count)
    ranked = values.loc[nonmissing].rank(method="average")
    discretized = pd.qcut(
        ranked,
        q=effective_bins,
        labels=False,
        duplicates="drop",
    )
    result.loc[nonmissing] = discretized.astype("int64")
    return result.to_numpy()


def _discretize_categorical(series: pd.Series) -> np.ndarray:
    values = series.astype("string").fillna("<missing>")
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.int64)


def discretize_frame(
    frame: pd.DataFrame,
    variable_types: dict[str, str],
) -> dict[str, np.ndarray]:
    discrete = {}
    for column in frame.columns:
        if variable_types[column] == "numeric":
            discrete[column] = _discretize_numeric(frame[column], N_NUMERIC_BINS)
        else:
            discrete[column] = _discretize_categorical(frame[column])
    return discrete


def rank_mrmr(
    frame: pd.DataFrame,
    labels: pd.Series,
    variable_types: dict[str, str],
) -> tuple[list[str], dict[str, float]]:
    columns = list(frame.columns)
    discrete = discretize_frame(frame, variable_types)
    y = labels.map({"no": 0, "yes": 1}).to_numpy(dtype=np.int64)
    relevance = {
        column: float(
            normalized_mutual_info_score(
                discrete[column],
                y,
                average_method="arithmetic",
            )
        )
        for column in columns
    }
    redundancy = {}
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            score = float(
                normalized_mutual_info_score(
                    discrete[left],
                    discrete[right],
                    average_method="arithmetic",
                )
            )
            redundancy[(left, right)] = score
            redundancy[(right, left)] = score

    selected: list[str] = []
    remaining = set(columns)
    while remaining:
        if not selected:
            best = min(
                remaining,
                key=lambda column: (-relevance[column], column),
            )
        else:
            def score(column: str) -> float:
                mean_redundancy = float(
                    np.mean([redundancy[(column, chosen)] for chosen in selected])
                )
                return relevance[column] - mean_redundancy

            best = min(remaining, key=lambda column: (-score(column), column))
        selected.append(best)
        remaining.remove(best)
    return selected, relevance


def _set_stability(rankings: list[list[str]], feature_count: int) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        sets = [set(ranking[:k]) for ranking in rankings]
        jaccard_scores = []
        kuncheva_scores = []
        for left, right in itertools.combinations(sets, 2):
            intersection = len(left & right)
            union = len(left | right)
            jaccard_scores.append(intersection / union)
            kuncheva_scores.append(
                (intersection * feature_count - k * k) / (k * (feature_count - k))
            )
        rows.append(
            {
                "k": k,
                "pair_count": len(jaccard_scores),
                "mean_pairwise_jaccard": float(np.mean(jaccard_scores)),
                "sd_pairwise_jaccard": float(np.std(jaccard_scores, ddof=1)),
                "mean_pairwise_kuncheva": float(np.mean(kuncheva_scores)),
                "sd_pairwise_kuncheva": float(np.std(kuncheva_scores, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def run_mrmr_stability(
    development_path: Path,
    external_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    split = create_locked_split(development_path, external_path, output_dir)
    development = split["development_eligible_in_memory"]
    train_relative = split["train_relative_indices"]
    registry = split["registry"]
    predictors = registry.loc[registry["role"] == "predictor", "canonical_name"].tolist()
    variable_types = registry.set_index("canonical_name")["variable_type"].to_dict()
    train_frame = development.iloc[train_relative][predictors].reset_index(drop=True)
    train_labels = development.iloc[train_relative]["mci"].reset_index(drop=True)

    sampler = StratifiedShuffleSplit(
        n_splits=N_REPEATS,
        train_size=SUBSAMPLE_FRACTION,
        random_state=MRMR_SEED,
    )
    rankings: list[list[str]] = []
    relevance_records = []
    for repeat_index, (subsample_indices, _) in enumerate(
        sampler.split(train_frame, train_labels),
        start=1,
    ):
        ranking, relevance = rank_mrmr(
            train_frame.iloc[subsample_indices],
            train_labels.iloc[subsample_indices],
            {column: variable_types[column] for column in predictors},
        )
        rankings.append(ranking)
        relevance_records.extend(
            {
                "repeat": repeat_index,
                "canonical_name": feature,
                "rank": rank,
                "normalized_mi_relevance": relevance[feature],
            }
            for rank, feature in enumerate(ranking, start=1)
        )

    long_results = pd.DataFrame(relevance_records)
    stability_rows = []
    for feature in predictors:
        subset = long_results.loc[long_results["canonical_name"] == feature]
        row = {
            "canonical_name": feature,
            "variable_type": variable_types[feature],
            "mean_normalized_mi_relevance": subset["normalized_mi_relevance"].mean(),
            "median_rank": subset["rank"].median(),
            "mean_rank": subset["rank"].mean(),
            "rank_q1": subset["rank"].quantile(0.25),
            "rank_q3": subset["rank"].quantile(0.75),
            "rank_min": int(subset["rank"].min()),
            "rank_max": int(subset["rank"].max()),
        }
        for k in K_VALUES:
            row[f"selection_frequency_top_{k}"] = float((subset["rank"] <= k).mean())
        stability_rows.append(row)
    rank_stability = pd.DataFrame(stability_rows).sort_values(
        ["median_rank", "mean_rank", "canonical_name"]
    )
    set_stability = _set_stability(rankings, len(predictors))

    write_csv(rank_stability, output_dir / "mrmr_rank_stability.csv")
    write_csv(set_stability, output_dir / "mrmr_set_stability.csv")
    write_csv(long_results, output_dir / "mrmr_repeat_long.csv")
    manifest = {
        "development_source_sha256": split["manifest"]["development_sha256"],
        "harmonization_registry_sha256": sha256_file(output_dir / "harmonization_registry.csv"),
        "split_manifest_sha256": sha256_file(output_dir / "split_manifest.json"),
        "input_partition": "train_80_only",
        "train_rows": int(len(train_frame)),
        "internal_test_rows_used": 0,
        "external_rows_used": 0,
        "feature_count": int(len(predictors)),
        "repeats": N_REPEATS,
        "resampling": "StratifiedShuffleSplit without replacement",
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "subsample_rows_per_repeat": int(round(len(train_frame) * SUBSAMPLE_FRACTION)),
        "random_seed": MRMR_SEED,
        "numeric_discretization": f"within-repeat rank quantile bins, maximum {N_NUMERIC_BINS}; missing=-1",
        "categorical_encoding": "within-repeat deterministic factorization; missing explicit category",
        "relevance": "normalized mutual information with MCI",
        "redundancy": "pairwise normalized mutual information",
        "criterion": "relevance minus mean redundancy (mRMR difference)",
        "k_values_reported": list(K_VALUES),
        "k_selected": None,
        "selection_policy": "k will be tuned inside nested CV; this notebook reports stability only",
        "participant_level_outputs_written": False,
    }
    (output_dir / "mrmr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "rank_stability": rank_stability,
        "set_stability": set_stability,
        "long_results": long_results,
        "manifest": manifest,
    }

