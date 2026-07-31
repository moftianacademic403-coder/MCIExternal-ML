from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

from harmonization import build_harmonized_frames, run_harmonization_resolution
from mci_qc import load_inputs, sha256_file, write_csv


SPLIT_SEED = 20260728
TEST_SIZE = 0.20


def _index_hash(index_values: np.ndarray) -> str:
    normalized = np.asarray(index_values, dtype=np.int64)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _class_counts(labels: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in labels.value_counts(dropna=False).sort_index().items()
    }


def create_locked_split(
    development_path: Path,
    external_path: Path,
    output_dir: Path,
    external_education_path: Path | None = None,
) -> dict:
    harmonization = run_harmonization_resolution(
        development_path,
        external_path,
        output_dir,
        external_education_path=external_education_path,
    )
    registry = harmonization["registry"]
    unresolved = harmonization["unresolved"]
    if not unresolved.empty:
        raise ValueError(
            "Cannot create a locked split while harmonization rules remain unresolved: "
            + ", ".join(unresolved["development_column_or_expression"].tolist())
        )

    development_raw = harmonization["development_raw_in_memory"]
    external_raw = harmonization["external_raw_in_memory"]
    source_metadata = harmonization["source_metadata"]
    development, external = build_harmonized_frames(
        development_raw,
        external_raw,
        registry,
    )

    outcome = development["mci"]
    valid_outcome = outcome.isin(["yes", "no"])
    excluded_outcome_missing_n = int((~valid_outcome).sum())
    development_model = development.loc[valid_outcome].copy()
    original_row_positions = np.flatnonzero(valid_outcome.to_numpy())
    labels = development_model["mci"]
    predictors = registry.loc[registry["role"] == "predictor", "canonical_name"].tolist()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
    )
    train_relative, test_relative = next(
        splitter.split(development_model[predictors], labels)
    )
    train_original = original_row_positions[train_relative]
    test_original = original_row_positions[test_relative]
    train_labels = labels.iloc[train_relative]
    test_labels = labels.iloc[test_relative]

    split_summary = pd.DataFrame(
        [
            {
                "partition": "development_eligible",
                "rows": int(len(development_model)),
                "mci_yes_n": int(labels.eq("yes").sum()),
                "mci_no_n": int(labels.eq("no").sum()),
                "mci_yes_pct": round(100 * labels.eq("yes").mean(), 3),
            },
            {
                "partition": "train_80",
                "rows": int(len(train_relative)),
                "mci_yes_n": int(train_labels.eq("yes").sum()),
                "mci_no_n": int(train_labels.eq("no").sum()),
                "mci_yes_pct": round(100 * train_labels.eq("yes").mean(), 3),
            },
            {
                "partition": "internal_test_20_locked",
                "rows": int(len(test_relative)),
                "mci_yes_n": int(test_labels.eq("yes").sum()),
                "mci_no_n": int(test_labels.eq("no").sum()),
                "mci_yes_pct": round(100 * test_labels.eq("yes").mean(), 3),
            },
            {
                "partition": "external_reserved",
                "rows": int(len(external)),
                "mci_yes_n": np.nan,
                "mci_no_n": np.nan,
                "mci_yes_pct": np.nan,
            },
        ]
    )
    write_csv(split_summary, output_dir / "split_summary.csv")

    matrix_validation = []
    for feature in predictors:
        matrix_validation.append(
            {
                "canonical_name": feature,
                "development_missing_pct": round(100 * development_model[feature].isna().mean(), 3),
                "train_missing_pct": round(100 * development_model.iloc[train_relative][feature].isna().mean(), 3),
                "internal_test_missing_pct": round(100 * development_model.iloc[test_relative][feature].isna().mean(), 3),
                "external_missing_pct_structural_only": round(100 * external[feature].isna().mean(), 3),
            }
        )
    matrix_validation_frame = pd.DataFrame(matrix_validation)
    write_csv(matrix_validation_frame, output_dir / "harmonized_matrix_validation.csv")

    split_manifest = {
        "development_sha256": source_metadata["development_sha256"],
        "external_sha256": source_metadata["external_sha256"],
        "harmonization_registry_sha256": sha256_file(output_dir / "harmonization_registry.csv"),
        "split_algorithm": "sklearn.model_selection.StratifiedShuffleSplit",
        "sklearn_version": sklearn.__version__,
        "split_seed": SPLIT_SEED,
        "test_size": TEST_SIZE,
        "eligible_development_rows": int(len(development_model)),
        "excluded_missing_or_invalid_mci_rows": excluded_outcome_missing_n,
        "predictor_count": int(len(predictors)),
        "outcome": "mci",
        "eligible_class_counts": _class_counts(labels),
        "train_rows": int(len(train_relative)),
        "train_class_counts": _class_counts(train_labels),
        "internal_test_rows": int(len(test_relative)),
        "internal_test_class_counts": _class_counts(test_labels),
        "train_original_row_position_sha256": _index_hash(train_original),
        "internal_test_original_row_position_sha256": _index_hash(test_original),
        "external_rows_reserved": int(len(external)),
        "external_policy": "reserved_for_locked_external_validation_only",
        "external_outcome_used_during_split": False,
        "participant_level_harmonized_files_written": False,
        "participant_level_split_assignments_written": False,
        "correlation_pruning_applied": False,
        "mrmr_candidate_predictors_before_ranking": int(len(predictors)),
        "education_harmonization_mode": (
            "four_level_code_matched_auxiliary_source"
            if external_education_path is not None
            else "three_level_collapsed_external_source"
        ),
        "reconstruction": "Re-run this function with identical source hashes, registry hash, sklearn version, seed, and row order.",
    }
    if harmonization["four_level_education_audit"] is not None:
        split_manifest["four_level_education_audit"] = harmonization[
            "four_level_education_audit"
        ]
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest": split_manifest,
        "summary": split_summary,
        "matrix_validation": matrix_validation_frame,
        "registry": registry,
        "development_harmonized_in_memory": development,
        "development_eligible_in_memory": development_model,
        "external_harmonized_in_memory": external,
        "train_relative_indices": train_relative,
        "test_relative_indices": test_relative,
    }
