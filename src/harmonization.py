from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mci_qc import load_inputs, write_csv


KEYS = ["development_column_or_expression", "external_column_or_expression"]

NUMERIC_IDENTITY = {
    "age": "age",
    "BMI": "bmi",
    "waist_circumference_cm": "waist_circumference_cm",
    "waist_hip_ratio": "waist_hip_ratio",
    "hip_circumference_cm": "hip_circumference_cm",
    "PLT": "plt",
    "HCT": "hct",
    "Triglycerides": "triglycerides",
    "rbc": "rbc",
    "wbc": "wbc",
    "hemoglobin": "hemoglobin",
    "total_cholesterol": "total_cholesterol",
    "hdl_cholesterol": "hdl_cholesterol",
    "ldl_cholesterol": "ldl_cholesterol",
    "uric_acid": "uric_acid",
    "ast": "ast",
    "alt": "alt",
    "vitamin_d": "vitamin_d",
    "whisper_test_left": "whisper_test_left",
    "whisper_test_right": "whisper_test_right",
    "visual_acuity_both": "visual_acuity_both",
    "TOTAL_PFAT": "total_pfat",
    "ANDROID_PFAT": "android_pfat",
    "GYNOID_PFAT": "gynoid_pfat",
}

CATEGORICAL_IDENTITY = {
    "sex": "sex",
    "employment_status": "employment_status",
    "sleep_quality": "sleep_quality",
    "adl": "adl",
    "iadl": "iadl",
    "osteoporosis_status": "osteoporosis_status",
    "depression_status": "depression_status",
    "diabetes": "diabetes",
    "hypertension": "hypertension",
}

HISTORY_COLUMNS = {
    "history_heart_failure": "history_heart_failure",
    "history_stroke": "history_stroke",
    "history_myocardial_infarction": "history_myocardial_infarction",
}

UNRESOLVED_REQUIREMENTS: dict[str, tuple[str, str]] = {}


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA).str.casefold()


def _clean_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    stripped = series.astype("string").str.strip().replace("", pd.NA)
    numeric = pd.to_numeric(stripped, errors="coerce")
    invalid = int((stripped.notna() & numeric.isna()).sum())
    return numeric, invalid


def _yes(response: str) -> bool:
    return str(response).strip().casefold() in {"بله", "yes", "true", "1"}


def _response_confirms_rule(development_expression: str, response: str) -> bool:
    normalized = str(response).strip().casefold()
    if development_expression == "household_income":
        required = {"poorest", "poorer", "low", "middle", "moderate", "richer", "richest", "high"}
        return all(token in normalized for token in required)
    if development_expression == "EDU":
        required = {
            "illitrate",
            "primary",
            "illitrate&primary school",
            "secendry&high",
            "secondary school",
            "academic",
        }
        return all(token in normalized for token in required)
    if development_expression == "housing_status":
        required = {
            "living with one of children and their family",
            "living with siblings or a family member and their family",
            "living with spouses",
            "with child & family’s&with brother &sister",
            "label_independent",
            "label_extended",
        }
        return all(token in normalized for token in required)
    if development_expression == "mean_sitting_min_day":
        return all(token in normalized for token in {"sit_work", "sit_weekend", "5", "2", "7"})
    if development_expression == "max_handgrip_right + max_handgrip_left":
        return all(token in normalized for token in {"دست موجود", "هر دو دست", "missing"})
    return _yes(response)


def _rule_for_row(development_expression: str, external_expression: str) -> dict | None:
    if development_expression in NUMERIC_IDENTITY:
        return {
            "canonical_name": NUMERIC_IDENTITY[development_expression],
            "role": "predictor",
            "variable_type": "numeric",
            "transformation": "blank_to_missing_then_numeric_identity",
        }
    if development_expression in CATEGORICAL_IDENTITY:
        return {
            "canonical_name": CATEGORICAL_IDENTITY[development_expression],
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "trim_casefold_identity",
        }
    if development_expression == "marital_status":
        return {
            "canonical_name": "marital_status",
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_trim_casefold; external_m_to_married_s_to_single",
        }
    if development_expression == "housing_status":
        return {
            "canonical_name": "housing_status",
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_living_with_spouses_to_label_independent_combined_family_to_label_extended; external_independent_to_label_independent_child_or_siblings_family_to_label_extended",
        }
    if development_expression == "smoking_status":
        return {
            "canonical_name": "current_smoker",
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_regular_or_somtimes_to_yes_no_to_no; external_trim_casefold",
        }
    if development_expression == "household_income":
        return {
            "canonical_name": "household_income",
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_low_moderate_high_identity; external_poorest_or_poorer_to_low_middle_to_moderate_richer_or_richest_to_high",
        }
    if development_expression == "EDU":
        return {
            "canonical_name": "education",
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_illitrate_or_primary_to_illitrate_and_primary_school_secendry_and_high_to_secondary_school_academic_identity; external_trim_casefold",
        }
    if development_expression == "mean_sitting_min_day":
        return {
            "canonical_name": "mean_sitting_min_day",
            "role": "predictor",
            "variable_type": "numeric",
            "transformation": "development_numeric_identity; external_weighted_weekly_mean_5_sit_work_plus_2_sit_weekend_divided_by_7",
        }
    if development_expression == "max_handgrip_right + max_handgrip_left":
        return {
            "canonical_name": "max_handgrip",
            "role": "predictor",
            "variable_type": "numeric",
            "transformation": "development_rowwise_max_available_hand_both_missing_to_missing; external_max_hndgrp_zero_to_missing_then_numeric",
        }
    if development_expression in HISTORY_COLUMNS:
        return {
            "canonical_name": HISTORY_COLUMNS[development_expression],
            "role": "predictor",
            "variable_type": "categorical",
            "transformation": "development_99_to_missing_then_trim_casefold; external_trim_casefold",
        }
    if development_expression == "MCI":
        return {
            "canonical_name": "mci",
            "role": "outcome",
            "variable_type": "categorical",
            "transformation": "trim_casefold_yes_no",
        }
    return None


def resolve_questionnaire(questionnaire_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    questionnaire = pd.read_csv(questionnaire_path, encoding="utf-8-sig", dtype="string").fillna("")
    registry_rows = []
    for index, row in questionnaire.iterrows():
        development_expression = row["development_column_or_expression"]
        external_expression = row["external_column_or_expression"]
        response = row["user_response"]
        if development_expression in UNRESOLVED_REQUIREMENTS:
            status, explanation = UNRESOLVED_REQUIREMENTS[development_expression]
            questionnaire.at[index, "status"] = status
            questionnaire.at[index, "allowed_for_external_model"] = "FALSE"
            questionnaire.at[index, "final_harmonization_rule"] = explanation
            continue
        if not _response_confirms_rule(development_expression, response):
            questionnaire.at[index, "status"] = "needs_response_or_clarification"
            questionnaire.at[index, "allowed_for_external_model"] = "FALSE"
            questionnaire.at[index, "final_harmonization_rule"] = ""
            continue
        rule = _rule_for_row(development_expression, external_expression)
        if rule is None:
            questionnaire.at[index, "status"] = "needs_explicit_rule"
            questionnaire.at[index, "allowed_for_external_model"] = "FALSE"
            questionnaire.at[index, "final_harmonization_rule"] = ""
            continue
        questionnaire.at[index, "status"] = "confirmed"
        questionnaire.at[index, "allowed_for_external_model"] = "TRUE"
        questionnaire.at[index, "final_harmonization_rule"] = rule["transformation"]
        registry_rows.append(
            {
                **{key: row[key] for key in KEYS},
                **rule,
                "source_confirmation": response,
                "status": "confirmed",
                "eligible_for_locked_external_matrix": True,
            }
        )
    registry = pd.DataFrame(registry_rows)
    write_csv(questionnaire, questionnaire_path)
    return questionnaire, registry


def _apply_rule(
    development: pd.DataFrame,
    external: pd.DataFrame,
    registry_row: pd.Series,
) -> tuple[pd.Series, pd.Series, int, int]:
    development_column = registry_row["development_column_or_expression"]
    external_column = registry_row["external_column_or_expression"]
    transformation = registry_row["transformation"]
    if transformation == "development_rowwise_max_available_hand_both_missing_to_missing; external_max_hndgrp_zero_to_missing_then_numeric":
        right, right_invalid = _clean_numeric(development["max_handgrip_right"])
        left, left_invalid = _clean_numeric(development["max_handgrip_left"])
        development_values = pd.concat([right, left], axis=1).max(axis=1, skipna=True)
        development_values[right.isna() & left.isna()] = np.nan
        external_values, external_invalid = _clean_numeric(external["max_Hndgrp"])
        external_values = external_values.mask(external_values.eq(0), np.nan)
        return (
            development_values,
            external_values,
            right_invalid + left_invalid,
            external_invalid,
        )
    if transformation == "development_numeric_identity; external_weighted_weekly_mean_5_sit_work_plus_2_sit_weekend_divided_by_7":
        development_values, development_invalid = _clean_numeric(
            development["mean_sitting_min_day"]
        )
        sit_work, sit_work_invalid = _clean_numeric(external["sit_work"])
        sit_weekend, sit_weekend_invalid = _clean_numeric(external["sit_weekend"])
        external_values = (5 * sit_work + 2 * sit_weekend) / 7
        return (
            development_values,
            external_values,
            development_invalid,
            sit_work_invalid + sit_weekend_invalid,
        )
    if transformation == "development_living_with_spouses_to_label_independent_combined_family_to_label_extended; external_independent_to_label_independent_child_or_siblings_family_to_label_extended":
        development_values = _clean_text(development["housing_status"]).map(
            {
                "living with spouses": "label_independent",
                "with child & family’s&with brother &sister": "label_extended",
            }
        )
        external_values = _clean_text(external["homep"]).map(
            {
                "independent / living in own/parner/children's house": "label_independent",
                "living with one of children and their family": "label_extended",
                "living with siblings or a family member and their family": "label_extended",
            }
        )
        return development_values, external_values, 0, 0
    if registry_row["variable_type"] == "numeric":
        development_values, development_invalid = _clean_numeric(development[development_column])
        external_values, external_invalid = _clean_numeric(external[external_column])
        return development_values, external_values, development_invalid, external_invalid

    development_values = _clean_text(development[development_column])
    external_values = _clean_text(external[external_column])
    if transformation == "development_trim_casefold; external_m_to_married_s_to_single":
        external_values = external_values.map({"m": "married", "s": "single"})
    elif transformation == "development_regular_or_somtimes_to_yes_no_to_no; external_trim_casefold":
        development_values = development_values.map(
            {"regular": "yes", "somtimes": "yes", "no": "no"}
        )
    elif transformation == "development_low_moderate_high_identity; external_poorest_or_poorer_to_low_middle_to_moderate_richer_or_richest_to_high":
        external_values = external_values.map(
            {
                "poorest": "low",
                "poorer": "low",
                "middle": "moderate",
                "richer": "high",
                "richest": "high",
            }
        )
    elif transformation == "development_illitrate_or_primary_to_illitrate_and_primary_school_secendry_and_high_to_secondary_school_academic_identity; external_trim_casefold":
        development_values = development_values.map(
            {
                "illitrate": "illitrate&primary school",
                "primary": "illitrate&primary school",
                "secendry&high": "secondary school",
                "academic": "academic",
            }
        )
    elif transformation == "development_99_to_missing_then_trim_casefold; external_trim_casefold":
        development_values = development_values.replace("99", pd.NA)
    return development_values, external_values, 0, 0


def validate_registry(
    development: pd.DataFrame,
    external: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    validation_rows = []
    for _, row in registry.iterrows():
        development_values, external_values, development_invalid, external_invalid = _apply_rule(
            development,
            external,
            row,
        )
        record = {
            "canonical_name": row["canonical_name"],
            "role": row["role"],
            "variable_type": row["variable_type"],
            "development_nonmissing_n": int(development_values.notna().sum()),
            "external_nonmissing_n": int(external_values.notna().sum()),
            "development_missing_pct": round(100 * development_values.isna().mean(), 3),
            "external_missing_pct": round(100 * external_values.isna().mean(), 3),
            "development_invalid_nonblank_n": development_invalid,
            "external_invalid_nonblank_n": external_invalid,
            "development_levels": "",
            "external_levels": "",
            "external_levels_not_in_development": "",
            "validation_status": "pass",
        }
        if row["variable_type"] == "categorical":
            development_levels = sorted(development_values.dropna().unique().tolist())
            external_levels = sorted(external_values.dropna().unique().tolist())
            unseen = sorted(set(external_levels) - set(development_levels))
            record["development_levels"] = json.dumps(development_levels, ensure_ascii=False)
            record["external_levels"] = json.dumps(external_levels, ensure_ascii=False)
            record["external_levels_not_in_development"] = json.dumps(unseen, ensure_ascii=False)
            if unseen:
                record["validation_status"] = "fail_unseen_external_categories"
        elif development_invalid or external_invalid:
            record["validation_status"] = "fail_numeric_parse"
        validation_rows.append(record)
    return pd.DataFrame(validation_rows)


def build_harmonized_frames(
    development: pd.DataFrame,
    external: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_columns: dict[str, pd.Series] = {}
    external_columns: dict[str, pd.Series] = {}
    for _, row in registry.iterrows():
        canonical_name = row["canonical_name"]
        if canonical_name in development_columns:
            raise ValueError(f"Duplicate canonical name in registry: {canonical_name}")
        development_values, external_values, development_invalid, external_invalid = _apply_rule(
            development,
            external,
            row,
        )
        if development_invalid or external_invalid:
            raise ValueError(
                f"Nonblank parsing failures for {canonical_name}: "
                f"development={development_invalid}, external={external_invalid}"
            )
        development_columns[canonical_name] = development_values
        external_columns[canonical_name] = external_values
    return pd.DataFrame(development_columns), pd.DataFrame(external_columns)


def run_harmonization_resolution(
    development_path: Path,
    external_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    resolved_questionnaire_path = output_dir / "harmonization_questionnaire_resolved.csv"
    questionnaire_path = (
        resolved_questionnaire_path
        if resolved_questionnaire_path.exists()
        else output_dir / "harmonization_questionnaire.csv"
    )
    questionnaire, registry = resolve_questionnaire(questionnaire_path)
    development, external, _ = load_inputs(development_path, external_path)
    validation = validate_registry(development, external, registry)
    unresolved = questionnaire.loc[questionnaire["status"] != "confirmed"].copy()
    write_csv(registry, output_dir / "harmonization_registry.csv")
    write_csv(validation, output_dir / "harmonization_validation.csv")
    write_csv(unresolved, output_dir / "unresolved_harmonization.csv")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update(
        {
            "confirmed_harmonization_rules": int(len(registry)),
            "unresolved_harmonization_rules": int(len(unresolved)),
            "active_harmonization_questionnaire": questionnaire_path.name,
            "harmonized_participant_level_files_written": False,
            "harmonization_validation_policy": "structural_validation_only; external_not_used_for_model_selection",
        }
    )
    manifest["outputs"] = sorted(
        set(manifest.get("outputs", []))
        | {
            "harmonization_registry.csv",
            "harmonization_validation.csv",
            "unresolved_harmonization.csv",
            questionnaire_path.name,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "questionnaire": questionnaire,
        "registry": registry,
        "validation": validation,
        "unresolved": unresolved,
    }
