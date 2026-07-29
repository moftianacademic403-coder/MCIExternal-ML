from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "MCI"
SENTINEL_TOKENS = {
    "99",
    "999",
    "-99",
    "-999",
    "unknown",
    "don't know",
    "dont know",
    "n/a",
    "na",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_csv_format(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:200_000]
    decoded = None
    encoding = None
    for candidate in ("utf-8-sig", "utf-8", "cp1256", "windows-1252"):
        try:
            decoded = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if decoded is None or encoding is None:
        raise UnicodeError(f"Unable to detect a supported encoding for {path}")
    try:
        delimiter = csv.Sniffer().sniff(decoded[:50_000], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def load_inputs(development_path: Path, external_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    encoding, delimiter = detect_csv_format(development_path)
    development = pd.read_csv(
        development_path,
        encoding=encoding,
        sep=delimiter,
        low_memory=False,
    )
    workbook = pd.ExcelFile(external_path)
    if len(workbook.sheet_names) != 1:
        raise ValueError(
            "External workbook has more than one worksheet. Sheet selection requires user confirmation: "
            + ", ".join(workbook.sheet_names)
        )
    external = pd.read_excel(external_path, sheet_name=workbook.sheet_names[0])
    metadata = {
        "development_path": str(development_path),
        "external_path": str(external_path),
        "development_sha256": sha256_file(development_path),
        "external_sha256": sha256_file(external_path),
        "development_encoding": encoding,
        "development_delimiter": delimiter,
        "external_sheet": workbook.sheet_names[0],
    }
    return development, external, metadata


def _string_view(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _blank_mask(series: pd.Series) -> pd.Series:
    string_values = _string_view(series)
    return string_values.notna() & string_values.str.strip().eq("")


def _normalized_series(series: pd.Series) -> pd.Series:
    string_values = _string_view(series).str.strip().replace("", pd.NA)
    numeric_values = pd.to_numeric(string_values, errors="coerce")
    nonmissing_count = int(string_values.notna().sum())
    if nonmissing_count and numeric_values.notna().sum() / nonmissing_count >= 0.95:
        return numeric_values
    return string_values.str.casefold()


def dataset_summary(
    development: pd.DataFrame,
    external: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    records = []
    for name, frame, path_key, hash_key in (
        ("development", development, "development_path", "development_sha256"),
        ("external", external, "external_path", "external_sha256"),
    ):
        target_missing = int(frame[TARGET].isna().sum()) if TARGET in frame else np.nan
        target_yes = (
            int(frame[TARGET].astype("string").str.strip().str.casefold().eq("yes").sum())
            if TARGET in frame
            else np.nan
        )
        target_no = (
            int(frame[TARGET].astype("string").str.strip().str.casefold().eq("no").sum())
            if TARGET in frame
            else np.nan
        )
        records.append(
            {
                "dataset": name,
                "source_path": metadata[path_key],
                "sha256": metadata[hash_key],
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "target_column_present": TARGET in frame,
                "target_missing_n": target_missing,
                "target_yes_n": target_yes,
                "target_no_n": target_no,
                "target_yes_pct_among_all_rows": round(100 * target_yes / len(frame), 3),
            }
        )
    return pd.DataFrame(records)


def column_profile(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for dataset, frame in frames.items():
        row_count = len(frame)
        for column in frame.columns:
            series = frame[column]
            blanks = int(_blank_mask(series).sum())
            raw_missing = int(series.isna().sum())
            normalized = _normalized_series(series)
            numeric_values = pd.to_numeric(
                _string_view(series).str.strip().replace("", pd.NA),
                errors="coerce",
            )
            nonblank_nonmissing = int((~series.isna() & ~_blank_mask(series)).sum())
            parseable_numeric = int(numeric_values.notna().sum())
            numeric_parse_pct = (
                100 * parseable_numeric / nonblank_nonmissing if nonblank_nonmissing else np.nan
            )
            records.append(
                {
                    "dataset": dataset,
                    "column": str(column),
                    "raw_dtype": str(series.dtype),
                    "raw_missing_n": raw_missing,
                    "blank_or_whitespace_n": blanks,
                    "effective_missing_n": raw_missing + blanks,
                    "effective_missing_pct": round(100 * (raw_missing + blanks) / row_count, 3),
                    "unique_nonmissing_normalized_n": int(normalized.nunique(dropna=True)),
                    "numeric_parse_pct_nonblank": round(numeric_parse_pct, 3)
                    if pd.notna(numeric_parse_pct)
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def categorical_levels(frames: dict[str, pd.DataFrame], max_levels: int = 15) -> pd.DataFrame:
    records = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            values = _string_view(frame[column]).str.strip().replace("", pd.NA)
            unique_count = int(values.nunique(dropna=True))
            if unique_count <= max_levels:
                counts = values.fillna("<MISSING>").value_counts(dropna=False)
                for level, count in counts.items():
                    records.append(
                        {
                            "dataset": dataset,
                            "column": str(column),
                            "level": str(level),
                            "count": int(count),
                            "pct": round(100 * count / len(frame), 3),
                        }
                    )
    return pd.DataFrame(records)


def numeric_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            stripped = _string_view(frame[column]).str.strip().replace("", pd.NA)
            numeric = pd.to_numeric(stripped, errors="coerce")
            nonmissing = int(stripped.notna().sum())
            numeric_count = int(numeric.notna().sum())
            if not nonmissing or numeric_count / nonmissing < 0.95:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "column": str(column),
                    "numeric_n": numeric_count,
                    "missing_after_blank_normalization_n": int(len(frame) - numeric_count),
                    "min": numeric.min(),
                    "p01": numeric.quantile(0.01),
                    "q1": numeric.quantile(0.25),
                    "median": numeric.median(),
                    "q3": numeric.quantile(0.75),
                    "p99": numeric.quantile(0.99),
                    "max": numeric.max(),
                    "zero_n": int(numeric.eq(0).sum()),
                    "negative_n": int(numeric.lt(0).sum()),
                }
            )
    return pd.DataFrame(records)


def sentinel_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            stripped = _string_view(frame[column]).str.strip().replace("", pd.NA)
            numeric = pd.to_numeric(stripped, errors="coerce")
            nonmissing_count = int(stripped.notna().sum())
            numeric_parse_ratio = (
                numeric.notna().sum() / nonmissing_count if nonmissing_count else 0.0
            )
            # Values such as 99 can be valid measurements or identifiers. Only flag
            # sentinel-like tokens in fields that are not fully numeric high-cardinality
            # variables; interpretation still remains pending user confirmation.
            if numeric_parse_ratio == 1.0 and stripped.nunique(dropna=True) > 10:
                continue
            values = stripped.str.casefold()
            counts = values[values.isin(SENTINEL_TOKENS)].value_counts()
            for token, count in counts.items():
                records.append(
                    {
                        "dataset": dataset,
                        "column": str(column),
                        "observed_token": str(token),
                        "count": int(count),
                        "interpretation": "pending_user_confirmation",
                    }
                )
    return pd.DataFrame(records)


def duplicate_columns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for dataset, frame in frames.items():
        normalized = {str(column): _normalized_series(frame[column]) for column in frame.columns}
        columns = list(normalized)
        for index, left_name in enumerate(columns):
            for right_name in columns[index + 1 :]:
                left = normalized[left_name]
                right = normalized[right_name]
                overlap = left.notna() & right.notna()
                if int(overlap.sum()) < max(1, int(0.5 * len(frame))):
                    continue
                values_equal = left[overlap].reset_index(drop=True).equals(
                    right[overlap].reset_index(drop=True)
                )
                if values_equal:
                    records.append(
                        {
                            "dataset": dataset,
                            "column_a": left_name,
                            "column_b": right_name,
                            "overlapping_nonmissing_n": int(overlap.sum()),
                            "same_missing_pattern": bool(left.isna().equals(right.isna())),
                            "decision": "pending_user_confirmation",
                        }
                    )
    return pd.DataFrame(records)


def schema_alignment(development: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    development_lookup = {str(column).strip().casefold(): str(column) for column in development.columns}
    external_lookup = {str(column).strip().casefold(): str(column) for column in external.columns}
    keys = sorted(set(development_lookup) | set(external_lookup))
    records = []
    for key in keys:
        in_development = key in development_lookup
        in_external = key in external_lookup
        records.append(
            {
                "normalized_name": key,
                "development_column": development_lookup.get(key, ""),
                "external_column": external_lookup.get(key, ""),
                "exact_name_match_after_trim_casefold": in_development and in_external,
                "external_model_eligibility": "pending_definition_and_unit_confirmation"
                if in_development and in_external
                else "not_matched",
            }
        )
    return pd.DataFrame(records)


def harmonization_questionnaire() -> pd.DataFrame:
    candidate_rows = [
        ("age", "age", "same_name", "آیا تعریف سن و واحد آن در هر دو فایل یکسان است؟"),
        ("sex", "sex", "same_name", "آیا کدگذاری sex در هر دو فایل از یک تعریف استفاده می‌کند؟"),
        ("employment_status", "Job", "name_similarity_only", "آیا Job همان employment_status است و Retired/retired دقیقاً یک سطح‌اند؟"),
        ("housing_status", "homep", "name_similarity_only", "آیا homep از نظر تعریف با housing_status قابل همسان‌سازی است؟ در صورت تأیید، نگاشت سطوح را مشخص کنید."),
        ("marital_status", "marital_new", "name_similarity_only", "آیا m و s به‌ترتیب married و single هستند و سطح دیگری ادغام نشده است؟"),
        ("household_income", "wealth_index", "possible_related_concept_only", "آیا wealth_index قابل معادل‌سازی با household_income است؟ اگر بله، نگاشت پنج سطح به سه سطح را مشخص کنید."),
        ("smoking_status", "Current_Smoker", "possible_related_concept_only", "آیا regular و somtimes در Development هر دو باید Current_Smoker=Yes شوند؟"),
        ("mean_sitting_min_day", "sit_work + sit_weekend", "possible_derivation_only", "فرمول دقیق ساخت mean_sitting_min_day از sit_work و sit_weekend چیست؟"),
        ("sleep_quality", "Global_PSQI_Binary", "name_similarity_only", "آیا هر دو بر اساس یک ابزار و cut-off یکسان good/poor ساخته شده‌اند؟"),
        ("adl", "Disable_ADL", "name_similarity_only", "آیا تعریف dependent/independent و ابزار ADL در هر دو فایل یکسان است؟"),
        ("iadl", "Disable_IADL", "name_similarity_only", "آیا تعریف dependent/independent و ابزار IADL در هر دو فایل یکسان است؟"),
        ("BMI", "BMI", "same_name", "آیا واحد و فرمول BMI در هر دو فایل یکسان است؟"),
        ("waist_circumference_cm", "Waistcm", "name_and_unit_similarity_only", "آیا پروتکل اندازه‌گیری و واحد هر دو دور کمر سانتی‌متر است؟"),
        ("waist_hip_ratio", "WHR", "name_similarity_only", "آیا WHR دقیقاً با همان فرمول waist_hip_ratio محاسبه شده است؟"),
        ("hip_circumference_cm", "Hipcm", "name_and_unit_similarity_only", "آیا پروتکل اندازه‌گیری و واحد هر دو دور باسن سانتی‌متر است؟"),
        ("osteoporosis_status", "osteoporosis_final2", "name_similarity_only", "آیا معیار تشخیص osteoporosis در هر دو فایل یکسان است؟"),
        ("PLT", "plt", "case_insensitive_name_match", "آیا واحد PLT در هر دو فایل یکسان است؟"),
        ("HCT", "hct", "case_insensitive_name_match", "آیا HCT در هر دو فایل درصد و با روش یکسان است؟"),
        ("Triglycerides", "tg", "common_abbreviation_only", "آیا tg همان Triglycerides و واحد هر دو یکسان است؟"),
        ("rbc", "rbc", "same_name", "آیا واحد RBC در هر دو فایل یکسان است؟"),
        ("wbc", "wbc", "same_name", "آیا واحد WBC در هر دو فایل یکسان است؟"),
        ("hemoglobin", "hgb", "common_abbreviation_only", "آیا hgb همان hemoglobin و واحد هر دو یکسان است؟"),
        ("total_cholesterol", "chol", "common_abbreviation_only", "آیا chol همان total_cholesterol و واحد هر دو یکسان است؟"),
        ("hdl_cholesterol", "hdl", "common_abbreviation_only", "آیا hdl همان hdl_cholesterol و واحد هر دو یکسان است؟"),
        ("ldl_cholesterol", "ldl", "common_abbreviation_only", "آیا ldl همان ldl_cholesterol و واحد/روش محاسبه هر دو یکسان است؟"),
        ("uric_acid", "UricAcid", "name_similarity_only", "آیا واحد uric acid در هر دو فایل یکسان است؟"),
        ("ast", "ast", "same_name", "آیا واحد و روش AST در هر دو فایل یکسان است؟"),
        ("alt", "alt", "same_name", "آیا واحد و روش ALT در هر دو فایل یکسان است؟"),
        ("vitamin_d", "VitD", "name_similarity_only", "آیا نوع آزمایش و واحد Vitamin D در هر دو فایل یکسان است؟"),
        ("whisper_test_left", "WSPR_LEFT", "name_similarity_only", "آیا مقیاس 0 تا 6 و پروتکل آزمون گوش چپ در هر دو فایل یکسان است؟"),
        ("whisper_test_right", "WSPR_RIGHT", "name_similarity_only", "آیا مقیاس 0 تا 6 و پروتکل آزمون گوش راست در هر دو فایل یکسان است؟"),
        ("visual_acuity_both", "VISACU_BOTH", "name_similarity_only", "آیا مقیاس visual acuity در هر دو فایل یکسان است؟ Development تا 12 و External تا 10 مشاهده شده است."),
        ("max_handgrip_right + max_handgrip_left", "max_Hndgrp", "possible_derivation_only", "آیا max_Hndgrp بیشینه دو دست است؟ اگر نه، تعریف دقیق آن چیست؟"),
        ("depression_status", "depression", "name_similarity_only", "آیا ابزار و cut-off افسردگی در هر دو فایل یکسان است؟"),
        ("diabetes", "dm", "common_abbreviation_only", "آیا معیار diabetes/dm در هر دو فایل یکسان است؟"),
        ("hypertension", "htn", "common_abbreviation_only", "آیا معیار hypertension/htn در هر دو فایل یکسان است؟"),
        ("history_heart_failure", "CHF01", "clinical_abbreviation_only", "آیا CHF01 همان history_heart_failure است و کد 99 در Development به‌معنای missing/unknown است؟"),
        ("history_stroke", "Stroke01", "name_similarity_only", "آیا Stroke01 همان history_stroke است و کد 99 در Development به‌معنای missing/unknown است؟"),
        ("history_myocardial_infarction", "MI01", "clinical_abbreviation_only", "آیا MI01 همان history_myocardial_infarction است و کد 99 در Development به‌معنای missing/unknown است؟"),
        ("EDU", "Education", "name_similarity_only", "نگاشت دقیق سطوح تحصیلات بین چهار سطح Development و سه سطح External چیست؟"),
        ("TOTAL_PFAT", "TOTAL_PFAT", "same_name", "آیا تعریف/دستگاه/واحد TOTAL_PFAT در هر دو فایل یکسان است؟"),
        ("ANDROID_PFAT", "ANDROID_PFAT", "same_name", "آیا تعریف/دستگاه/واحد ANDROID_PFAT در هر دو فایل یکسان است؟"),
        ("GYNOID_PFAT", "GYNOID_PFAT", "same_name", "آیا تعریف/دستگاه/واحد GYNOID_PFAT در هر دو فایل یکسان است؟"),
        ("MCI", "MCI", "same_name", "آیا تعریف، ابزار و cut-off برچسب MCI در Development و External دقیقاً یکسان است؟"),
    ]
    return pd.DataFrame(
        [
            {
                "development_column_or_expression": development_column,
                "external_column_or_expression": external_column,
                "candidate_basis": basis,
                "status": "pending_user_confirmation",
                "allowed_for_external_model": False,
                "user_question": question,
                "user_response": "",
                "final_harmonization_rule": "",
            }
            for development_column, external_column, basis, question in candidate_rows
        ]
    )


def qc_findings(
    development: pd.DataFrame,
    external: pd.DataFrame,
    profile: pd.DataFrame,
    duplicates: pd.DataFrame,
    sentinels: pd.DataFrame,
) -> pd.DataFrame:
    findings: list[dict] = []

    def add(issue_id: str, severity: str, dataset: str, evidence: str, risk: str, action: str):
        findings.append(
            {
                "issue_id": issue_id,
                "severity": severity,
                "dataset": dataset,
                "evidence": evidence,
                "modeling_risk": risk,
                "required_action": action,
                "status": "open",
            }
        )

    development_target_missing = int(development[TARGET].isna().sum())
    if development_target_missing:
        add(
            "DQ-001",
            "high",
            "development",
            f"{development_target_missing} row(s) have missing MCI.",
            "Rows without an outcome cannot enter supervised model development.",
            "Confirm whether the row is an import artifact or a participant with unavailable outcome before exclusion.",
        )

    blank_rows = profile.loc[profile["blank_or_whitespace_n"] > 0]
    if not blank_rows.empty:
        top = blank_rows.sort_values("blank_or_whitespace_n", ascending=False).head(5)
        evidence = "; ".join(
            f"{row.dataset}.{row.column}={int(row.blank_or_whitespace_n)}"
            for row in top.itertuples()
        )
        add(
            "DQ-002",
            "high",
            "both",
            f"Blank/whitespace values are not consistently encoded as missing. Largest counts: {evidence}.",
            "Raw null counts understate missingness and numeric columns can be imported as text.",
            "Normalize whitespace to missing in a derived analysis copy after variable-specific confirmation; do not alter source files.",
        )

    high_external_missing = profile.loc[
        (profile["dataset"] == "external") & (profile["effective_missing_pct"] >= 20)
    ].sort_values("effective_missing_pct", ascending=False)
    if not high_external_missing.empty:
        evidence = "; ".join(
            f"{row.column}={row.effective_missing_pct:.2f}%"
            for row in high_external_missing.itertuples()
        )
        add(
            "DQ-003",
            "high",
            "external",
            f"Features with at least 20% effective missingness: {evidence}.",
            "A model depending on these variables may be difficult to transport and external performance may depend on imputation behavior.",
            "Do not use External to choose features. Decide the transportable predictor panel from definitions and Development only, then prespecify missing-data sensitivity analyses.",
        )

    if not duplicates.empty:
        evidence = "; ".join(
            f"{row.dataset}.{row.column_a}={row.column_b}"
            for row in duplicates.itertuples()
        )
        add(
            "DQ-004",
            "medium",
            "development",
            f"Exact duplicate-value column pairs detected: {evidence}.",
            "Redundant columns can make mRMR rankings arbitrary and inflate apparent feature instability.",
            "Confirm provenance and select one canonical variable from each pair before feature selection.",
        )

    if not sentinels.empty:
        evidence = "; ".join(
            f"{row.dataset}.{row.column}:{row.observed_token} (n={row.count})"
            for row in sentinels.itertuples()
        )
        add(
            "DQ-005",
            "high",
            "both",
            f"Potential sentinel-like tokens observed: {evidence}.",
            "Treating all tokens globally as missing can erase valid measurements; treating true sentinels as values can bias models.",
            "Obtain variable-specific confirmation for every token; no automatic replacement is authorized.",
        )

    development_id_like = [
        str(column)
        for column in development.columns
        if development[column].nunique(dropna=True) >= 0.98 * len(development)
    ]
    external_id_like = [
        str(column)
        for column in external.columns
        if external[column].nunique(dropna=True) >= 0.98 * len(external)
    ]
    add(
        "DQ-006",
        "high",
        "both",
        f"ID-like columns: Development={development_id_like or 'none detected'}; External={external_id_like or 'none detected'}.",
        "Identifiers must not enter predictors; absence of a Development identifier limits duplicate-participant verification.",
        "Confirm identifier fields and explicitly exclude them from the predictor registry.",
    )

    common_names = {
        str(column).strip().casefold() for column in development.columns
    } & {str(column).strip().casefold() for column in external.columns}
    add(
        "DQ-007",
        "critical",
        "both",
        f"Only {len(common_names)} column names match after trim/case normalization; semantic candidates remain unconfirmed.",
        "A selected Development feature may be unavailable or differently defined in External.",
        "Complete the harmonization questionnaire before creating any external-validation feature matrix.",
    )

    return pd.DataFrame(findings)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def merge_questionnaire_responses(
    generated: pd.DataFrame,
    existing_path: Path,
) -> pd.DataFrame:
    if not existing_path.exists():
        return generated
    existing = pd.read_csv(existing_path, encoding="utf-8-sig", dtype="string").fillna("")
    keys = ["development_column_or_expression", "external_column_or_expression"]
    if not set(keys).issubset(existing.columns):
        raise ValueError("Existing harmonization questionnaire is missing its key columns.")
    preserved_columns = [
        "status",
        "allowed_for_external_model",
        "user_response",
        "final_harmonization_rule",
    ]
    preserved = existing[keys + [c for c in preserved_columns if c in existing.columns]].copy()
    merged = generated.drop(columns=preserved_columns).merge(
        preserved,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for column in preserved_columns:
        if column not in merged:
            merged[column] = generated[column]
        else:
            fallback = generated.set_index(keys)[column]
            missing = merged[column].isna() | merged[column].astype("string").str.strip().eq("")
            if column in {"status", "allowed_for_external_model"}:
                fallback_values = [fallback.loc[tuple(row)] for row in merged.loc[missing, keys].to_numpy()]
                merged.loc[missing, column] = fallback_values
            else:
                merged.loc[missing, column] = ""
    ordered = list(generated.columns)
    return merged[ordered]


def run_qc(development_path: Path, external_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    development, external, metadata = load_inputs(development_path, external_path)
    frames = {"development": development, "external": external}
    questionnaire_path = output_dir / "harmonization_questionnaire.csv"
    resolved_questionnaire_path = output_dir / "harmonization_questionnaire_resolved.csv"
    questionnaire_source_path = (
        resolved_questionnaire_path
        if resolved_questionnaire_path.exists()
        else questionnaire_path
    )
    results = {
        "dataset_summary": dataset_summary(development, external, metadata),
        "column_profile": column_profile(frames),
        "categorical_levels": categorical_levels(frames),
        "numeric_summary": numeric_summary(frames),
        "sentinel_audit": sentinel_audit(frames),
        "duplicate_columns": duplicate_columns(frames),
        "schema_alignment": schema_alignment(development, external),
        "harmonization_questionnaire": merge_questionnaire_responses(
            harmonization_questionnaire(),
            questionnaire_source_path,
        ),
    }
    results["qc_findings"] = qc_findings(
        development,
        external,
        results["column_profile"],
        results["duplicate_columns"],
        results["sentinel_audit"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        if name == "harmonization_questionnaire":
            if resolved_questionnaire_path.exists():
                write_csv(frame, resolved_questionnaire_path)
            elif not questionnaire_path.exists():
                write_csv(frame, questionnaire_path)
            # Preserve an existing user-edited questionnaire verbatim. The
            # resolution stage writes a separate resolved copy when necessary.
            continue
        write_csv(frame, output_dir / f"{name}.csv")
    manifest = {
        **metadata,
        "target": TARGET,
        "external_use_policy": "external_validation_only",
        "harmonization_policy": "no_mapping_or_transformation_without_user_confirmation",
        "raw_files_modified": False,
        "outputs": sorted(f"{name}.csv" for name in results),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
