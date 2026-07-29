from __future__ import annotations

from pathlib import Path

import pandas as pd

from mci_qc import write_csv


ROOT = Path(__file__).resolve().parents[1]
QUESTIONNAIRE_PATH = ROOT / "outputs" / "qc" / "harmonization_questionnaire.csv"
RESOLVED_QUESTIONNAIRE_PATH = (
    ROOT / "outputs" / "qc" / "harmonization_questionnaire_resolved.csv"
)

UPDATES = {
    "housing_status": (
        "بله- Independent / living in own/parner/children's house: LABEL_INDEPENDENT; "
        "Living with one of children and their family: LABEL_EXTENDED; "
        "Living with siblings or a family member and their family: LABEL_EXTENDED; "
        "living with spouses: LABEL_INDEPENDENT; "
        "With child & family’s&With brother &sister: LABEL_EXTENDED"
    ),
    "max_handgrip_right + max_handgrip_left": (
        "اگر فقط یک دست مقدار دارد از مقدار دست موجود استفاده شود؛ "
        "اگر هر دو دست missing هستند max_handgrip نیز missing باشد؛ "
        "در External مقدار صفر max_Hndgrp به‌عنوان missing در نظر گرفته شود"
    ),
    "mean_sitting_min_day": (
        "external: (5 * sit_work + 2 * sit_weekend) / 7; "
        "development: mean_sitting_min_day بدون تغییر"
    ),
    "household_income": (
        "Poorest + Poorer → low; Middle → moderate; Richer + Richest → high"
    ),
    "EDU": (
        "Illitrate + primary → illitrate&primary school; "
        "secendry&high → secondary school; Academic → Academic"
    ),
}


def main() -> None:
    source_path = (
        RESOLVED_QUESTIONNAIRE_PATH
        if RESOLVED_QUESTIONNAIRE_PATH.exists()
        else QUESTIONNAIRE_PATH
    )
    questionnaire = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        dtype="string",
    ).fillna("")
    for development_expression, response in UPDATES.items():
        mask = questionnaire["development_column_or_expression"].eq(development_expression)
        if int(mask.sum()) != 1:
            raise ValueError(
                f"Expected exactly one questionnaire row for {development_expression!r}; "
                f"found {int(mask.sum())}."
            )
        questionnaire.loc[mask, "user_response"] = response
    write_csv(questionnaire, RESOLVED_QUESTIONNAIRE_PATH)
    print(
        f"Recorded {len(UPDATES)} user response updates in "
        f"{RESOLVED_QUESTIONNAIRE_PATH}"
    )


if __name__ == "__main__":
    main()
