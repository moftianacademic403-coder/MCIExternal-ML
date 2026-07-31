from __future__ import annotations

import argparse
import json
from pathlib import Path

from light_calibration_dca import run_light_calibration_dca
from light_evaluation import run_light_evaluation
from light_modeling import run_light_tuning
from light_operating_points_recalibration import (
    run_light_operating_points_recalibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight all-predictor nested-mRMR experiment using the prior "
            "four-level education definition matched to External by Code."
        )
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-education", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-operating-analysis",
        action="store_true",
        help=(
            "Also run bootstrap operating points and cross-validated External "
            "recalibration. Omit for the default lightweight run."
        ),
    )
    args = parser.parse_args()

    qc_output = args.output / "qc"
    light_output = args.output / "light"
    qc_output.mkdir(parents=True, exist_ok=True)
    light_output.mkdir(parents=True, exist_ok=True)

    tuning = run_light_tuning(
        args.development,
        args.external,
        qc_output,
        light_output,
        external_education_path=args.external_education,
    )
    evaluation = run_light_evaluation(
        args.development,
        args.external,
        qc_output,
        light_output,
        external_education_path=args.external_education,
    )
    calibration = run_light_calibration_dca(
        args.development,
        args.external,
        qc_output,
        light_output,
        external_education_path=args.external_education,
    )
    operating = None
    if args.full_operating_analysis:
        operating = run_light_operating_points_recalibration(
            args.development,
            args.external,
            qc_output,
            light_output,
            external_education_path=args.external_education,
        )

    summary = {
        "status": "completed_lightweight_experiment",
        "correlation_pruning_applied": False,
        "mrmr_candidate_predictor_count": tuning["manifest"][
            "mrmr_candidate_predictor_count"
        ],
        "education_harmonization_mode": tuning["manifest"][
            "education_harmonization_mode"
        ],
        "selected_models": tuning["selected_models"].to_dict(orient="records"),
        "evaluation_rows": int(len(evaluation["metrics"])),
        "calibration_rows": int(len(calibration["calibration_metrics"])),
        "full_operating_analysis_run": bool(args.full_operating_analysis),
        "operating_point_rows": (
            int(len(operating["operating_points"])) if operating is not None else 0
        ),
    }
    (args.output / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
