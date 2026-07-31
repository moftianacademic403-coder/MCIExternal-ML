from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mci_qc import sha256_file


ALLOWED_FILES = (
    "final_evaluation_manifest.json",
    "final_model_configs.csv",
    "development_thresholds.csv",
    "external_dca.csv",
)


def prepare_aggregate_dataset(
    heavy_output_dir: Path,
    staging_dir: Path,
    owner: str,
    slug: str,
) -> dict:
    final_dir = heavy_output_dir / "final_evaluation"
    manifest_path = final_dir / "final_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "manuscript_grade_locked_evaluation_completed":
        raise RuntimeError(f"Unexpected heavy status: {manifest.get('status')}")
    if manifest.get("participant_level_predictions_written") is not False:
        raise RuntimeError("Participant-level output guard failed.")
    if (
        manifest.get("education_harmonization_mode")
        != "four_level_code_matched_auxiliary_source"
    ):
        raise RuntimeError("Heavy results did not use four-level education.")

    staging_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in staging_dir.iterdir() if path.is_file()]
    if unexpected:
        raise RuntimeError(
            "Staging directory must be empty before an aggregate-only rebuild: "
            + ", ".join(unexpected)
        )
    copied = []
    for filename in ALLOWED_FILES:
        source = final_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        destination = staging_dir / filename
        shutil.copy2(source, destination)
        copied.append(
            {
                "filename": filename,
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    dataset_metadata = {
        "title": "MCI Heavy Aggregate Results",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    transfer_manifest = {
        "status": "aggregate_only_ready_for_private_kaggle_dataset",
        "source_heavy_status": manifest["status"],
        "education_harmonization_mode": manifest[
            "education_harmonization_mode"
        ],
        "participant_level_outputs_included": False,
        "files": copied,
    }
    (staging_dir / "aggregate_transfer_manifest.json").write_text(
        json.dumps(transfer_manifest, indent=2), encoding="utf-8"
    )
    return transfer_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage only the aggregate heavy outputs required by the post-hoc kernel."
    )
    parser.add_argument("--heavy-output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="mci-heavy-aggregate-results")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = prepare_aggregate_dataset(
        args.heavy_output,
        args.staging,
        args.owner,
        args.slug,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
