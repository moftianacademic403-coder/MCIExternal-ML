from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "mci-heavy-nested-cv"
NOTEBOOK_PATH = KERNEL_DIR / "mci_heavy_nested_cv_kaggle.ipynb"
TEMPLATE_PATH = KERNEL_DIR / "kernel-metadata.template.json"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"


def prepare(
    kaggle_username: str,
    private_dataset_slug: str,
    repository_url: str,
    repository_branch: str,
) -> None:
    metadata = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    metadata["id"] = f"{kaggle_username}/mci-heavy-selection-and-validation"
    metadata["dataset_sources"] = [
        f"{kaggle_username}/{private_dataset_slug}"
    ]
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    replacements = {
        'REPOSITORY_URL = "https://github.com/REPLACE_USERNAME/REPLACE_REPOSITORY.git"': (
            f"REPOSITORY_URL = {repository_url!r}"
        ),
        'REPOSITORY_BRANCH = "main"': (
            f"REPOSITORY_BRANCH = {repository_branch!r}"
        ),
    }
    replacement_counts = {key: 0 for key in replacements}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in replacements.items():
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                replacement_counts[old] += 1
    if any(count != 1 for count in replacement_counts.values()):
        raise RuntimeError(
            f"Unexpected notebook parameter replacement counts: {replacement_counts}"
        )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill the Kaggle kernel metadata and GitHub repository URL."
    )
    parser.add_argument("--kaggle-username", required=True)
    parser.add_argument("--private-dataset-slug", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--repository-branch", default="main")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare(
        args.kaggle_username,
        args.private_dataset_slug,
        args.repository_url,
        args.repository_branch,
    )
    print(NOTEBOOK_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()

