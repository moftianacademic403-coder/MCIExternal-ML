from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "kaggle" / "mci-heavy-nested-cv" / "mci_heavy_nested_cv_kaggle.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}
notebook["cells"] = [
    markdown(
        """
# MCI heavy selection and locked evaluation on Kaggle GPU

## Goal

Run Development-only repeated nested cross-validation for five prespecified model families, including TabPFN-3. The selected family is then frozen, calibrated and assigned operating thresholds using Development training only, followed by one-time locked internal and External evaluation. The separately labelled External local-update analysis is performed only after locked validation.
"""
    ),
    markdown(
        """
## Setup

Before pressing **Save Version → Save & Run All**:

1. Enable a GPU and Internet in notebook settings.
2. Attach one **private Kaggle dataset** containing exactly `Developement.csv` and `External.xlsx`.
3. Create a Kaggle Secret named `TABPFN_TOKEN` if the TabPFN checkpoint/license flow requires authentication.
4. Replace `REPOSITORY_URL` below with the public code-only GitHub repository. Never place participant data or access tokens in GitHub.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys

REPOSITORY_URL = "https://github.com/REPLACE_USERNAME/REPLACE_REPOSITORY.git"
REPOSITORY_BRANCH = "main"
PROJECT_DIR = Path("/kaggle/working/MCIExternal")
OUTPUT_DIR = Path("/kaggle/working/mci_heavy_nested_outputs")
INPUT_ROOT = Path("/kaggle/input")
TABPFN_VERSION = "8.2.0"
"""
    ),
    markdown("### 1. Install the pinned TabPFN release"),
    code(
        r"""
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        f"tabpfn=={TABPFN_VERSION}",
    ],
    check=True,
)
"""
    ),
    markdown("### 2. Load TabPFN authentication from Kaggle Secrets"),
    code(
        r"""
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
token = None
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("TABPFN_TOKEN")
    if token:
        os.environ["TABPFN_TOKEN"] = token
        print("TABPFN_TOKEN loaded from Kaggle Secrets.")
except Exception as error:
    raise RuntimeError(
        "TABPFN_TOKEN is unavailable. Add and enable this Kaggle Secret before "
        "the unattended GPU run. The token must never be pasted into this notebook."
    ) from error

if not token:
    raise RuntimeError(
        "TABPFN_TOKEN is empty. Add and enable this Kaggle Secret before the run."
    )
"""
    ),
    markdown("### 3. Verify GPU and clone the code-only repository"),
    code(
        r"""
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Enable a Kaggle GPU before running.")
print("GPU:", torch.cuda.get_device_name(0))

if "REPLACE_" in REPOSITORY_URL:
    raise RuntimeError("Set REPOSITORY_URL to the new code-only GitHub repository.")
if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        REPOSITORY_BRANCH,
        REPOSITORY_URL,
        str(PROJECT_DIR),
    ],
    check=True,
)
"""
    ),
    markdown("### 4. Find and validate the private input files"),
    code(
        r"""
def find_unique(filename: str) -> Path:
    matches = [
        path
        for path in INPUT_ROOT.rglob("*")
        if path.is_file() and path.name.casefold() == filename.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename!r} under /kaggle/input; found {matches}."
        )
    return matches[0]


development_path = find_unique("Developement.csv")
external_path = find_unique("External.xlsx")
print("Development input found:", development_path.parent.name, development_path.name)
print("External input found:", external_path.parent.name, external_path.name)
print("Participant-level contents are not displayed.")
"""
    ),
    markdown("## Steps\n\n### 5. Run Development-only repeated nested CV"),
    code(
        r"""
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
qc_output = OUTPUT_DIR / "qc"
command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "heavy_nested_cv.py"),
    "--development",
    str(development_path),
    "--external",
    str(external_path),
    "--qc-output",
    str(qc_output),
    "--output",
    str(OUTPUT_DIR / "selection"),
]
print("Starting the full GPU run. This cell can take several hours.")
subprocess.run(command, cwd=PROJECT_DIR, check=True)
"""
    ),
    markdown("### 6. Freeze Development choices and run locked evaluation"),
    code(
        r"""
final_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "heavy_final_evaluation.py"),
    "--development",
    str(development_path),
    "--external",
    str(external_path),
    "--qc-output",
    str(qc_output),
    "--selection-output",
    str(OUTPUT_DIR / "selection"),
    "--output",
    str(OUTPUT_DIR / "final_evaluation"),
]
print("Starting Development finalization and locked evaluation.")
subprocess.run(final_command, cwd=PROJECT_DIR, check=True)
"""
    ),
    markdown("### 7. Record the execution environment"),
    code(
        r"""
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "repository_url": REPOSITORY_URL,
    "repository_branch": REPOSITORY_BRANCH,
    "repository_commit": subprocess.check_output(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], text=True
    ).strip(),
}
(OUTPUT_DIR / "execution_environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
with (OUTPUT_DIR / "pip_freeze.txt").open("w", encoding="utf-8") as handle:
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=handle,
        text=True,
        check=True,
    )
print(environment)
"""
    ),
    markdown("## Checks\n\n### 8. Verify aggregate outputs and create a ZIP"),
    code(
        r"""
selection_path = OUTPUT_DIR / "selection" / "model_selection.json"
manifest_path = OUTPUT_DIR / "selection" / "nested_cv_manifest.json"
final_manifest_path = (
    OUTPUT_DIR / "final_evaluation" / "final_evaluation_manifest.json"
)
if not selection_path.exists() or not manifest_path.exists() or not final_manifest_path.exists():
    raise RuntimeError("Required selection outputs were not produced.")

selection = json.loads(selection_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
if selection["external_used_for_selection"]:
    raise RuntimeError("Leakage guard failed: External was marked as used for selection.")
if selection["internal_test_20_used_for_selection"]:
    raise RuntimeError("Leakage guard failed: internal test was used for selection.")
if not selection["tabpfn_included"]:
    raise RuntimeError("TabPFN was not included in the full run.")
if manifest["status"] != "development_only_nested_cv_model_family_selected":
    raise RuntimeError(f"Unexpected run status: {manifest['status']}")
if final_manifest["status"] != "manuscript_grade_locked_evaluation_completed":
    raise RuntimeError(f"Unexpected final status: {final_manifest['status']}")
if final_manifest["selected_model_name"] != selection["selected_model_name"]:
    raise RuntimeError("Final evaluation did not preserve the nested-CV selection.")
if not final_manifest["tabpfn_included"]:
    raise RuntimeError("TabPFN was missing from final family comparison.")

print(json.dumps(selection, indent=2))
print(json.dumps(final_manifest, indent=2))
archive = shutil.make_archive(
    "/kaggle/working/mci_heavy_nested_outputs",
    "zip",
    root_dir=OUTPUT_DIR,
)
print("Aggregate output archive:", archive)
"""
    ),
    markdown(
        """
## Next Steps

- Download and inspect the aggregate ZIP and both leakage manifests.
- Do not switch to another model because it looks better on External.
- Subgroup interaction analyses, MICE/complete-case sensitivity analyses, and selected-model SHAP or TabPFN-native interpretability remain separate post-selection work and are explicitly listed in the final manifest.
"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
