from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT
    / "kaggle"
    / "mci-posthoc-sensitivity"
    / "mci_posthoc_sensitivity_kaggle.ipynb"
)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


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
# MCI post-hoc transportability, sensitivity, subgroup and interpretability

This notebook preserves whichever primary model was selected using Development
train-80 repeated nested CV. It uses
the completed heavy-run output as a frozen input and performs only explicitly
labelled secondary analyses. External outcomes never re-select the primary
model. MICE is intentionally excluded at the investigator's request.
"""
    ),
    markdown(
        """
## Setup

- Attach the private `MCIExternal` dataset.
- Attach the private aggregate-only `mci-heavy-aggregate-results` dataset.
- Enable Internet. Enable GPU and the `TABPFN_TOKEN` Kaggle Secret when the
  Development-selected model is TabPFN.
"""
    ),
    code(
        r"""
from pathlib import Path
import json
import os
import platform
import pandas as pd
import shutil
import subprocess
import sys

REPOSITORY_URL = "https://github.com/REPLACE_USERNAME/REPLACE_REPOSITORY.git"
REPOSITORY_BRANCH = "main"
PROJECT_DIR = Path("/kaggle/working/MCIExternal")
OUTPUT_DIR = Path("/kaggle/working/mci_posthoc_outputs")
INPUT_ROOT = Path("/kaggle/input")
"""
    ),
    markdown("### 1. Install pinned analysis dependencies"),
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
        "tabpfn==8.2.0",
        "tabpfn-extensions==0.4.1",
        "shap",
        "statsmodels",
    ],
    check=True,
)
"""
    ),
    markdown("### 2. Load TabPFN authentication if it is available"),
    code(
        r"""
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from kaggle_secrets import UserSecretsClient

token = None
try:
    token = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception as error:
    print("TABPFN_TOKEN is not enabled; this is acceptable only for a classical selected model.")
if token:
    os.environ["TABPFN_TOKEN"] = token
    print("TABPFN_TOKEN loaded from Kaggle Secrets.")
"""
    ),
    markdown("### 3. Verify GPU and clone the frozen code repository"),
    code(
        r"""
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if "REPLACE_" in REPOSITORY_URL:
    raise RuntimeError("Configure the code-only GitHub repository URL.")
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
    markdown("### 4. Locate private data and the completed heavy-run output"),
    code(
        r"""
def find_unique(filename: str) -> Path:
    matches = [
        path
        for path in INPUT_ROOT.rglob(filename)
        if path.is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename!r}; found {matches}.")
    return matches[0]


development_path = find_unique("Developement.csv")
external_path = find_unique("External.xlsx")
external_education_path = find_unique("phas3_DR.Moftian.xlsx")
config_matches = list(INPUT_ROOT.rglob("final_model_configs.csv"))
if len(config_matches) != 1:
    raise RuntimeError(
        "Expected one attached aggregate final_model_configs.csv; found "
        f"{config_matches}."
    )
threshold_matches = list(INPUT_ROOT.rglob("development_thresholds.csv"))
dca_matches = list(INPUT_ROOT.rglob("external_dca.csv"))
manifest_matches = list(INPUT_ROOT.rglob("final_evaluation_manifest.json"))
if len(threshold_matches) != 1 or len(dca_matches) != 1 or len(manifest_matches) != 1:
    raise RuntimeError(
        "Expected one aggregate threshold table, DCA table, and final manifest; found "
        f"{threshold_matches=}, {dca_matches=}, and {manifest_matches=}."
    )
prior_output = OUTPUT_DIR / "frozen_heavy_input"
prior_final = prior_output / "final_evaluation"
prior_final.mkdir(parents=True, exist_ok=True)
for source in [config_matches[0], threshold_matches[0], dca_matches[0], manifest_matches[0]]:
    shutil.copy2(source, prior_final / source.name)
configs = pd.read_csv(config_matches[0])
selected_mask = configs["selected_by_nested_cv_for_primary_analysis"].astype(str).str.lower().eq("true")
selected_rows = configs.loc[selected_mask]
if len(selected_rows) != 1:
    raise RuntimeError("Expected exactly one Development-selected primary model.")
primary_model = str(selected_rows.iloc[0]["model_name"])
if primary_model == "tabpfn":
    if not token:
        raise RuntimeError("The selected model is TabPFN, but TABPFN_TOKEN is not enabled.")
    if not torch.cuda.is_available():
        raise RuntimeError("The selected model is TabPFN, but no CUDA GPU is available.")
    from tabpfn_extensions.interpretability import shapiq as _tabpfn_shapiq
    from tabpfn_extensions.interpretability import shapiq_to_shap_explanation
    assert hasattr(_tabpfn_shapiq, "get_tabpfn_imputation_explainer")
    assert callable(shapiq_to_shap_explanation)
    print("TabPFN SHAP API check passed.")
print("Frozen Development-selected model:", primary_model)
print("Development input found:", development_path.name)
print("External input found:", external_path.name)
print("Four-level education source found:", external_education_path.name)
print("Frozen aggregate heavy output staged locally.")
print("Participant-level contents are not displayed.")
"""
    ),
    markdown("### 5. Run the post-hoc analyses"),
    code(
        r"""
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "heavy_posthoc_analysis.py"),
    "--development",
    str(development_path),
    "--external",
    str(external_path),
    "--external-education",
    str(external_education_path),
    "--qc-output",
    str(OUTPUT_DIR / "qc"),
    "--prior-output",
    str(prior_output),
    "--output",
    str(OUTPUT_DIR / "analysis"),
    "--bootstrap-repeats",
    "2000",
]
print("Starting the post-hoc GPU run. This can take several hours.")
subprocess.run(command, cwd=PROJECT_DIR, check=True)
"""
    ),
    markdown("### 6. Validate aggregate outputs and archive"),
    code(
        r"""
manifest_path = OUTPUT_DIR / "analysis" / "posthoc_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest["status"] != "posthoc_sensitivity_transportability_and_interpretability_completed":
    raise RuntimeError(f"Unexpected post-hoc status: {manifest['status']}")
if manifest["primary_model_changed"]:
    raise RuntimeError("Leakage guard failed: the primary model changed.")
if manifest["mice_performed"]:
    raise RuntimeError("MICE was unexpectedly executed.")
if manifest["education_harmonization_mode"] != "four_level_code_matched_auxiliary_source":
    raise RuntimeError("Post-hoc analysis did not use four-level education.")
required = [
    "posthoc_sensitivity_metrics.csv",
    "subgroup_performance.csv",
    "subgroup_interaction_tests.csv",
    "local_calibration_and_brier_decomposition.csv",
    "external_calibration_before_after.png",
    "external_dca_with_ci.png",
    "selected_model_shap_global_importance.csv",
    "selected_model_shap_summary.png",
]
missing = [name for name in required if not (OUTPUT_DIR / "analysis" / name).exists()]
if missing:
    raise RuntimeError(f"Missing required outputs: {missing}")
environment = {
    "python": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "repository_commit": subprocess.check_output(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], text=True
    ).strip(),
}
(OUTPUT_DIR / "execution_environment.json").write_text(
    json.dumps(environment, indent=2), encoding="utf-8"
)
archive = shutil.make_archive(
    "/kaggle/working/mci_posthoc_outputs",
    "zip",
    root_dir=OUTPUT_DIR,
)
print(json.dumps(manifest, indent=2))
print("Aggregate archive:", archive)
"""
    ),
    markdown(
        """
## Interpretation boundary

- The original locked External validation remains the confirmatory result.
- Alternative feature sets and the transportable-core analysis are secondary,
  post-hoc sensitivity analyses and require validation in a new cohort.
- SHAP describes model associations and is not a causal analysis.
"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
