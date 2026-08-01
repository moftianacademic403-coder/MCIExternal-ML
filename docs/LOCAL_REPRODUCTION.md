# Local workstation reproduction

This repository stores code and aggregate workflow definitions only. Raw
participant-level data, cached TabPFN checkpoints, fitted models, and generated
outputs are intentionally excluded from Git.

## Required private inputs

Place these files at the repository root, or pass their paths to the runner:

- `Developement.csv`
- `External.xlsx`
- `phas3_DR.Moftian.xlsx`

The filenames are ignored by Git. Do not force-add them to a public repository.

## Environment

The completed run used Python 3.11.15, PyTorch 2.11.0+cu128, CUDA on an NVIDIA
RTX 5090, TabPFN 8.2.0, and `tabpfn-extensions[interpretability]` 0.4.1.

On the original workstation, reuse the already downloaded packages by cloning
the known-good environment:

```powershell
conda create --prefix .\.conda-env --clone D:\Nazila\NAFLD\env
.\.conda-env\python.exe -m pip install -r .\requirements-local.txt
.\.conda-env\python.exe -m pip check
```

On another workstation, create Python 3.11 environment `.conda-env`, install a
PyTorch/CUDA build supported by that machine, then install
`requirements-local.txt`. TabPFN also requires an authorized Prior Labs model
download or a valid `TABPFN_TOKEN`; the model cache is deliberately not stored
in Git.

## One-command resumable run

From PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_local_full_pipeline.ps1
```

The runner validates required files, starts nested cross-validation when no
completed manifest exists, and then runs the locked final evaluation, post-hoc
analyses, manuscript artifacts, and publication figure pack in order. Completed
stages are skipped only when their expected success manifest is present.

Custom data or Python paths can be supplied without editing the script:

```powershell
.\run_local_full_pipeline.ps1 `
  -PythonPath 'E:\envs\mci\python.exe' `
  -DevelopmentPath 'E:\private\Developement.csv' `
  -ExternalPath 'E:\private\External.xlsx' `
  -ExternalEducationPath 'E:\private\phas3_DR.Moftian.xlsx'
```

Run status and per-stage logs are written under
`outputs/workstation_heavy_four_level_final`. Generated outputs remain local and
can be regenerated from the tracked code plus the private inputs.

## Publication figures

The final stage creates eight journal-oriented figures in PNG (400 dpi), TIFF
(600 dpi, LZW), vector PDF, and SVG. It includes global SHAP, local SHAP for
three synthetic score-tertile profiles, calibration, decision-curve,
discrimination, subgroup, stability, and transportability views. The manifest
is `outputs/publication_figure_pack/publication_figure_manifest.json`.

SHAP values are TabPFN logit-scale attributions. Local profiles are synthetic
medians/modes and are not identifiable participants. Interpretations are
associational, not causal.
