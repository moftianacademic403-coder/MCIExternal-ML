# MCI screening with external validation

This public code-only repository contains the reproducible analysis pipeline for the MCI screening project. Participant-level data, fitted models, predictions, locally executed notebooks, and run outputs are intentionally excluded.

## Fixed study policy

- `Developement.csv` is the only development source.
- `External.xlsx` is used first for locked external validation. Any subsequent local recalibration or local-threshold analysis is secondary, cross-fitted within External, and reported separately.
- External data must not influence feature selection, hyperparameter tuning, Development-model calibration, Development threshold selection, or model-family selection.
- No harmonization rule is applied until the user confirms the variable definition, unit, categories, and any derivation.
- Raw participant-level files are never copied into this repository and are never modified.

## Current stage

The local lightweight pipeline is complete through QC, locked splitting, fold-fitted preprocessing, 30-repeat mRMR stability, four model families, locked internal and external evaluation, calibration, subgroup/sensitivity analyses, SHAP, training-only operating points, prevalence scenarios, cross-validated External local recalibration, held-out local threshold updating, Brier decomposition, updated DCA with paired-bootstrap confidence intervals, reliability diagrams, and a three-layer deployment summary. Its executed notebooks and outputs remain local and are not part of this public repository.

- Prepared heavy-run configuration: `config/heavy_kaggle.json`
- Kaggle handoff notes: `docs/KAGGLE_HEAVY_RUN.md`
- Kaggle GPU notebook: `kaggle/mci-heavy-nested-cv/mci_heavy_nested_cv_kaggle.ipynb`
- Heavy runners: `src/heavy_nested_cv.py` and `src/heavy_final_evaluation.py`

The heavy code is locally smoke-tested but has not been remotely executed. The manuscript run should not be launched until the exact MCI outcome definition and any ADL/IADL contribution to that definition are recorded.
