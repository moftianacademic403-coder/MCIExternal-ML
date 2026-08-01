# MCI screening with external validation

This public code-only repository contains the reproducible analysis pipeline for the MCI screening project. Participant-level data, fitted models, predictions, locally executed notebooks, and run outputs are intentionally excluded.

## Fixed study policy

- `Developement.csv` is the only development source.
- `External.xlsx` is used first for locked external validation. Any subsequent local recalibration or local-threshold analysis is secondary, cross-fitted within External, and reported separately.
- External data must not influence feature selection, hyperparameter tuning, Development-model calibration, Development threshold selection, or model-family selection.
- No harmonization rule is applied until the user confirms the variable definition, unit, categories, and any derivation.
- Raw participant-level files are never copied into this repository and are never modified.

## Current stage

The complete four-level pipeline has been executed locally and reached
`ready_for_manuscript_writing`. Five prespecified model families were compared
using Development train-80 repeated nested CV only. TabPFN was selected, frozen,
and evaluated once on the locked internal test and External cohort. The original
locked result remains the primary validation result; External local
recalibration and threshold updating are explicitly secondary analyses.

The follow-up GPU workflow adds the analyses that must not alter the primary
model: 30-repeat mRMR stability, Development-only elastic-net stability
selection, all-feature and alternative-feature-set sensitivity analyses,
complete-case analysis, training-fold winsorization, exclusion of ADL/IADL,
subgroup interaction tests, full-bootstrap local calibration and Brier
decomposition, publication figures, and selected-model SHAP. MICE is excluded at
the investigator's request. A transportable-core analysis uses External
predictor distributions but not External outcomes and is therefore labelled
post-hoc; it requires a new cohort before any confirmatory claim.

- Prepared heavy-run configuration: `config/heavy_kaggle.json`
- Kaggle handoff notes: `docs/KAGGLE_HEAVY_RUN.md`
- Kaggle GPU notebook: `kaggle/mci-heavy-nested-cv/mci_heavy_nested_cv_kaggle.ipynb`
- Heavy runners: `src/heavy_nested_cv.py` and `src/heavy_final_evaluation.py`
- Post-hoc Kaggle notebook: `kaggle/mci-posthoc-sensitivity/mci_posthoc_sensitivity_kaggle.ipynb`
- Post-hoc runner: `src/heavy_posthoc_analysis.py`
- Predictor-shift audit: `src/transportability_audit.py`
- Resumable local runner: `run_local_full_pipeline.ps1`
- Local environment and rerun guide: `docs/LOCAL_REPRODUCTION.md`
- Publication figure builder: `src/build_publication_figure_pack.py`

Participant-level predictions are not exported. Aggregate executed outputs remain
outside the public repository.
