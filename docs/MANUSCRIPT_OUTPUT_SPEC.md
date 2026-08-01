# Manuscript output completion specification

The analysis is ready for Results writing only when `manuscript_readiness_manifest.json`
has status `ready_for_manuscript_writing` and all of the following conditions hold:

- four-level education is matched one-to-one to External by `Code`;
- the full repeated nested-CV model-family comparison is rerun after that harmonization change;
- model selection uses Development train-80 only;
- the locked internal test and locked External validation are evaluated once;
- local External recalibration and local threshold updating are reported separately;
- 30-repeat mRMR stability, alternative selector, complete-case, winsorization,
  ADL/IADL exclusion and transportability sensitivities are present;
- MICE is not run;
- DCA includes bootstrap confidence intervals and screen-all/screen-none references;
- subgroup performance includes confidence intervals and interaction tests;
- SHAP is generated for the locked Development-selected model (TabPFN-native
  SHAP-IQ when TabPFN wins; model-appropriate SHAP otherwise);
- all manuscript tables and figures are aggregate-only and contain no participant-level predictions.

Required primary figures are participant flow, nested model comparison, ROC curves,
mRMR stability, calibration before/after, DCA, subgroup performance and SHAP.
Precision-recall, transportability and feature-set sensitivity figures are supplementary.
