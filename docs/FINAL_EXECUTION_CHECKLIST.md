# Final four-level MCI execution checklist

The manuscript run is complete only after every guard below passes. Smoke outputs
are never substituted for manuscript results.

## 1. Publish the frozen code

- Branch: `codex/final-four-level-manuscript-pipeline`
- Repository: `https://github.com/moftianacademic403-coder/MCIExternal-ML.git`
- Raw participant data and `outputs/` remain excluded by `.gitignore`.

Configure both notebooks after the branch is available:

```powershell
python src\prepare_kaggle_kernel.py `
  --kaggle-username nazilamoftian `
  --private-dataset-slug mciexternal `
  --repository-url https://github.com/moftianacademic403-coder/MCIExternal-ML.git `
  --repository-branch codex/final-four-level-manuscript-pipeline

python src\prepare_kaggle_posthoc_kernel.py `
  --kaggle-username nazilamoftian `
  --private-dataset-slug mciexternal `
  --aggregate-results-slug mci-heavy-aggregate-results `
  --repository-url https://github.com/moftianacademic403-coder/MCIExternal-ML.git `
  --repository-branch codex/final-four-level-manuscript-pipeline
```

## 2. Run and validate the heavy kernel

The private `mciexternal` dataset must contain exactly the three required source
files: `Developement.csv`, `External.xlsx`, and `phas3_DR.Moftian.xlsx`. Enable
Internet, a T4 GPU, and the existing `TABPFN_TOKEN` secret.

```powershell
kaggle kernels push -p kaggle\mci-heavy-nested-cv
kaggle kernels status nazilamoftian/mci-heavy-selection-and-validation-with-tabpfn
kaggle kernels output nazilamoftian/mci-heavy-selection-and-validation-with-tabpfn `
  -p outputs\kaggle_heavy_four_level_final
```

Required statuses:

- `selection/nested_cv_manifest.json`: `development_only_nested_cv_model_family_selected`
- `final_evaluation/final_evaluation_manifest.json`: `manuscript_grade_locked_evaluation_completed`
- education mode: `four_level_code_matched_auxiliary_source`
- TabPFN included in the family comparison
- no participant-level predictions written

## 3. Transfer only aggregate heavy inputs

Stage only the four post-hoc inputs (three aggregate tables plus the heavy-run
validation manifest), inspect the transfer manifest, then update the private
Kaggle dataset `mci-heavy-aggregate-results`.

```powershell
python src\prepare_kaggle_aggregate_dataset.py `
  --heavy-output outputs\kaggle_heavy_four_level_final `
  --staging kaggle\private-upload\mci-heavy-aggregate-results-four-level `
  --owner nazilamoftian

kaggle datasets version `
  -p kaggle\private-upload\mci-heavy-aggregate-results-four-level `
  -m "Final four-level education heavy aggregate results"
```

## 4. Run the post-hoc kernel

MICE is intentionally absent. If TabPFN is the Development-selected family,
enable the same `TABPFN_TOKEN`; otherwise the notebook uses selected-model SHAP
without requiring TabPFN authentication.

```powershell
kaggle kernels push -p kaggle\mci-posthoc-sensitivity
kaggle kernels status nazilamoftian/mci-posthoc-transportability-sensitivity-and-shap
kaggle kernels output nazilamoftian/mci-posthoc-transportability-sensitivity-and-shap `
  -p outputs\kaggle_posthoc_four_level_final
```

Required post-hoc status:
`posthoc_sensitivity_transportability_and_interpretability_completed`.

## 5. Build the manuscript package

```powershell
python src\manuscript_artifacts.py `
  --development "D:\Papers\Cognetive impairment\Final with external\Developement.csv" `
  --external "D:\Papers\Cognetive impairment\Final with external\External.xlsx" `
  --external-education "D:\Papers\Cognetive impairment\Final with external\phas3_DR.Moftian.xlsx" `
  --heavy-output outputs\kaggle_heavy_four_level_final\mci_heavy_nested_outputs `
  --posthoc-output outputs\kaggle_posthoc_four_level_final\mci_posthoc_outputs\analysis `
  --output outputs\manuscript_four_level_final
```

The final gate is
`outputs/manuscript_four_level_final/manuscript_readiness_manifest.json` with
status `ready_for_manuscript_writing`. Build and visually verify
`MCI_Manuscript_Tables.xlsx` only after that gate passes.
