# Kaggle GPU handoff for the MCI project

Status: locally implemented and smoke-tested; not yet published or remotely executed.

## What the Kaggle run does

The notebook `kaggle/mci-heavy-nested-cv/mci_heavy_nested_cv_kaggle.ipynb` runs two ordered stages:

1. **Development-only model-family selection**
   - locked Development train-80 only;
   - repeated nested CV: 5 outer folds × 3 repeats, with 4 inner folds;
   - mRMR recomputed inside every inner-training fold and on each full outer-training fold;
   - 40 sampled configurations per classical family;
   - candidate feature counts 5, 10, 15, 20, 25 and 30;
   - elastic-net logistic regression, RBF SVM, random forest, XGBoost and TabPFN-3;
   - primary selection criterion: mean outer-fold AUROC;
   - secondary criterion: mean outer-fold Brier score;
   - internal test and External are not used for model selection.

2. **Development finalization and locked evaluation**
   - full-training hyperparameter/feature-count selection inside Development train-80;
   - Platt/logistic calibration from cross-fitted Development train-80 predictions;
   - Youden and 80%, 85%, 90% sensitivity thresholds learned from Development only;
   - one-time locked evaluation on the internal 20% and External;
   - 2,000 bootstrap repetitions for discrimination and operating metrics;
   - calibration intercept/slope, reliability bins and DCA with confidence intervals;
   - a separately labelled 10-fold OOF External local-recalibration/threshold analysis;
   - no full base-model refit on External.

Subgroup interaction testing, complete-case sensitivity analysis, and final selected-model SHAP are intentionally listed as pending in the final manifest. MICE is excluded at the investigator's request. These analyses should be run after the selected family is known rather than multiplying expensive analyses across every candidate family.

## TabPFN source and version

- Package: `tabpfn==8.2.0`.
- Default checkpoint/model family at this pinned release: TabPFN-3.
- The code follows the official interface: `from tabpfn import TabPFNClassifier`.
- TabPFN receives numeric and categorical columns with native missing values. It is not one-hot encoded or scaled.
- A CUDA GPU is required by the heavy runner.
- The first model download can require accepting the Prior Labs model terms. For unattended Kaggle execution, store the resulting `TABPFN_TOKEN` in Kaggle Secrets under exactly that name. Never paste it into code, a notebook cell, GitHub, or chat.

## Privacy-safe architecture

- GitHub contains code, configuration, aggregate example outputs and documentation only.
- `Developement.csv` and `External.xlsx` must be attached through a **private Kaggle dataset**.
- The repository `.gitignore` excludes CSV/XLS/XLSX source files.
- The runners do not write participant-level predictions. Only aggregate fold metrics, selected configurations, stability summaries and figures/tables are exported.

For the easiest unattended run, use a **public code-only GitHub repository** and a **private Kaggle data source**. A private GitHub repository would require an additional GitHub token in Kaggle Secrets and creates avoidable authentication failure risk.

## Remaining scientific confirmations

Before treating the full run as the manuscript run, record:

1. The exact operational definition used to create `MCI`.
2. Whether `ADL` or `IADL` contributed directly or indirectly to that definition. If yes, exclusion of these variables should be the primary analysis or a mandatory strong sensitivity analysis.
3. Whether sensitivity 85% is the clinically chosen main operating target. Until confirmed, 80%, 85% and 90% remain scenarios.
4. The clinically relevant DCA threshold-probability range. The value 0.30 is currently illustrative, not prespecified.

## One-time setup

1. Create a new public GitHub repository for code only.
2. Create a private Kaggle dataset containing exactly:
   - `Developement.csv`
   - `External.xlsx`
   - `phas3_DR.Moftian.xlsx`
3. In Kaggle, accept the TabPFN model terms and add `TABPFN_TOKEN` as a Kaggle Secret if requested.
4. Enable Internet and an NVIDIA GPU for the Kaggle notebook.

After the repository and private Kaggle dataset exist, fill the template locally:

```powershell
python src\prepare_kaggle_kernel.py `
  --kaggle-username YOUR_KAGGLE_USERNAME `
  --private-dataset-slug YOUR_PRIVATE_DATASET_SLUG `
  --repository-url https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git `
  --repository-branch main
```

This creates the untracked user-specific file:

```text
kaggle/mci-heavy-nested-cv/kernel-metadata.json
```

## Start the cloud job

Install and authenticate the official Kaggle CLI once:

```powershell
python -m pip install kaggle
kaggle auth login
```

Submit the notebook:

```powershell
kaggle kernels push -p kaggle\mci-heavy-nested-cv --accelerator NvidiaTeslaT4 --timeout 43200
```

Check that Kaggle accepted the version and that its status is pending or running:

```powershell
kaggle kernels status YOUR_KAGGLE_USERNAME/mci-heavy-selection-and-validation
```

Once the push has succeeded and Kaggle shows the cloud job as pending/running, the local laptop is no longer doing the computation and can be shut down. The job can later be checked from another device.

After completion, download only aggregate outputs:

```powershell
kaggle kernels output YOUR_KAGGLE_USERNAME/mci-heavy-selection-and-validation `
  -p outputs\kaggle_download --file-pattern ".*mci_heavy_nested_outputs.zip$"
```

## Validation already completed locally

- `src/heavy_nested_cv.py` compiled and completed a two-fold/two-inner-fold classical-model smoke run on the actual harmonized inputs.
- `src/heavy_final_evaluation.py` compiled and completed the smoke finalization, locked internal/External evaluation, local update, reliability and DCA stages.
- The Kaggle notebook has valid `nbformat` structure and contains explicit GPU, secret, input-count and leakage guards.
- TabPFN itself and the full repeated nested-CV run have not been executed locally because this machine has no configured Kaggle GPU session or TabPFN installation.
