# Polynomial regression — ML Assignment 1

**Pulkit Pandey · BT2024060**

Repository: [COolAlien35/polynomial-regression](https://github.com/COolAlien35/polynomial-regression)

Polynomial models for the two personalised geothermal datasets. Degree selection,
regularisation, feature selection and stability comparisons use development data.
The selected settings were frozen before the reserved holdout assessment.

## Submission deliverables

| Deliverable | File |
|---|---|
| Five-page PDF report, including selection heatmaps and validation figures | [BT2024060_Report.pdf](submission/BT2024060_Report.pdf) |
| var1 test predictions | [BT2024060_pred_var1.csv](submission/BT2024060_pred_var1.csv) |
| var2 test predictions | [BT2024060_pred_var2.csv](submission/BT2024060_pred_var2.csv) |
| Training code | [final_model_check.py](scripts/final_model_check.py) and [polynomial_models.py](scripts/polynomial_models.py) |
| Inference code | [predict_submission.py](scripts/predict_submission.py) |

Both CSVs contain exactly 1,000 rows and one `y` column, with no index. Predictions
retain the original test row order. Submit the PDF, the two CSVs and this repository
URL. The local work must be committed and pushed before the URL contains it.

## Selected models and results

| Problem | Inputs | Total degree | Model | Repeated CV MSE | Holdout MSE | Holdout R² |
|---|---|---:|---|---:|---:|---:|
| var1 | x1–x6 | 5 | Equal average of Lasso α=0.01 and an OLS refit of terms selected by Lasso α=0.03 | 0.333282 | 0.318443 | 0.967621 |
| var2 | x1–x3 | 11 | Ridge α=3 | 0.273724 | 0.398228 | 0.993394 |

The var1 average is a single polynomial: the raw coefficients are combined for
inference. Every term respects the assignment's total-degree definition and the
limits of 10 for var1 and 20 for var2. Repeated CV uses five fresh grouped 5-fold
splits; duplicate inputs stay together. Scaling and term selection are fitted
inside each training fold. The holdout contains 200 var1 and 202 var2 rows.

These are internal validation results; hidden test scores are unknown. var1 has a
train/test boundary-distribution shift, and rare var2 corners are an observed
weakness. Full discussion and uncertainty intervals are in the
[supporting validation review](outputs/final_validation/validation_report.md).

## Setup and reproduce

Run commands from the repository root. The checked environment uses Python
3.13.2; dependency versions are pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Regenerate both prediction CSVs directly from the included raw coefficients:

```bash
python scripts/predict_submission.py
```

Refit the fixed selected models on all supplied labelled rows, export their
coefficients, and regenerate predictions:

```bash
python scripts/final_model_check.py
python scripts/predict_submission.py
python scripts/check_deliverables.py
```

Saved configurations and training-file hashes are checked. Existing holdout
results are reused on reruns. The final refit does not perform another model
search. Coefficient-based inference uses only NumPy and pandas, without labels.

Rebuild the supporting review and five-page PDF from the preserved results:

```bash
python scripts/validation_summary.py
python scripts/build_report.py
```

## Project layout

```text
data/             Original four datasets and sample submission
docs/             Assignment specification and deliverable audit
scripts/          Training, inference, inspection and analysis code
submission/       The PDF and two prediction CSVs to submit
outputs/          Recorded experiments, splits, metrics and final coefficients
requirements.txt  Pinned Python dependencies
```

Original dataset bytes are preserved. Local environments, Python caches, macOS
metadata, Matplotlib caches and PDF page previews are excluded from Git.

## Analysis workflow

The saved outputs record the completed searches. Re-running the full Lasso grid
can take considerable time; it is unnecessary for reproducing the submission.

| Scripts under `scripts/` | Purpose |
|---|---|
| `inspect_data.py`, `inspect_patterns.py` | Data quality, distributions, boundaries, correlations and marginal patterns |
| `baseline.py` | Fixed grouped development/holdout split and mean/linear baselines |
| `compare_degrees.py` | Ordinary polynomial regression by degree |
| `tune_ridge.py --full-range --resume`, `tune_lasso.py` | Degree and penalty searches over the allowed ranges; convergence checks |
| `search_features.py` | Feature-subset screening |
| `refine_models.py`, `refine_lasso.py`, `verify_lasso_candidate.py` | Penalty refinements and convergence verification |
| `improve_models.py` | Alternative polynomial bases, Elastic Net, sparse refits and boundary weighting |
| `robustness_check.py` | Six shortlisted candidates over five fresh grouped split repetitions |
| `learning_curves.py` | Fixed-model performance at increasing training sizes |
| `final_model_check.py` | Frozen-choice assessment, full-data fitting and polynomial export |
| `validation_summary.py`, `build_report.py` | Supporting review, diagnostic figures and PDF |
| `check_deliverables.py` | Format, degree, split-isolation, metric and inference consistency audit |

The search was staged; it did not exhaust every possible joint configuration.
Initial descriptive EDA covered the full labelled data. The reserved partition
was subsequently excluded from model fitting, tuning and selection. Learning
curves and validation justify model complexity; iterative early stopping is not
a proof that a polynomial degree is correct.

## Push the completed assignment

The configured `origin` points to the repository above; the branch is `main`.
Review the changes, then commit and push:

```bash
git status --short
git add .
git commit -m "Complete polynomial regression assignment for BT2024060"
git push origin main
```

After pushing, verify that the report, both CSVs, `scripts/`, `requirements.txt`
and `outputs/final_models/` are visible on GitHub, and that your evaluator can
access the repository.
