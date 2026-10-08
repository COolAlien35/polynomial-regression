# Deliverable audit — Pulkit Pandey, BT2024060

Reviewed on 8 October 2026 against [the assignment](ML_Assignment_1.pdf).
The supplied datasets were used separately, and the two excluded degree/feature
hints were not used to choose the models.

## Required deliverables

| Requirement | Status and evidence |
|---|---|
| Concise PDF, maximum 4–5 pages | **Pass:** [report](../submission/BT2024060_Report.pdf) has exactly five A4 pages. |
| Approach, selected degrees, rationale and techniques | **Pass:** pages 1–3 and 5 explain preprocessing, total-degree expansion, Ridge/Lasso, refinements, feature screening, sparse refitting, robustness comparisons and degree rationale. |
| Important figures and model-selection heatmaps | **Pass:** input correlations, boundary coverage, four degree/penalty heatmaps, shortlist comparisons, learning curves, predicted-versus-actual and residual plots are included. |
| Name and roll number | **Pass:** Pulkit Pandey and BT2024060 appear in the PDF; metadata identifies the author. |
| var1 prediction CSV | **Pass:** `submission/BT2024060_pred_var1.csv`, exactly 1,000 rows and one `y` column. |
| var2 prediction CSV | **Pass:** `submission/BT2024060_pred_var2.csv`, exactly 1,000 rows and one `y` column. |
| Sample submission format and original test order | **Pass:** no index column; coefficient inference reproduces each row; repeated inputs receive exactly identical saved predictions. |
| Polynomial regression only | **Pass:** raw exported coefficients represent the final single polynomial for each problem, including the fixed var1 average. |
| Total-degree limits | **Pass:** var1 degree 5 ≤ 10; var2 degree 11 ≤ 20. The sum of feature powers is checked for every exported term. |
| Training and inference code | **Pass locally:** all code is under `scripts/`; fixed-model training and coefficient-based inference run successfully. |
| GitHub repository | **Pending your push:** local `origin` matches `https://github.com/COolAlien35/polynomial-regression.git`; publication and evaluator access must be checked after pushing. |

## Verification performed

- Refit the frozen selected models on the original 1,000 training rows per
  problem after reorganisation; all fits converged and no parameters changed.
- Regenerated both submission CSVs through independent coefficient-based
  inference. Saved pipelines, exported formulas and CSVs agree to about 1e-14.
- Recomputed holdout MSE/R² from saved predictions and repeated CV MSE/R² from
  the complete saved out-of-fold vectors. The reported final scores agree.
- Checked actual repeated input groups against development/holdout partitions
  and original folds. No identical-input group crosses these splits.
- Verified all 25 repeated-validation fits for each final model converged.
- Checked unchanged original dataset and assignment PDF bytes; unchanged saved
  splits, training hashes, frozen choices and reserved-holdout score JSON files.
- Compiled all source scripts and checked the pinned package environment;
  `pip check` found no broken requirements.
- Opened the final PDF with macOS PDFKit, confirmed five A4 pages, extracted
  selectable text, verified the author and repository link, and visually reviewed
  all five rendered pages for legibility and layout.
- Tested reproduction from a separate directory containing only files eligible
  for Git, to check that ignored local caches and working-directory assumptions
  are unnecessary.

Re-run the numerical/artifact audit with:

```bash
python scripts/check_deliverables.py
```

Its detailed result, including hashes and package versions, is saved in
[`outputs/deliverable_audit.json`](../outputs/deliverable_audit.json).

## Final evaluation results

| Problem | Repeated CV MSE | Repeated CV R² | Holdout MSE | Holdout R² |
|---|---:|---:|---:|---:|
| var1 | 0.333282 | 0.966759 | 0.318443 | 0.967621 |
| var2 | 0.273724 | 0.993277 | 0.398228 | 0.993394 |

These are internal validation scores, not hidden-test results. They support the
chosen settings but do not prove global optimality or zero overfitting. The
report discusses var1's boundary shift and var2's rare-corner weakness. The
reserved holdout was excluded from model fitting/tuning/selection; initial
descriptive EDA did include all labelled rows.

## Repository organisation and remaining action

The root now contains `README.md`, `requirements.txt`, `.gitignore`, and the
`data/`, `docs/`, `scripts/`, `submission/`, `outputs/` folders. The local `.venv`
remains in place and is ignored. All reproducibility commands and data paths
have been updated. Original files were moved intact; generated root caches were
archived outside the project. Analysis results are retained, while regenerable
PDF previews and caches are excluded from Git.

Commit and push the complete local directory using the instructions in the
[README](../README.md), then verify that the evaluator can open the GitHub URL
and see all deliverables and source code. Direct remote inspection failed in
this execution environment because GitHub's hostname could not be resolved;
the supplied URL is not certified as publicly accessible by this audit.
