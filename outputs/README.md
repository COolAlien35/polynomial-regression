# Recorded analysis outputs

These results preserve the completed analysis and allow rebuilding the report.

- `inspection/`, `patterns/`: descriptive EDA and data-quality checks.
- `baseline/`: original row partitions, grouped fold assignments and baselines.
- `degree_comparison/`, `ridge_comparison/`, `lasso_comparison/`: initial degree
  and regularisation searches. Failed-convergence Lasso candidates are recorded,
  but are ineligible for selection.
- `feature_search/`, `refinement/`, `lasso_refinement/`, `lasso_verification/`:
  staged feature and penalty checks. Some early strict Lasso runs did not
  converge; later verification is recorded separately.
- `improvements/`: 255 additional configurations/control settings.
- `robustness/`: frozen shortlists, fresh grouped splits, out-of-fold predictions
  and summaries. These comparisons determine the final selection.
- `learning_curves/`: fixed selected models at increasing training sizes.
- `final_validation/`: frozen final choices, training hashes, reserved holdout
  predictions/scores/intervals and the supporting review.
- `final_models/`: fitted pipelines and raw polynomial JSON/CSV coefficients.
- `deliverable_audit.json`: generated consistency audit of the final submission.

`model_comparison.csv` is an intermediate Ridge/Lasso comparison, preceding the
robustness search; use `final_validation/validation_summary.csv` for final scores.

`report/` contains regenerable PDF previews and is excluded from Git. Run
`python scripts/build_report.py` from the repository root to recreate them.
The report itself is included in `submission/`. Saved joblib models require the
pinned environment and the custom estimator module in `scripts/`; exported
polynomial JSON supports inference without loading joblib.
