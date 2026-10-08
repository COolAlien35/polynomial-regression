"""Save a concise validation report and holdout diagnostic plots."""

import json
import os
from pathlib import Path

from project_paths import ROOT
CACHE = ROOT / "outputs" / ".matplotlib_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    final = ROOT / "outputs" / "final_validation"
    rows = []
    for problem in ["var1", "var2"]:
        metrics = json.loads((final / f"{problem}_holdout_metrics.json").read_text())
        summary = pd.read_csv(ROOT / "outputs" / "robustness" / f"{problem}_summary.csv")
        cv = summary[summary.candidate.eq(metrics["candidate"])].iloc[0]
        rows.append({
            "problem": problem, "candidate": metrics["candidate"],
            "repeated_cv_mse": cv.validation_mse, "repeated_cv_split_sd": cv.split_mse_std,
            "repeated_cv_r2": cv.validation_r2, **metrics,
        })
        points = pd.read_csv(final / f"{problem}_holdout_predictions.csv")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        scatter = axes[0].scatter(points.actual_y, points.predicted_y,
                                 c=points.boundary_feature_count, cmap="viridis", s=22, alpha=0.8)
        lo = min(points.actual_y.min(), points.predicted_y.min())
        hi = max(points.actual_y.max(), points.predicted_y.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[0].set(xlabel="Actual y", ylabel="Predicted y", title="Reserved holdout predictions")
        fig.colorbar(scatter, ax=axes[0], label="Boundary feature count")
        axes[1].scatter(points.predicted_y, points.residual, c=points.boundary_feature_count,
                        cmap="viridis", s=22, alpha=0.8)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set(xlabel="Predicted y", ylabel="Actual minus predicted", title="Residuals")
        groups = points.groupby("boundary_feature_count").residual
        values = [group.to_numpy() for _, group in groups]
        labels = [f"{count}\n(n={len(group)})" for count, group in groups]
        axes[2].boxplot(values, tick_labels=labels)
        axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[2].set(xlabel="Boundary feature count", ylabel="Residual", title="Boundary diagnostics")
        fig.suptitle(f"{problem}: MSE {metrics['holdout_mse']:.4f}, R2 {metrics['holdout_r2']:.4f}")
        fig.tight_layout()
        fig.savefig(final / f"{problem}_holdout_diagnostics.png", dpi=180)
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(final / "validation_summary.csv", index=False)

    report = """# Final modelling and robustness review

**Pulkit Pandey · BT2024060**

## Selected polynomial models

- **var1:** all six inputs, total degree 5. Equal average of (a) Lasso with alpha 0.01 and (b) an ordinary least-squares refit of terms selected by Lasso with alpha 0.03. Both selections and fits occur inside each validation training fold. The average is itself a single degree-5 polynomial; its combined coefficients are exported.
- **var2:** all three inputs, total degree 11, Ridge alpha 3.

Feature scaling is fitted within each fold, after polynomial expansion. No training row is used to fit the model that produces its validation prediction. Identical input rows stay in the same partition and fold. The two excluded assignment lines played no role in selection.

## Improvement searches

The original search covered degrees 1–10 for var1 and 1–20 for var2 with Ridge and Lasso penalties. Feature-subset screening near promising degrees favoured retaining every supplied feature. Further screening compared 255 configurations/control settings across Elastic Net, Legendre/Chebyshev polynomial bases, Lasso selection followed by coefficient refitting, and boundary-weighted training for var1. This was a staged search, not an exhaustive search over every joint configuration.

Changing the polynomial basis did not improve the leading validation scores. Boundary weighting worsened var1 performance. Var2's sparse refit improved the original folds slightly, but the gain did not persist across fresh splits. Degree-11 Ridge had more consistent split performance, so it was retained.

Var1's fixed equal average improved MSE in all five fresh split repetitions compared with the original Lasso. Its repeated-validation MSE decreased from 0.337740 to 0.333282, a modest 1.3% improvement. Mixing weights were fixed at 0.5; none were fitted to validation targets.

## Repeated grouped validation and learning curves

Six shortlisted candidates per problem were compared over five fresh, shuffled 5-fold group splits (seeds 7, 19, 43, 71, 101). Each candidate therefore had 25 validation fits. Candidate selection used development data only. Reported repeated R2 values below are computed on the complete out-of-fold prediction vector per repetition and then averaged. They are not the same aggregation as averaging individual fold R2 scores.

Learning curves used fixed model settings and training fractions 40%, 60%, 80%, and 100% inside five validation folds. Var1's mean validation MSE decreased from 0.491250 to 0.328600 as training rows per fold increased from 256 to 640. Var2's decreased from 0.666476 to 0.270118 as rows increased from about 255 to 638. Every selected-model fit converged. The training–validation gap narrowed with more data; it did not vanish.

## Reserved holdout assessment

The reserved partition contained 200 var1 and 202 var2 rows. Selected configurations and training-file hashes were saved before assessment. Holdout scores did not determine model selection or subsequent hyperparameters. Initial descriptive EDA had summarised the full supplied training data, including this partition; the partition was excluded from model fitting and tuning. The holdout assessment is separate from the assignment's hidden test evaluation.

| Problem | Repeated CV MSE | Split SD | Repeated CV R2 | Holdout MSE | Holdout R2 |
|---|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (f"| {row['problem']} | {row['repeated_cv_mse']:.6f} | "
                   f"{row['repeated_cv_split_sd']:.6f} | {row['repeated_cv_r2']:.6f} | "
                   f"{row['holdout_mse']:.6f} | {row['holdout_r2']:.6f} |\n")
    report += "\nApproximate 95% holdout intervals use 2,000 bootstrap resamples of input groups, preserving repeated observations together:\n\n"
    for row in rows:
        report += (f"- {row['problem']}: MSE [{row['mse_ci_low']:.4f}, {row['mse_ci_high']:.4f}]; "
                   f"R2 [{row['r2_ci_low']:.4f}, {row['r2_ci_high']:.4f}].\n")
    report += """
## Practical limits

These results support generalisation within the sampled input domain, but do not prove globally optimal parameters or zero overfitting/underfitting. Repeated splits share observations, so their standard deviation is a stability measure, not an independent confidence interval. Extensive development searches can make their best CV scores optimistic.

Var1 test inputs contain substantially more simultaneous boundary values than its training data, so training-distribution validation is an imperfect proxy for hidden-test error. Small extreme-boundary groups provide limited evidence.

Var2's six holdout observations with all three coordinates at ±1 had MSE 3.4145, versus 0.2341 for its 77 interior holdout rows. Three repeated observations at one corner contributed large errors. These corners are the main observed weakness. Refitting on all training data adds the reserved observations back into training, but cannot guarantee accuracy at unobserved combinations. Predictions were not clipped to the training target range.

Var2 development data contain only three repeated-input groups with enough observations to estimate within-group variation. Their pooled target variance was approximately 0.2253 (six residual degrees of freedom). This suggests noisy targets; it is a rough local observation, not a proven global noise floor or ceiling on R2.

## Final fitting and export

After assessment, the fixed models were trained on all 1,000 labelled rows per problem. Each submission contains 1,000 predictions under a single `y` column in the original test order. Distinct test inputs are evaluated once and expanded back to their original rows, ensuring identical inputs receive identical saved predictions.

The pipelines were saved as joblib models, and combined raw polynomial coefficients were saved in JSON/CSV. Exported polynomial predictions matched pipeline predictions to about 1e-14. `predict_submission.py` reproduces the submission CSVs from those coefficients without retraining or accessing labels.

Files:

- `submission/BT2024060_pred_var1.csv`, `submission/BT2024060_pred_var2.csv`
- `outputs/final_models/var1_polynomial.json`, `var2_polynomial.json`
- `outputs/final_validation/validation_summary.csv`
- `outputs/final_validation/var1_holdout_diagnostics.png`, `var2_holdout_diagnostics.png`
- `outputs/learning_curves/var1_learning_curve.png`, `var2_learning_curve.png`

The five-page assignment report is `submission/BT2024060_Report.pdf`. This review provides additional supporting detail.
"""
    (final / "validation_report.md").write_text(report)
    print(table[["problem", "repeated_cv_mse", "repeated_cv_r2", "holdout_mse", "holdout_r2"]]
          .round(6).to_string(index=False))
    print(f"Saved review and plots to {final}")


if __name__ == "__main__":
    main()
