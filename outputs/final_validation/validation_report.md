# Final modelling and robustness review

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
| var1 | 0.333282 | 0.006169 | 0.966759 | 0.318443 | 0.967621 |
| var2 | 0.273724 | 0.005876 | 0.993277 | 0.398228 | 0.993394 |

Approximate 95% holdout intervals use 2,000 bootstrap resamples of input groups, preserving repeated observations together:

- var1: MSE [0.2616, 0.3769]; R2 [0.9570, 0.9748].
- var2: MSE [0.2489, 0.6194]; R2 [0.9911, 0.9950].

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
