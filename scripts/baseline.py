"""Create fixed data splits and evaluate simple regression baselines.

Run: python scripts/baseline.py
The reserved holdout is not scored during model development.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "baseline"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    diagnostics = []
    for problem in ["var1", "var2"]:
        frame = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
        X = frame.drop(columns="y")
        y = frame["y"]

        # Identical input rows share a group, even when their targets differ.
        groups = pd.MultiIndex.from_frame(X).factorize()[0]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        development, holdout = next(splitter.split(X, y, groups))
        X_dev = X.iloc[development]
        y_dev = y.iloc[development]
        boundary_counts = X_dev.abs().eq(1).sum(axis=1).to_numpy()
        cv_splits = list(GroupKFold(n_splits=5).split(X_dev, y_dev, groups[development]))

        # Save original row positions so later scripts reuse these exact splits.
        assignments = pd.DataFrame({
            "row_position": np.arange(len(frame)),
            "group": groups,
            "partition": "holdout",
            "validation_fold": -1,
        })
        assignments.loc[development, "partition"] = "development"
        for fold, (_, validation) in enumerate(cv_splits):
            assignments.loc[development[validation], "validation_fold"] = fold
        assignments.to_csv(OUTPUT / f"{problem}_splits.csv", index=False)

        print(f"\n{problem.upper()}: {len(development)} development rows, "
              f"{len(holdout)} reserved holdout rows", flush=True)
        # Inspect input coverage without consulting reserved holdout targets.
        test = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_test_{problem}.csv")
        coverage = pd.DataFrame({
            "development": X_dev.abs().eq(1).sum(axis=1).value_counts(normalize=True),
            "holdout": X.iloc[holdout].abs().eq(1).sum(axis=1).value_counts(normalize=True),
            "test": test.abs().eq(1).sum(axis=1).value_counts(normalize=True),
        }).fillna(0).sort_index() * 100
        coverage.index.name = "boundary_feature_count"
        coverage.to_csv(OUTPUT / f"{problem}_boundary_coverage.csv")
        print("\nPercentage of rows by number of features exactly at -1 or +1:")
        print(coverage.round(2).to_string())
        models = {
            "Predict training mean": DummyRegressor(strategy="mean"),
            "Degree 1, all features": make_pipeline(
                PolynomialFeatures(degree=1, include_bias=False),
                StandardScaler(),
                LinearRegression(),
            ),
        }
        for name, model in models.items():
            scores = cross_validate(
                model, X_dev, y_dev, cv=cv_splits,
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                return_train_score=True, return_estimator=True, n_jobs=1,
                error_score="raise",
            )
            # Every prediction is made by a model that did not train on that row.
            predictions = np.full(len(y_dev), np.nan)
            large_target = np.zeros(len(y_dev), dtype=bool)
            for estimator, (fit_rows, validation_rows) in zip(scores["estimator"], cv_splits):
                predictions[validation_rows] = estimator.predict(X_dev.iloc[validation_rows])
                # Define the threshold using this fold's fitting targets only.
                threshold = np.quantile(np.abs(y_dev.iloc[fit_rows]), 0.90)
                large_target[validation_rows] = np.abs(y_dev.iloc[validation_rows]) >= threshold
            if not np.isfinite(predictions).all():
                raise ValueError("Cross-validation produced invalid predictions.")
            record = {
                "problem": problem,
                "model": name,
                "train_mse": float(-scores["train_mse"].mean()),
                "validation_mse": float(-scores["test_mse"].mean()),
                "validation_mse_std": float(scores["test_mse"].std(ddof=1)),
                "validation_r2": float(scores["test_r2"].mean()),
                "validation_r2_std": float(scores["test_r2"].std(ddof=1)),
                "pooled_validation_mse": float(mean_squared_error(y_dev, predictions)),
                "pooled_validation_r2": float(r2_score(y_dev, predictions)),
            }
            results.append(record)
            print(f"  {name}\n"
                  f"    Training MSE:   {record['train_mse']:.6f}\n"
                  f"    Validation MSE: {record['validation_mse']:.6f} "
                  f"(+/- {record['validation_mse_std']:.6f} fold SD)\n"
                  f"    Validation R2:  {record['validation_r2']:.6f} "
                  f"(+/- {record['validation_r2_std']:.6f} fold SD)", flush=True)

            segments = {
                "All development rows": np.ones(len(y_dev), dtype=bool),
                "Interior: no boundary features": boundary_counts == 0,
                "At least one boundary feature": boundary_counts >= 1,
                "Largest |y|: fold training 90th percentile": large_target,
            }
            segments.update({f"Exactly {count} boundary features": boundary_counts == count
                             for count in np.unique(boundary_counts) if count > 0})
            model_diagnostics = []
            for segment, mask in segments.items():
                if not mask.any():
                    continue
                observed = y_dev.to_numpy()[mask]
                predicted = predictions[mask]
                diagnostic = {
                    "problem": problem, "model": name, "segment": segment,
                    "rows": int(mask.sum()),
                    "mse": float(mean_squared_error(observed, predicted)),
                    "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else np.nan,
                }
                model_diagnostics.append(diagnostic)
                diagnostics.append(diagnostic)
            print("    Errors from pooled out-of-fold predictions:")
            print(pd.DataFrame(model_diagnostics)[["segment", "rows", "mse", "r2"]]
                  .round(6).to_string(index=False))
            slug = "mean" if isinstance(model, DummyRegressor) else "degree1"
            pd.DataFrame({
                "row_position": development,
                "validation_fold": assignments.loc[development, "validation_fold"].to_numpy(),
                "boundary_feature_count": boundary_counts,
                "actual_y": y_dev.to_numpy(), "predicted_y": predictions,
                "residual": y_dev.to_numpy() - predictions,
                "large_target": large_target,
            }).to_csv(OUTPUT / f"{problem}_{slug}_validation_predictions.csv", index=False)

    pd.DataFrame(results).to_csv(OUTPUT / "baseline_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "baseline_diagnostics.csv", index=False)
    print(f"\nResults and fixed splits saved to {OUTPUT}")
    print("The holdout remains unscored. No test predictions have been generated.")


if __name__ == "__main__":
    main()
