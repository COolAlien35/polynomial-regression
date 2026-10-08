"""Compare polynomial degrees 1--4 with every supplied feature.

Run after baseline.py: python scripts/compare_degrees.py
Example: python scripts/compare_degrees.py --problem var2 --min-degree 5 --max-degree 8
Uses the saved development folds; never scores the reserved holdout.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "degree_comparison"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["var1", "var2", "both"], default="both")
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--max-degree", type=int, default=4)
    args = parser.parse_args()
    if args.min_degree < 1 or args.max_degree < args.min_degree:
        parser.error("Degrees must satisfy 1 <= min-degree <= max-degree.")
    problems = ["var1", "var2"] if args.problem == "both" else [args.problem]
    for problem in problems:
        limit = 10 if problem == "var1" else 20
        if args.max_degree > limit:
            parser.error(f"{problem} has a maximum degree of {limit}.")
    output = OUTPUT
    if (args.problem, args.min_degree, args.max_degree) != ("both", 1, 4):
        output = OUTPUT / f"{args.problem}_degrees_{args.min_degree}_to_{args.max_degree}"
    output.mkdir(parents=True, exist_ok=True)
    results, diagnostics = [], []
    for problem in problems:
        data = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
        splits_path = ROOT / "outputs" / "baseline" / f"{problem}_splits.csv"
        if not splits_path.exists():
            raise FileNotFoundError("Run baseline.py first to create the fixed folds.")
        assignments = pd.read_csv(splits_path)
        if len(assignments) != len(data):
            raise ValueError("Saved splits do not match the dataset length.")
        development = assignments.loc[assignments["partition"].eq("development")]
        rows = development["row_position"].to_numpy()
        X = data.iloc[rows].drop(columns="y").reset_index(drop=True)
        y = data.iloc[rows]["y"].reset_index(drop=True)
        fold_ids = development["validation_fold"].to_numpy()
        folds = [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
                 for f in sorted(np.unique(fold_ids))]
        boundary_counts = X.abs().eq(1).sum(axis=1).to_numpy()
        large_target = np.zeros(len(y), dtype=bool)
        for fit_rows, validation_rows in folds:
            threshold = np.quantile(np.abs(y.iloc[fit_rows]), 0.90)
            large_target[validation_rows] = np.abs(y.iloc[validation_rows]) >= threshold

        print(f"\n{problem.upper()}: all {X.shape[1]} features, "
              f"{len(y)} development rows, saved 5-fold splits", flush=True)
        for degree in range(args.min_degree, args.max_degree + 1):
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                StandardScaler(),
                LinearRegression(),
            )
            print(f"  Evaluating degree {degree}...", flush=True)
            scores = cross_validate(
                model, X, y, cv=folds,
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                return_train_score=True, return_estimator=True,
                n_jobs=1, error_score="raise",
            )
            predictions = np.full(len(y), np.nan)
            for estimator, (_, validation_rows) in zip(scores["estimator"], folds):
                predictions[validation_rows] = estimator.predict(X.iloc[validation_rows])
            if not np.isfinite(predictions).all():
                raise ValueError(f"Invalid predictions for {problem}, degree {degree}.")
            terms = scores["estimator"][0].named_steps["polynomialfeatures"].n_output_features_
            record = {
                "problem": problem, "degree": degree, "polynomial_terms": terms,
                "train_mse": float(-scores["train_mse"].mean()),
                "validation_mse": float(-scores["test_mse"].mean()),
                "validation_mse_std": float(scores["test_mse"].std(ddof=1)),
                "validation_r2": float(scores["test_r2"].mean()),
                "pooled_validation_mse": float(mean_squared_error(y, predictions)),
                "pooled_validation_r2": float(r2_score(y, predictions)),
            }
            results.append(record)
            print(f"    Terms: {terms} plus intercept; train MSE: {record['train_mse']:.6f}\n"
                  f"    Validation MSE: {record['validation_mse']:.6f} "
                  f"(+/- {record['validation_mse_std']:.6f} fold SD); "
                  f"R2: {record['validation_r2']:.6f}", flush=True)

            segments = {
                "Interior": boundary_counts == 0,
                "At least one boundary": boundary_counts >= 1,
                "Largest absolute targets": large_target,
            }
            segments.update({f"Exactly {count} boundaries": boundary_counts == count
                             for count in np.unique(boundary_counts) if count > 0})
            local_diagnostics = []
            for segment, mask in segments.items():
                if mask.any():
                    item = {
                        "problem": problem, "degree": degree, "segment": segment,
                        "rows": int(mask.sum()),
                        "mse": float(mean_squared_error(y.to_numpy()[mask], predictions[mask])),
                    }
                    diagnostics.append(item)
                    local_diagnostics.append(item)
            print(pd.DataFrame(local_diagnostics)[["segment", "rows", "mse"]]
                  .round(6).to_string(index=False), flush=True)
            pd.DataFrame({
                "row_position": rows, "validation_fold": fold_ids,
                "boundary_feature_count": boundary_counts,
                "actual_y": y, "predicted_y": predictions,
                "residual": y.to_numpy() - predictions,
            }).to_csv(output / f"{problem}_degree{degree}_validation_predictions.csv", index=False)
            # Save progress after each completed candidate.
            pd.DataFrame(results).to_csv(output / "degree_results.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(output / "degree_diagnostics.csv", index=False)

        table = pd.DataFrame([r for r in results if r["problem"] == problem])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(table["degree"], table["train_mse"], "o-", label="Training MSE")
        ax.errorbar(table["degree"], table["validation_mse"],
                    yerr=table["validation_mse_std"], fmt="o-", capsize=3,
                    label="Validation MSE +/- 1 fold SD")
        ax.set(xlabel="Polynomial degree", ylabel="MSE", title=f"{problem}: all features")
        ax.set_xticks(table["degree"])
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{problem}_degree_comparison.png", dpi=160)
        plt.close(fig)

    print("\nComparison summary:")
    print(pd.DataFrame(results).round(6).to_string(index=False))
    print(f"\nSaved results to {output}")
    print("This degree comparison is not the final model selection.")
    print("The reserved holdout remains unscored.")


if __name__ == "__main__":
    main()
