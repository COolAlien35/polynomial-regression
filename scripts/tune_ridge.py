"""Compare ordinary and Ridge polynomial regression on saved development folds.

Run: python scripts/tune_ridge.py
Full ranges, reusing completed candidates: python scripts/tune_ridge.py --full-range --resume
alpha=0 uses ordinary least squares. Positive alpha penalizes large coefficients.
The reserved holdout is never evaluated here.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "ridge_comparison"
DEGREES = {"var1": [3, 4, 5], "var2": [6, 7, 8, 9, 10]}
ALPHAS = [0, 0.001, 0.01, 0.1, 1, 10, 100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-range", action="store_true",
                        help="Search positive Ridge penalties at every allowed degree.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed candidates from the saved results.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results, fold_results, diagnostic_results = [], [], []
    if args.resume:
        for filename, records in [
            ("ridge_results.csv", results),
            ("ridge_fold_results.csv", fold_results),
            ("ridge_diagnostics.csv", diagnostic_results),
        ]:
            path = OUTPUT / filename
            if path.exists():
                records.extend(pd.read_csv(path).to_dict("records"))
    completed = {(r["problem"], int(r["degree"]), float(r["alpha"])) for r in results}
    degrees_by_problem = {"var1": list(range(1, 11)), "var2": list(range(1, 21))} if args.full_range else DEGREES
    # High-degree ordinary least squares has already shown instability. The full
    # range search uses positive penalties, retaining earlier OLS comparisons.
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000] if args.full_range else ALPHAS
    for problem, degrees in degrees_by_problem.items():
        data = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
        assignments = pd.read_csv(ROOT / "outputs" / "baseline" / f"{problem}_splits.csv")
        development = assignments.loc[assignments["partition"].eq("development")]
        rows = development["row_position"].to_numpy()
        X = data.iloc[rows].drop(columns="y").reset_index(drop=True)
        y = data.iloc[rows]["y"].reset_index(drop=True)
        fold_ids = development["validation_fold"].to_numpy()
        folds = [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
                 for f in sorted(np.unique(fold_ids))]
        boundary_counts = X.abs().eq(1).sum(axis=1).to_numpy()
        large_target = np.zeros(len(y), dtype=bool)
        for fitting, validation in folds:
            threshold = np.quantile(np.abs(y.iloc[fitting]), 0.90)
            large_target[validation] = np.abs(y.iloc[validation]) >= threshold

        print(f"\n{problem.upper()}: degrees {degrees}, all {X.shape[1]} features", flush=True)
        for degree in degrees:
            for alpha in alphas:
                if (problem, degree, float(alpha)) in completed:
                    continue
                model = make_pipeline(
                    PolynomialFeatures(degree=degree, include_bias=False),
                    StandardScaler(),
                    LinearRegression() if alpha == 0 else Ridge(alpha=alpha, solver="cholesky"),
                )
                print(f"  Degree {degree}, alpha={alpha:g} ...", end=" ", flush=True)
                started = perf_counter()
                scores = cross_validate(
                    model, X, y, cv=folds,
                    scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                    return_train_score=True, return_estimator=True,
                    error_score="raise", n_jobs=1,
                )
                predictions = np.full(len(y), np.nan)
                for fold, (estimator, (_, validation)) in enumerate(zip(scores["estimator"], folds)):
                    predictions[validation] = estimator.predict(X.iloc[validation])
                    fold_results.append({
                        "problem": problem, "degree": degree, "alpha": alpha,
                        "fold": fold, "validation_mse": -scores["test_mse"][fold],
                        "validation_r2": scores["test_r2"][fold],
                    })
                if not np.isfinite(predictions).all():
                    raise ValueError("Nonfinite validation predictions.")
                item = {
                    "problem": problem, "degree": degree, "alpha": alpha,
                    "polynomial_terms": scores["estimator"][0].named_steps[
                        "polynomialfeatures"].n_output_features_,
                    "train_mse": float(-scores["train_mse"].mean()),
                    "validation_mse": float(-scores["test_mse"].mean()),
                    "validation_mse_std": float(scores["test_mse"].std(ddof=1)),
                    "validation_r2": float(scores["test_r2"].mean()),
                    "pooled_validation_mse": float(mean_squared_error(y, predictions)),
                    "pooled_validation_r2": float(r2_score(y, predictions)),
                    "fit_seconds": perf_counter() - started,
                }
                results.append(item)
                print(f"train MSE={item['train_mse']:.6f}, "
                      f"validation MSE={item['validation_mse']:.6f}, "
                      f"R2={item['validation_r2']:.6f}, "
                      f"time={item['fit_seconds']:.1f}s", flush=True)
                segments = {
                    "Interior": boundary_counts == 0,
                    "At least one boundary": boundary_counts >= 1,
                    "Largest absolute targets": large_target,
                }
                segments.update({f"Exactly {count} boundaries": boundary_counts == count
                                 for count in np.unique(boundary_counts) if count > 0})
                for segment, mask in segments.items():
                    if mask.any():
                        diagnostic_results.append({
                            "problem": problem, "degree": degree, "alpha": alpha,
                            "segment": segment, "rows": int(mask.sum()),
                            "mse": float(mean_squared_error(y.to_numpy()[mask], predictions[mask])),
                        })
                pd.DataFrame({
                    "row_position": rows, "validation_fold": fold_ids,
                    "boundary_feature_count": boundary_counts,
                    "actual_y": y, "predicted_y": predictions,
                    "residual": y.to_numpy() - predictions,
                }).to_csv(OUTPUT / f"{problem}_degree{degree}_alpha{alpha:g}_predictions.csv", index=False)
                pd.DataFrame(results).to_csv(OUTPUT / "ridge_results.csv", index=False)
                pd.DataFrame(fold_results).to_csv(OUTPUT / "ridge_fold_results.csv", index=False)
                pd.DataFrame(diagnostic_results).to_csv(OUTPUT / "ridge_diagnostics.csv", index=False)

        table = pd.DataFrame([r for r in results if r["problem"] == problem])
        table = table.sort_values("validation_mse")
        print(f"\n{problem.upper()}: best 10 candidates by mean validation MSE")
        print(table.head(10).round(6).to_string(index=False), flush=True)
        best = table.iloc[0]
        diagnostic_table = pd.DataFrame(diagnostic_results)
        selected = diagnostic_table[
            diagnostic_table["problem"].eq(problem)
            & diagnostic_table["degree"].eq(best["degree"])
            & diagnostic_table["alpha"].eq(best["alpha"])
        ]
        print("\nDiagnostics for the lowest-MSE candidate:")
        print(selected[["segment", "rows", "mse"]].round(6).to_string(index=False), flush=True)
        best_per_degree = table.loc[table.groupby("degree")["validation_mse"].idxmin()].sort_values("degree")
        best_per_degree.to_csv(OUTPUT / f"{problem}_best_per_degree.csv", index=False)
        print("\nBest tested penalty at each degree:")
        print(best_per_degree[["degree", "alpha", "train_mse", "validation_mse", "validation_r2"]]
              .round(6).to_string(index=False), flush=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(best_per_degree["degree"], best_per_degree["train_mse"], "o-", label="Training")
        ax.plot(best_per_degree["degree"], best_per_degree["validation_mse"], "o-", label="Validation")
        ax.set(xlabel="Degree", ylabel="MSE", title=f"{problem}: lowest validation MSE per degree")
        ax.set_xticks(best_per_degree["degree"])
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{problem}_best_per_degree.png", dpi=160)
        plt.close(fig)

    print(f"\nResults saved to {OUTPUT}")
    print("Candidates are still being compared; the reserved holdout remains unscored.")


if __name__ == "__main__":
    main()
