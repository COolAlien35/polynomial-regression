"""Refine penalties near the best converged Lasso degree with tighter tolerance."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from search_features import load_development


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "lasso_refinement"
TOL = 1e-7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["var1", "var2", "both"], default="both")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem, limit in [("var1", 10), ("var2", 20)]:
        if args.problem != "both" and args.problem != problem:
            continue
        initial = pd.read_csv(ROOT / "outputs" / "lasso_comparison" / f"{problem}_results.csv")
        eligible = initial[initial["converged_folds"] == 5]
        # An unconverged preliminary fit can identify a promising region, but
        # it must converge in this stricter rerun before it is eligible.
        best = initial.loc[initial["validation_mse"].idxmin()]
        center = int(best["degree"])
        alphas = np.sort(float(best["alpha"]) * np.array([0.3, 0.5, 0.7, 1, 1.5, 2, 3]))[::-1]
        X, y, folds = load_development(problem)
        boundaries = X.abs().eq(1).sum(axis=1).to_numpy()
        records, fold_records = [], []
        for degree in range(max(1, center - 1), min(limit, center + 1) + 1):
            print(f"{problem}: strict Lasso refinement, degree {degree}", flush=True)
            expanded = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
            predictions = np.full((len(y), len(alphas)), np.nan)
            for fold, (fitting, validation) in enumerate(folds):
                scaler = StandardScaler()
                fit_X = np.asfortranarray(scaler.fit_transform(expanded[fitting]))
                val_X = scaler.transform(expanded[validation])
                fit_y = y.iloc[fitting].to_numpy()
                intercept = fit_y.mean()
                centered = fit_y - intercept
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    _, coefficients, gaps, iterations = lasso_path(
                        fit_X, centered, alphas=alphas, tol=TOL,
                        max_iter=100000, return_n_iter=True,
                    )
                predictions[validation] = val_X @ coefficients + intercept
                for i, alpha in enumerate(alphas):
                    fold_records.append({
                        "problem": problem, "degree": degree, "alpha": float(alpha), "fold": fold,
                        "train_mse": mean_squared_error(fit_y, fit_X @ coefficients[:, i] + intercept),
                        "validation_mse": mean_squared_error(y.iloc[validation], predictions[validation, i]),
                        "validation_r2": r2_score(y.iloc[validation], predictions[validation, i]),
                        "nonzero_terms": int(np.count_nonzero(coefficients[:, i])),
                        "iterations": int(iterations[i]), "dual_gap": float(gaps[i]),
                        "converged": bool(gaps[i] <= TOL * np.mean(centered ** 2)),
                    })
            for i, alpha in enumerate(alphas):
                table = pd.DataFrame([r for r in fold_records
                                      if r["degree"] == degree and r["alpha"] == float(alpha)])
                record = {
                    "problem": problem, "degree": degree, "alpha": float(alpha),
                    "polynomial_terms": expanded.shape[1],
                    "train_mse": table["train_mse"].mean(),
                    "validation_mse": table["validation_mse"].mean(),
                    "validation_mse_std": table["validation_mse"].std(ddof=1),
                    "validation_r2": table["validation_r2"].mean(),
                    "mean_nonzero_terms": table["nonzero_terms"].mean(),
                    "converged_folds": int(table["converged"].sum()),
                    "interior_mse": mean_squared_error(y.to_numpy()[boundaries == 0], predictions[boundaries == 0, i]),
                    "boundary_mse": mean_squared_error(y.to_numpy()[boundaries > 0], predictions[boundaries > 0, i]),
                }
                records.append(record)
                pd.DataFrame({"development_row": np.arange(len(y)), "actual_y": y,
                              "predicted_y": predictions[:, i], "boundary_count": boundaries}).to_csv(
                    OUTPUT / f"{problem}_degree{degree}_alpha{alpha:g}_predictions.csv", index=False)
            pd.DataFrame(records).to_csv(OUTPUT / f"{problem}_results.csv", index=False)
            pd.DataFrame(fold_records).to_csv(OUTPUT / f"{problem}_fold_results.csv", index=False)
        table = pd.DataFrame(records)
        print(table[table["converged_folds"] == 5].sort_values("validation_mse")
              .head(10).round(6).to_string(index=False), flush=True)
    print("Holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
