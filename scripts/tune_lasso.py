"""Search Lasso polynomial regression using the saved development folds.

Run: python scripts/tune_lasso.py
Searches degrees 1--10 for var1 and 1--20 for var2. Standardisation is
fitted separately inside each fold. Descending alpha paths reuse solutions.
Results flag convergence; unconverged candidates are not eligible winners.
"""

from pathlib import Path
from time import perf_counter
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
OUTPUT = ROOT / "outputs" / "lasso_comparison"
ALPHAS = np.logspace(0, -3, 7)
TOL = 1e-5
MAX_ITER = 10000


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem, maximum in [("var1", 10), ("var2", 20)]:
        X, y, folds = load_development(problem)
        path = OUTPUT / f"{problem}_results.csv"
        fold_path = OUTPUT / f"{problem}_fold_results.csv"
        results = pd.read_csv(path).to_dict("records") if path.exists() else []
        fold_results = pd.read_csv(fold_path).to_dict("records") if fold_path.exists() else []
        completed = {int(r["degree"]) for r in results}
        counts = X.abs().eq(1).sum(axis=1).to_numpy()
        for degree in range(1, maximum + 1):
            if degree in completed:
                continue
            print(f"{problem}: degree {degree}, fitting 7 Lasso penalties across 5 folds...", flush=True)
            started = perf_counter()
            # PolynomialFeatures learns no sample statistics, so a single
            # expansion is safe. StandardScaler is fitted on fitting rows only.
            expanded = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
            predictions = np.full((len(y), len(ALPHAS)), np.nan)
            per_alpha = [[] for _ in ALPHAS]
            large_target = np.zeros(len(y), dtype=bool)
            for fold, (fitting, validation) in enumerate(folds):
                scaler = StandardScaler()
                fit_X = np.asfortranarray(scaler.fit_transform(expanded[fitting]))
                val_X = scaler.transform(expanded[validation])
                fit_y = y.iloc[fitting].to_numpy()
                intercept = fit_y.mean()
                centered_y = fit_y - intercept
                large_target[validation] = np.abs(y.iloc[validation]) >= np.quantile(np.abs(fit_y), 0.9)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    alphas, coefficients, gaps, iterations = lasso_path(
                        fit_X, centered_y, alphas=ALPHAS, tol=TOL,
                        max_iter=MAX_ITER, return_n_iter=True,
                    )
                if not np.allclose(alphas, ALPHAS):
                    raise ValueError("Unexpected alpha order from Lasso path.")
                predictions[validation] = val_X @ coefficients + intercept
                train_predictions = fit_X @ coefficients + intercept
                threshold = TOL * np.mean(centered_y ** 2)
                for i, alpha in enumerate(ALPHAS):
                    item = {
                        "problem": problem, "degree": degree, "alpha": float(alpha),
                        "fold": fold, "train_mse": mean_squared_error(fit_y, train_predictions[:, i]),
                        "validation_mse": mean_squared_error(y.iloc[validation], predictions[validation, i]),
                        "validation_r2": r2_score(y.iloc[validation], predictions[validation, i]),
                        "nonzero_terms": int(np.count_nonzero(coefficients[:, i])),
                        "iterations": int(iterations[i]), "dual_gap": float(gaps[i]),
                        "gap_tolerance": float(threshold),
                        "converged": bool(np.isfinite(gaps[i]) and gaps[i] <= threshold),
                    }
                    per_alpha[i].append(item)
                    fold_results.append(item)
            if not np.isfinite(predictions).all():
                raise ValueError("Lasso produced nonfinite predictions.")
            for i, alpha in enumerate(ALPHAS):
                table = pd.DataFrame(per_alpha[i])
                mask = counts >= 1
                record = {
                    "problem": problem, "degree": degree, "alpha": float(alpha),
                    "polynomial_terms": expanded.shape[1],
                    "train_mse": table["train_mse"].mean(),
                    "validation_mse": table["validation_mse"].mean(),
                    "validation_mse_std": table["validation_mse"].std(ddof=1),
                    "validation_r2": table["validation_r2"].mean(),
                    "mean_nonzero_terms": table["nonzero_terms"].mean(),
                    "converged_folds": int(table["converged"].sum()),
                    "boundary_mse": mean_squared_error(y.to_numpy()[mask], predictions[mask, i]),
                    "interior_mse": mean_squared_error(y.to_numpy()[~mask], predictions[~mask, i]),
                    "large_target_mse": mean_squared_error(y.to_numpy()[large_target], predictions[large_target, i]),
                }
                results.append(record)
            pd.DataFrame(results).to_csv(path, index=False)
            pd.DataFrame(fold_results).to_csv(fold_path, index=False)
            np.savez_compressed(OUTPUT / f"{problem}_degree{degree}_predictions.npz",
                                predictions=predictions, alphas=ALPHAS)
            degree_table = pd.DataFrame([r for r in results if int(r["degree"]) == degree])
            eligible = degree_table[degree_table["converged_folds"] == len(folds)]
            if not eligible.empty:
                best = eligible.loc[eligible["validation_mse"].idxmin()]
                print(f"  Best converged alpha={best['alpha']:.6g}, "
                      f"MSE={best['validation_mse']:.6f}; "
                      f"{len(eligible)}/7 candidates converged in all folds; "
                      f"{perf_counter() - started:.1f}s", flush=True)
            else:
                print("  No candidate converged in all folds.", flush=True)
        table = pd.DataFrame(results)
        eligible = table[table["converged_folds"] == len(folds)].sort_values("validation_mse")
        print(f"\n{problem}: best fully converged Lasso candidates")
        print(eligible.head(10).round(6).to_string(index=False), flush=True)
    print("Holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
