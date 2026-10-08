"""Screen additional polynomial models using the original development folds.

Compares Elastic Net, orthogonal bases, and selection followed by refitting.
No reserved holdout targets are used. Saves resumable results and predictions.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from polynomial_models import make_model
from search_features import load_development


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "improvements"


def configurations(problem):
    configs = []
    # Include the existing choices as controls on exactly the same folds.
    configs.append({"basis": "monomial", "method": "lasso", "degree": 5, "alpha": 0.01}
                   if problem == "var1" else
                   {"basis": "monomial", "method": "ridge", "degree": 11, "alpha": 3})
    ridge_degrees = [3, 4, 5, 6, 7] if problem == "var1" else [6, 7, 8, 9, 10, 11, 12]
    sparse_degrees = [4, 5, 6] if problem == "var1" else [7, 9, 11]
    sparse_alphas = [0.003, 0.01, 0.03] if problem == "var1" else [0.001, 0.003, 0.01]
    for basis in ["legendre", "chebyshev"]:
        for degree in ridge_degrees:
            for alpha in [0.01, 0.1, 1, 10, 100]:
                configs.append(dict(basis=basis, method="ridge", degree=degree, alpha=alpha))
        for degree in sparse_degrees:
            for alpha in sparse_alphas:
                configs.append(dict(basis=basis, method="lasso", degree=degree, alpha=alpha))
    for degree in sparse_degrees:
        for alpha in sparse_alphas:
            for ratio in [0.25, 0.5, 0.75]:
                configs.append(dict(basis="monomial", method="elastic_net", degree=degree,
                                    alpha=alpha, l1_ratio=ratio))
    for degree in ([5, 6] if problem == "var1" else [9, 11]):
        for alpha in sparse_alphas:
            for refit_alpha in [0, 1, 10]:
                configs.append(dict(basis="monomial", method="sparse_refit", degree=degree,
                                    selection_alpha=alpha, refit_alpha=refit_alpha))
    if problem == "var1":
        test = pd.read_csv(ROOT / "data" / "BT2024060" / "BT2024060_test_var1.csv")
        counts = test.abs().eq(1).sum(axis=1).to_numpy()
        proportions = (np.bincount(counts, minlength=7) / len(test)).tolist()
        for alpha in [0.007, 0.01, 0.015, 0.02]:
            configs.append(dict(basis="monomial", method="lasso", degree=5, alpha=alpha,
                                boundary_weighting=True, test_boundary_proportions=proportions))
        for alpha in [20, 30, 50]:
            configs.append(dict(basis="monomial", method="ridge", degree=5, alpha=alpha,
                                boundary_weighting=True, test_boundary_proportions=proportions))
    return configs


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def assess(config, X, y, folds):
    predicted = np.full(len(y), np.nan)
    train_errors, mse_values, r2_values, active_terms = [], [], [], []
    converged = True
    for fitting, validation in folds:
        model = make_model(config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X.iloc[fitting], y.iloc[fitting])
        converged &= not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        predicted[validation] = model.predict(X.iloc[validation])
        train_errors.append(mean_squared_error(y.iloc[fitting], model.predict(X.iloc[fitting])))
        mse_values.append(mean_squared_error(y.iloc[validation], predicted[validation]))
        r2_values.append(r2_score(y.iloc[validation], predicted[validation]))
        final = model.model_.steps[-1][1] if hasattr(model, "model_") else model.steps[-1][1]
        active_terms.append(len(final.support_) if hasattr(final, "support_")
                            else int(np.count_nonzero(final.coef_)))
        iterative = final.selector_ if hasattr(final, "selector_") else final
        if hasattr(iterative, "dual_gap_"):
            converged &= bool(iterative.dual_gap_ <= iterative.tol * np.var(y.iloc[fitting]))
    if not np.isfinite(predicted).all():
        raise ValueError("Invalid predictions.")
    boundaries = X.abs().eq(1).sum(axis=1).to_numpy()
    return {
        "train_mse": np.mean(train_errors), "validation_mse": np.mean(mse_values),
        "validation_mse_std": np.std(mse_values, ddof=1), "validation_r2": np.mean(r2_values),
        "converged": bool(converged), "mean_nonzero_terms": np.mean(active_terms),
        "boundary_mse": mean_squared_error(y.to_numpy()[boundaries > 0], predicted[boundaries > 0]),
        "interior_mse": mean_squared_error(y.to_numpy()[boundaries == 0], predicted[boundaries == 0]),
        "heavy_boundary_mse": mean_squared_error(y.to_numpy()[boundaries >= 3], predicted[boundaries >= 3]),
        **{f"fold_{i}_mse": value for i, value in enumerate(mse_values)},
    }, predicted


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem in ["var1", "var2"]:
        X, y, folds = load_development(problem)
        configs = configurations(problem)
        path = OUTPUT / f"{problem}_results.csv"
        records = pd.read_csv(path).to_dict("records") if path.exists() else []
        completed = {r["candidate_id"] for r in records}
        print(f"{problem}: screening {len(configs)} additional/control candidates", flush=True)
        for number, config in enumerate(configs, 1):
            key = identity(config)
            if key in completed:
                continue
            start = perf_counter()
            scores, prediction = assess(config, X, y, folds)
            records.append({"candidate_id": key, "problem": problem,
                            "config": json.dumps(config, sort_keys=True), **scores,
                            "fit_seconds": perf_counter() - start})
            np.save(OUTPUT / f"{problem}_{key}_predictions.npy", prediction)
            pd.DataFrame(records).to_csv(path, index=False)
            if number % 10 == 0 or number == len(configs):
                eligible = [r for r in records if r["converged"]]
                best = min(eligible, key=lambda r: r["validation_mse"])
                print(f"  {number}/{len(configs)}; best converged MSE={best['validation_mse']:.6f}; "
                      f"latest candidate took {perf_counter()-start:.1f}s", flush=True)
        table = pd.DataFrame(records)
        print(table[table.converged].sort_values("validation_mse")
              .drop(columns=[c for c in table if c.startswith("fold_")])
              .head(8).round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
