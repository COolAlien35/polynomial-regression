"""Screen every feature subset near promising polynomial degrees.

This complements the full degree search; it is not an exhaustive joint search
over every feature subset, degree, and penalty. Reuses saved development folds.
"""

from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "feature_search"
CONFIG = {"var1": ([4, 5, 6], [1, 10, 100]),
          "var2": ([7, 9, 11], [0.1, 1, 10])}


def load_development(problem):
    data = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
    splits = pd.read_csv(ROOT / "outputs" / "baseline" / f"{problem}_splits.csv")
    selected = splits.loc[splits["partition"].eq("development")]
    rows = selected["row_position"].to_numpy()
    X = data.iloc[rows].drop(columns="y").reset_index(drop=True)
    y = data.iloc[rows]["y"].reset_index(drop=True)
    fold_ids = selected["validation_fold"].to_numpy()
    folds = [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
             for f in sorted(np.unique(fold_ids))]
    return X, y, folds


def evaluate(X, y, folds, features, degree, alpha):
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                          StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))
    scores = cross_validate(model, X[list(features)], y, cv=folds,
                            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                            return_train_score=True, error_score="raise", n_jobs=1)
    return {
        "features": ",".join(features), "degree": degree, "alpha": alpha,
        "train_mse": float(-scores["train_mse"].mean()),
        "validation_mse": float(-scores["test_mse"].mean()),
        "validation_mse_std": float(scores["test_mse"].std(ddof=1)),
        "validation_r2": float(scores["test_r2"].mean()),
        **{f"fold_{i}_mse": float(-value) for i, value in enumerate(scores["test_mse"])},
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem, (degrees, alphas) in CONFIG.items():
        X, y, folds = load_development(problem)
        path = OUTPUT / f"{problem}_results.csv"
        records = pd.read_csv(path).to_dict("records") if path.exists() else []
        completed = {(r["features"], int(r["degree"]), float(r["alpha"])) for r in records}
        subsets = [subset for size in range(1, X.shape[1] + 1)
                   for subset in combinations(X.columns, size)]
        print(f"{problem}: {len(subsets)} subsets, degrees {degrees}, penalties {alphas}", flush=True)
        start = perf_counter()
        for number, subset in enumerate(subsets, 1):
            for degree in degrees:
                for alpha in alphas:
                    if (",".join(subset), degree, float(alpha)) not in completed:
                        records.append({"problem": problem,
                                        **evaluate(X, y, folds, subset, degree, alpha)})
            pd.DataFrame(records).to_csv(path, index=False)
            if number % 10 == 0 or number == len(subsets):
                best = min(records, key=lambda r: r["validation_mse"])
                print(f"  {number}/{len(subsets)} subsets completed; "
                      f"lowest MSE={best['validation_mse']:.6f}; "
                      f"elapsed={perf_counter() - start:.1f}s", flush=True)
        table = pd.DataFrame(records).sort_values("validation_mse")
        print("Best 10 candidates:")
        print(table.drop(columns=[c for c in table if c.startswith("fold_")])
              .head(10).round(6).to_string(index=False), flush=True)
        best_by_subset = table.drop_duplicates("features")
        best_by_subset.to_csv(OUTPUT / f"{problem}_best_by_subset.csv", index=False)
    print("Holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
