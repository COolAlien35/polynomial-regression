"""Give a promising var2 Lasso candidate a larger convergence budget.

Uses the same folds, without evaluating the reserved holdout.
"""

from pathlib import Path

from project_paths import ROOT
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from search_features import load_development


def main():
    X, y, folds = load_development("var2")
    records = []
    for fold, (fitting, validation) in enumerate(folds):
        model = make_pipeline(
            PolynomialFeatures(11, include_bias=False), StandardScaler(),
            Lasso(alpha=0.001, max_iter=500000, tol=1e-5,
                  selection="random", random_state=42),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X.iloc[fitting], y.iloc[fitting])
        estimator = model.named_steps["lasso"]
        pred = model.predict(X.iloc[validation])
        threshold = 1e-5 * np.var(y.iloc[fitting].to_numpy())
        item = {
            "fold": fold, "degree": 11, "alpha": 0.001,
            "validation_mse": mean_squared_error(y.iloc[validation], pred),
            "validation_r2": r2_score(y.iloc[validation], pred),
            "iterations": estimator.n_iter_, "dual_gap": estimator.dual_gap_,
            "converged": bool(estimator.dual_gap_ <= threshold and not any(
                issubclass(w.category, ConvergenceWarning) for w in caught)),
        }
        records.append(item)
        print(item, flush=True)
    output = ROOT / "outputs" / "lasso_verification"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output / "var2_fold_results.csv", index=False)
    print(f"Mean MSE: {table.validation_mse.mean():.6f}; "
          f"mean R2: {table.validation_r2.mean():.6f}; "
          f"converged folds: {table.converged.sum()}/5", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
