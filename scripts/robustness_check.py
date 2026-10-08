"""Compare a small polynomial shortlist across repeated grouped validation.

Runs five fresh 5-fold splits on development data only. Duplicate inputs stay
in one fold. Candidate definitions are frozen before the repeated comparison.
Equal averages retain polynomial predictions and use no fitted mixing weights.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from polynomial_models import make_model
from search_features import load_development


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "robustness"
SEEDS = [7, 19, 43, 71, 101]
FINAL_CHOICES = {"var1": "average_lasso_sparse", "var2": "current_ridge"}


def shortlist(problem):
    if problem == "var1":
        control = dict(basis="monomial", method="lasso", degree=5, alpha=0.01)
        sparse = dict(basis="monomial", method="sparse_refit", degree=5,
                      selection_alpha=0.03, refit_alpha=0)
        table = pd.read_csv(ROOT / "outputs" / "improvements" / "var1_results.csv")
        weighted = table[table.config.str.contains("boundary_weighting")].sort_values("validation_mse").iloc[0]
        return {
            "current_lasso": control,
            "sparse_refit": sparse,
            "elastic_net": dict(basis="monomial", method="elastic_net", degree=5,
                                alpha=0.01, l1_ratio=0.75),
            "stronger_lasso": dict(basis="monomial", method="lasso", degree=5, alpha=0.015),
            "boundary_weighted": json.loads(weighted.config),
            "average_lasso_sparse": dict(method="average", members=[control, sparse]),
        }
    control = dict(basis="monomial", method="ridge", degree=11, alpha=3)
    sparse = dict(basis="monomial", method="sparse_refit", degree=11,
                  selection_alpha=0.001, refit_alpha=1)
    simpler = dict(basis="monomial", method="ridge", degree=9, alpha=1)
    return {
        "current_ridge": control,
        "sparse_refit": sparse,
        "elastic_net": dict(basis="monomial", method="elastic_net", degree=11,
                            alpha=0.001, l1_ratio=0.75),
        "degree9_ridge": simpler,
        "average_ridge_sparse": dict(method="average", members=[control, sparse]),
        "average_ridge_degrees": dict(method="average", members=[control, simpler]),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem in ["var1", "var2"]:
        X, y, _ = load_development(problem)
        groups = pd.MultiIndex.from_frame(X).factorize()[0]
        boundaries = X.abs().eq(1).sum(axis=1).to_numpy()
        models = shortlist(problem)
        configs_path = OUTPUT / f"{problem}_shortlist.json"
        # Preserve reviewable candidate definitions before validation begins.
        configs_path.write_text(json.dumps(models, indent=2) + "\n")
        repeat_records, fold_records = [], []
        all_predictions = {}
        for name, config in models.items():
            candidate_predictions = []
            for repeat, seed in enumerate(SEEDS):
                cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
                predictions = np.full(len(y), np.nan)
                training_mses = []
                converged = True
                for fold, (fitting, validation) in enumerate(cv.split(X, y, groups)):
                    if np.intersect1d(groups[fitting], groups[validation]).size:
                        raise ValueError("A repeated input group crossed folds.")
                    model = make_model(config)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(X.iloc[fitting], y.iloc[fitting])
                    fold_converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                    converged &= fold_converged
                    predictions[validation] = model.predict(X.iloc[validation])
                    training_mses.append(mean_squared_error(y.iloc[fitting], model.predict(X.iloc[fitting])))
                    fold_records.append({
                        "problem": problem, "candidate": name, "seed": seed, "fold": fold,
                        "rows": len(validation), "converged": fold_converged,
                        "validation_mse": mean_squared_error(y.iloc[validation], predictions[validation]),
                        "validation_r2": r2_score(y.iloc[validation], predictions[validation]),
                    })
                if not np.isfinite(predictions).all():
                    raise ValueError("Nonfinite cross-validation prediction.")
                item = {
                    "problem": problem, "candidate": name, "seed": seed,
                    "train_mse": np.mean(training_mses), "validation_mse": mean_squared_error(y, predictions),
                    "validation_r2": r2_score(y, predictions), "converged": converged,
                    "boundary_mse": mean_squared_error(y.to_numpy()[boundaries > 0], predictions[boundaries > 0]),
                    "interior_mse": mean_squared_error(y.to_numpy()[boundaries == 0], predictions[boundaries == 0]),
                    "heavy_boundary_mse": mean_squared_error(y.to_numpy()[boundaries >= 3], predictions[boundaries >= 3]),
                }
                repeat_records.append(item)
                candidate_predictions.append(predictions)
                print(f"{problem} {name}, seed {seed}: MSE={item['validation_mse']:.6f}, "
                      f"R2={item['validation_r2']:.6f}, converged={converged}", flush=True)
                pd.DataFrame(repeat_records).to_csv(OUTPUT / f"{problem}_repeat_results.csv", index=False)
                pd.DataFrame(fold_records).to_csv(OUTPUT / f"{problem}_fold_results.csv", index=False)
            all_predictions[name] = np.array(candidate_predictions)
            np.savez_compressed(OUTPUT / f"{problem}_predictions.npz", **all_predictions)

        table = pd.DataFrame(repeat_records)
        summary = table.groupby("candidate").agg(
            train_mse=("train_mse", "mean"), validation_mse=("validation_mse", "mean"),
            split_mse_std=("validation_mse", "std"), worst_split_mse=("validation_mse", "max"),
            validation_r2=("validation_r2", "mean"), boundary_mse=("boundary_mse", "mean"),
            interior_mse=("interior_mse", "mean"), heavy_boundary_mse=("heavy_boundary_mse", "mean"),
            all_converged=("converged", "all"),
        ).sort_values("validation_mse")
        summary.to_csv(OUTPUT / f"{problem}_summary.csv")
        print(f"\n{problem} repeated-validation summary:")
        print(summary.round(6).to_string(), flush=True)
    print("Reserved holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
