"""Check learning curves for the proposed final polynomial models."""

import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from polynomial_models import make_model
from robustness_check import FINAL_CHOICES
from search_features import load_development


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "learning_curves"
CHOICES = FINAL_CHOICES


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for problem, candidate in CHOICES.items():
        models = json.loads((ROOT / "outputs" / "robustness" / f"{problem}_shortlist.json").read_text())
        config = models[candidate]
        X, y, _ = load_development(problem)
        groups = pd.MultiIndex.from_frame(X).factorize()[0]
        folds = GroupKFold(n_splits=5, shuffle=True, random_state=7).split(X, y, groups)
        for fold, (fitting, validation) in enumerate(folds):
            unique_groups = np.unique(groups[fitting])
            shuffled = np.random.default_rng(700 + fold).permutation(unique_groups)
            for fraction in [0.4, 0.6, 0.8, 1.0]:
                chosen_groups = shuffled[:max(1, int(np.ceil(len(shuffled) * fraction)))]
                subset = fitting[np.isin(groups[fitting], chosen_groups)]
                model = make_model(config)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X.iloc[subset], y.iloc[subset])
                records.append({
                    "problem": problem, "candidate": candidate, "fold": fold,
                    "fraction": fraction, "training_rows": len(subset),
                    "train_mse": mean_squared_error(y.iloc[subset], model.predict(X.iloc[subset])),
                    "validation_mse": mean_squared_error(y.iloc[validation], model.predict(X.iloc[validation])),
                    "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                })
        table = pd.DataFrame([r for r in records if r["problem"] == problem])
        table.to_csv(OUTPUT / f"{problem}_fold_results.csv", index=False)
        summary = table.groupby("fraction").agg(
            training_rows=("training_rows", "mean"), train_mse=("train_mse", "mean"),
            validation_mse=("validation_mse", "mean"),
            validation_mse_std=("validation_mse", "std"), all_converged=("converged", "all"),
        )
        summary.to_csv(OUTPUT / f"{problem}_summary.csv")
        print(f"\n{problem} learning curve:\n{summary.round(6).to_string()}", flush=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(summary.training_rows, summary.train_mse, "o-", label="Training MSE")
        ax.errorbar(summary.training_rows, summary.validation_mse,
                    yerr=summary.validation_mse_std, fmt="o-", capsize=3,
                    label="Validation MSE +/- fold SD")
        ax.set(xlabel="Training rows per fold", ylabel="MSE", title=f"{problem}: fixed-model learning curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{problem}_learning_curve.png", dpi=160)
        plt.close(fig)
    print("Reserved holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
