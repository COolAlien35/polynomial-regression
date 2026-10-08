"""Inspect boundaries, repeated inputs, and nonlinear patterns.

Run: python scripts/inspect_patterns.py
These exploratory summaries do not select the final model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "patterns"


def inspect(problem: str) -> None:
    train = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
    test = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_test_{problem}.csv")
    features = list(test.columns)
    print(f"\n{'=' * 60}\n{problem.upper()}\n{'=' * 60}")

    boundaries = []
    for feature in features:
        boundaries.append({
            "feature": feature,
            "train_at_minus1_pct": 100 * train[feature].eq(-1).mean(),
            "test_at_minus1_pct": 100 * test[feature].eq(-1).mean(),
            "train_at_plus1_pct": 100 * train[feature].eq(1).mean(),
            "test_at_plus1_pct": 100 * test[feature].eq(1).mean(),
            "train_unique_values": train[feature].nunique(),
        })
    boundaries = pd.DataFrame(boundaries).set_index("feature")
    boundaries.to_csv(OUTPUT / f"{problem}_boundaries.csv")
    print("\nExact boundary percentages:")
    print(boundaries.round(2).to_string())

    corr = train[features].corr()
    corr.to_csv(OUTPUT / f"{problem}_feature_correlations.csv")
    print("\nPearson correlations between input features:")
    print(corr.round(3).to_string())

    repeated = train.groupby(features, dropna=False)["y"].agg(
        count="size", mean="mean", std="std", minimum="min", maximum="max"
    )
    repeated = repeated[repeated["count"] > 1].copy()
    repeated["target_range"] = repeated["maximum"] - repeated["minimum"]
    repeated.to_csv(OUTPUT / f"{problem}_repeated_inputs.csv")
    print(f"\nRepeated input groups: {len(repeated)}")
    if not repeated.empty:
        print(repeated.round(4).to_string())
        print("Different y values at identical inputs suggest noise or unobserved factors;")
        print("they do not by themselves prove a data error.")

    boundary_count = train[features].abs().eq(1).sum(axis=1)
    boundary_target = train.assign(boundary_features=boundary_count).groupby(
        "boundary_features"
    )["y"].agg(["count", "mean", "std", "min", "max"])
    boundary_target.to_csv(OUTPUT / f"{problem}_target_by_boundary_count.csv")
    print("\nTarget summary by number of features exactly at a boundary:")
    print(boundary_target.round(4).to_string())

    nrows = (len(features) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 4 * nrows), squeeze=False)
    binned_tables = []
    for ax, feature in zip(axes.flat, features):
        # Boundary points get separate groups; interior observations use fixed bins.
        interior = train[feature].abs().lt(1)
        edges = np.linspace(-1, 1, 11)
        selected = train.loc[interior, [feature, "y"]].copy()
        selected["bin"] = pd.cut(selected[feature], bins=edges, include_lowest=True)
        table = selected.groupby("bin", observed=True).agg(
            x_mean=(feature, "mean"), count=("y", "size"),
            y_mean=("y", "mean"), y_std=("y", "std"),
        ).reset_index()
        for endpoint in [-1, 1]:
            values = train.loc[train[feature].eq(endpoint), "y"]
            if not values.empty:
                table = pd.concat([table, pd.DataFrame([{
                    "bin": f"exactly {endpoint}", "x_mean": endpoint,
                    "count": len(values), "y_mean": values.mean(), "y_std": values.std(),
                }])], ignore_index=True)
        table = table.sort_values("x_mean")
        table["y_standard_error"] = table["y_std"] / np.sqrt(table["count"])
        table.insert(0, "feature", feature)
        binned_tables.append(table)
        ax.scatter(train[feature], train["y"], s=7, alpha=0.12)
        ax.errorbar(table["x_mean"], table["y_mean"],
                    yerr=table["y_standard_error"], color="darkred", fmt="o-",
                    capsize=3, label="Bin mean +/- 1 standard error")
        ax.set(xlabel=feature, ylabel="y", title=f"{feature}: average target pattern")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    fig.suptitle(f"{problem}: marginal patterns (other features vary)")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{problem}_binned_target.png", dpi=160)
    plt.close(fig)
    pd.concat(binned_tables, ignore_index=True).to_csv(
        OUTPUT / f"{problem}_binned_target.csv", index=False
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem in ["var1", "var2"]:
        inspect(problem)
    print(f"\nSaved tables and plots to {OUTPUT}")
    print("Binned patterns alone cannot identify polynomial degree or rule out interactions.")


if __name__ == "__main__":
    main()
