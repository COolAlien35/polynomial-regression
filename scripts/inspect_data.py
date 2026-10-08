"""Inspect the assignment datasets without training any models.

Run from your activated environment: python scripts/inspect_data.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without opening desktop windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from project_paths import ROOT
DATA_DIR = ROOT / "data" / "BT2024060"
OUTPUT_DIR = ROOT / "outputs" / "inspection"


def inspect_problem(problem: str) -> None:
    train = pd.read_csv(DATA_DIR / f"BT2024060_train_{problem}.csv")
    test = pd.read_csv(DATA_DIR / f"BT2024060_test_{problem}.csv")
    features = [column for column in train.columns if column != "y"]

    if "y" not in train.columns or list(test.columns) != features:
        raise ValueError(f"Unexpected columns for {problem}.")
    for name, frame in [("train", train), ("test", test)]:
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"{problem} {name} contains nonnumeric columns.")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{problem} {name} contains missing or infinite values.")

    print(f"\n{'=' * 60}\n{problem.upper()}\n{'=' * 60}")
    print(f"Training shape: {train.shape}; test shape: {test.shape}")
    print(f"Features: {features}; target: y")
    print("All values are numeric and finite.")
    print(f"Duplicate full training rows: {train.duplicated().sum()}")
    print(f"Duplicate training feature rows: {train[features].duplicated().sum()}")
    print(f"Duplicate test feature rows: {test.duplicated().sum()}")

    train_keys = pd.MultiIndex.from_frame(train[features])
    test_keys = pd.MultiIndex.from_frame(test[features])
    print(f"Test rows with an exact feature match in training: {test_keys.isin(train_keys).sum()}")

    summary = train.describe().T
    summary.to_csv(OUTPUT_DIR / f"{problem}_train_summary.csv")
    print("\nTraining summary:")
    print(summary.round(4).to_string())

    ranges = pd.DataFrame({
        "train_min": train[features].min(),
        "train_max": train[features].max(),
        "test_min": test.min(),
        "test_max": test.max(),
    })
    ranges["test_rows_outside_train_range"] = [
        int(((test[f] < train[f].min()) | (test[f] > train[f].max())).sum())
        for f in features
    ]
    ranges.to_csv(OUTPUT_DIR / f"{problem}_feature_ranges.csv")
    print("\nFeature ranges:")
    print(ranges.round(4).to_string())

    correlations = pd.DataFrame({
        "pearson_with_y": train[features].corrwith(train["y"]),
        "spearman_with_y": train[features].corrwith(train["y"], method="spearman"),
    })
    correlations.to_csv(OUTPUT_DIR / f"{problem}_target_correlations.csv")
    print("\nFeature correlations with y:")
    print(correlations.round(4).to_string())
    print("Small correlations do not rule out polynomial effects or interactions.")

    rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        ax.scatter(train[feature], train["y"], s=10, alpha=0.35)
        ax.set(xlabel=feature, ylabel="y", title=f"{feature} versus y")
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    fig.suptitle(f"{problem}: training feature relationships")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{problem}_feature_vs_target.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        bins = np.linspace(min(train[feature].min(), test[feature].min()),
                           max(train[feature].max(), test[feature].max()), 26)
        ax.hist(train[feature], bins=bins, alpha=0.5, label="Train")
        ax.hist(test[feature], bins=bins, alpha=0.5, label="Test")
        ax.set(xlabel=feature, ylabel="Count", title=f"{feature} distribution")
        ax.legend()
    for ax in list(axes.flat)[len(features):]:
        ax.set_visible(False)
    fig.suptitle(f"{problem}: train and test feature distributions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{problem}_feature_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train["y"], bins=30, edgecolor="white")
    ax.set(xlabel="y", ylabel="Count", title=f"{problem}: training target distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{problem}_target_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for problem in ["var1", "var2"]:
        inspect_problem(problem)
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    print(f"\nSample submission: shape={sample.shape}, columns={sample.columns.tolist()}")
    print(f"\nSaved summaries and plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
