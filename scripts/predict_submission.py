"""Reproduce submission CSVs from the exported polynomial coefficients.

Run final_model_check.py first, then: python scripts/predict_submission.py
"""

import json
from pathlib import Path

from project_paths import ROOT

import numpy as np
import pandas as pd


def predict(formula, X):
    powers = np.asarray(formula["powers"], dtype=int)
    values = X[formula["features"]].to_numpy()
    terms = np.ones((len(values), len(powers)))
    for feature in range(values.shape[1]):
        terms *= values[:, feature, None] ** powers[None, :, feature]
    return terms @ np.asarray(formula["coefficients"])


def main():
    root = ROOT
    (root / "submission").mkdir(parents=True, exist_ok=True)
    for problem in ["var1", "var2"]:
        formula = json.loads((root / "outputs" / "final_models" / f"{problem}_polynomial.json").read_text())
        test = pd.read_csv(root / "data" / "BT2024060" / f"BT2024060_test_{problem}.csv")
        codes = pd.MultiIndex.from_frame(test).factorize()[0]
        unique_inputs = test.drop_duplicates().reset_index(drop=True)
        predictions = predict(formula, unique_inputs)[codes]
        if len(predictions) != 1000 or not np.isfinite(predictions).all():
            raise ValueError("Invalid submission predictions.")
        path = root / "submission" / f"BT2024060_pred_{problem}.csv"
        pd.DataFrame({"y": predictions}).to_csv(path, index=False)
        print(f"Saved {path.name}: {len(predictions)} rows, one y column.")


if __name__ == "__main__":
    main()
