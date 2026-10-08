"""Freeze selected models, assess the reserved holdout, and refit for submission.

Run: python scripts/final_model_check.py
The holdout does not select models. Existing holdout results are reused on reruns.
Final coefficients are exported as a single polynomial for each problem.
"""

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from polynomial_models import make_model
from robustness_check import FINAL_CHOICES


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "final_validation"
MODELS = ROOT / "outputs" / "final_models"


def bootstrap_intervals(X, y, predictions):
    """Resample input groups, preserving duplicate observations together."""
    groups = pd.MultiIndex.from_frame(X).factorize()[0]
    statistics = pd.DataFrame({
        "group": groups, "count": 1, "y_sum": y, "y_square_sum": y ** 2,
        "sse": (y - predictions) ** 2,
    }).groupby("group").sum().to_numpy()
    rng = np.random.default_rng(2026)
    mse_values, r2_values = [], []
    for _ in range(2000):
        sampled = statistics[rng.integers(0, len(statistics), size=len(statistics))].sum(axis=0)
        count, y_sum, y_square_sum, sse = sampled
        mse_values.append(sse / count)
        variance_sum = y_square_sum - y_sum ** 2 / count
        r2_values.append(1 - sse / variance_sum)
    return {
        "mse_ci_low": float(np.quantile(mse_values, 0.025)),
        "mse_ci_high": float(np.quantile(mse_values, 0.975)),
        "r2_ci_low": float(np.quantile(r2_values, 0.025)),
        "r2_ci_high": float(np.quantile(r2_values, 0.975)),
    }


def fit_checked(config, X, y):
    model = make_model(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError("The selected model failed to converge.")
    return model


def export_polynomial(model, features):
    """Convert scaled coefficients and equal averages to raw-input coefficients."""
    members = model.models_ if hasattr(model, "models_") else [model]
    coefficients = {}
    intercept = 0.0
    for member in members:
        expansion, scaler, regressor = (step[1] for step in member.steps)
        if not isinstance(expansion, PolynomialFeatures):
            raise ValueError("Raw polynomial export requires the monomial basis.")
        if hasattr(regressor, "support_"):
            scaled = np.zeros(len(expansion.powers_))
            if len(regressor.support_):
                scaled[regressor.support_] = regressor.regressor_.coef_
                offset = regressor.regressor_.intercept_
            else:
                offset = regressor.mean_
        else:
            scaled = regressor.coef_
            offset = regressor.intercept_
        raw = scaled / scaler.scale_
        intercept += (float(offset) - np.dot(raw, scaler.mean_)) / len(members)
        for powers, coefficient in zip(expansion.powers_, raw):
            key = tuple(int(p) for p in powers)
            coefficients[key] = coefficients.get(key, 0.0) + float(coefficient) / len(members)
    coefficients[(0,) * len(features)] = intercept
    ordered = sorted(coefficients, key=lambda term: (sum(term), term))
    return {"features": list(features), "degree": max(sum(term) for term in ordered),
            "powers": [list(term) for term in ordered],
            "coefficients": [coefficients[term] for term in ordered]}


def predict_formula(formula, X):
    powers = np.asarray(formula["powers"], dtype=int)
    values = X[formula["features"]].to_numpy()
    terms = np.ones((len(values), len(powers)))
    for feature in range(values.shape[1]):
        terms *= values[:, feature, None] ** powers[None, :, feature]
    return terms @ np.asarray(formula["coefficients"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (ROOT / "submission").mkdir(parents=True, exist_ok=True)
    frozen_path = OUTPUT / "frozen_choices.json"
    choices = {}
    for problem, name in FINAL_CHOICES.items():
        candidates = json.loads((ROOT / "outputs" / "robustness" / f"{problem}_shortlist.json").read_text())
        train_path = ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv"
        choices[problem] = {"candidate": name, "config": candidates[name],
                            "training_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != choices:
            raise ValueError("Frozen choices or training data changed; refusing to reuse holdout results.")
    else:
        # Written before any reserved holdout score is computed.
        frozen_path.write_text(json.dumps(choices, indent=2) + "\n")

    for problem, choice in choices.items():
        train = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv")
        X, y = train.drop(columns="y"), train.y
        assignments = pd.read_csv(ROOT / "outputs" / "baseline" / f"{problem}_splits.csv")
        fitting = assignments.loc[assignments.partition.eq("development"), "row_position"].to_numpy()
        holdout = assignments.loc[assignments.partition.eq("holdout"), "row_position"].to_numpy()
        if set(pd.MultiIndex.from_frame(X.iloc[fitting])) & set(pd.MultiIndex.from_frame(X.iloc[holdout])):
            raise ValueError("Repeated inputs crossed the reserved holdout partition.")
        score_path = OUTPUT / f"{problem}_holdout_metrics.json"
        if score_path.exists():
            metrics = json.loads(score_path.read_text())
        else:
            model = fit_checked(choice["config"], X.iloc[fitting], y.iloc[fitting])
            predictions = model.predict(X.iloc[holdout])
            if not np.isfinite(predictions).all():
                raise ValueError("Nonfinite holdout predictions.")
            observed = y.iloc[holdout].to_numpy()
            boundaries = X.iloc[holdout].abs().eq(1).sum(axis=1).to_numpy()
            metrics = {
                "problem": problem, "candidate": choice["candidate"], "holdout_rows": len(holdout),
                "development_train_mse": mean_squared_error(y.iloc[fitting], model.predict(X.iloc[fitting])),
                "holdout_mse": mean_squared_error(observed, predictions),
                "holdout_r2": r2_score(observed, predictions),
                "mean_residual": float(np.mean(observed - predictions)),
                **bootstrap_intervals(X.iloc[holdout], observed, predictions),
            }
            errors = pd.DataFrame({"row_position": holdout, "actual_y": observed,
                                   "predicted_y": predictions, "residual": observed - predictions,
                                   "boundary_feature_count": boundaries})
            errors.to_csv(OUTPUT / f"{problem}_holdout_predictions.csv", index=False)
            errors["squared_error"] = errors.residual ** 2
            errors.groupby("boundary_feature_count").agg(
                rows=("squared_error", "size"), mse=("squared_error", "mean"),
                mean_residual=("residual", "mean"),
            ).to_csv(OUTPUT / f"{problem}_holdout_boundaries.csv")
            score_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\n{problem}: {json.dumps(metrics, indent=2)}", flush=True)

        # After assessment, recover all labelled observations for final fitting.
        final_model = fit_checked(choice["config"], X, y)
        test = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_test_{problem}.csv")
        # Evaluate each distinct input once so duplicate rows have bit-identical
        # predictions despite small BLAS rounding differences across row blocks.
        codes = pd.MultiIndex.from_frame(test).factorize()[0]
        unique_inputs = test.drop_duplicates().reset_index(drop=True)
        predictions = final_model.predict(unique_inputs)[codes]
        if len(predictions) != 1000 or not np.isfinite(predictions).all():
            raise ValueError("Invalid final predictions.")
        formula = export_polynomial(final_model, X.columns)
        reconstructed = predict_formula(formula, unique_inputs)[codes]
        difference = float(np.max(np.abs(predictions - reconstructed)))
        if not np.allclose(predictions, reconstructed, rtol=1e-9, atol=1e-9):
            raise ValueError("Exported polynomial does not reproduce model predictions.")
        (MODELS / f"{problem}_polynomial.json").write_text(json.dumps(formula, indent=2) + "\n")
        pd.DataFrame({**{feature + "_power": np.asarray(formula["powers"])[:, j]
                        for j, feature in enumerate(formula["features"])},
                      "coefficient": formula["coefficients"]}).to_csv(
                          MODELS / f"{problem}_coefficients.csv", index=False)
        joblib.dump(final_model, MODELS / f"{problem}_model.joblib")
        destination = ROOT / "submission" / f"BT2024060_pred_{problem}.csv"
        pd.DataFrame({"y": predictions}).to_csv(destination, index=False)
        print(f"Final fit: degree {formula['degree']}; all {len(train)} training rows; "
              f"prediction range [{predictions.min():.4f}, {predictions.max():.4f}]; "
              f"polynomial export error {difference:.2e}; saved {destination.name}", flush=True)
    print("Models were selected before holdout assessment; no hidden test labels were used.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
