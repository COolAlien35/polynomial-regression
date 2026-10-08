"""Audit saved submission artifacts without fitting or selecting any models.

Run: python scripts/check_deliverables.py
Native PDF page/text/visual checks are documented in docs/DELIVERABLE_AUDIT.md.
"""

import hashlib
from importlib.metadata import version
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from predict_submission import predict
from project_paths import ROOT
from robustness_check import FINAL_CHOICES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    require(sample.shape == (1000, 1) and list(sample) == ["y"], "Unexpected sample format")
    frozen = json.loads((ROOT / "outputs" / "final_validation" / "frozen_choices.json").read_text())
    audit = {
        "author": "Pulkit Pandey", "roll_number": "BT2024060",
        "repository": "https://github.com/COolAlien35/polynomial-regression",
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in
                         ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "threadpoolctl"]},
        "problems": {},
    }
    for problem, expected_degree, limit in [("var1", 5, 10), ("var2", 11, 20)]:
        train_path = ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(train_path.with_name(f"BT2024060_test_{problem}.csv"))
        X, y = train.drop(columns="y"), train.y
        choice = frozen[problem]
        require(hashlib.sha256(train_path.read_bytes()).hexdigest() == choice["training_sha256"],
                f"{problem}: original training data changed")
        require(choice["candidate"] == FINAL_CHOICES[problem], f"{problem}: final choice changed")
        require(list(X) == list(test), f"{problem}: train/test feature order differs")
        require(np.isfinite(train.to_numpy()).all() and np.isfinite(test.to_numpy()).all(),
                f"{problem}: nonfinite data")

        # Recompute split isolation from actual inputs, not from a saved group ID.
        splits = pd.read_csv(ROOT / "outputs" / "baseline" / f"{problem}_splits.csv")
        require(len(splits) == len(train) and np.array_equal(splits.row_position, np.arange(len(train))),
                f"{problem}: saved splits do not cover every training row once")
        groups = pd.MultiIndex.from_frame(X).factorize()[0]
        checked = splits.assign(actual_group=groups)
        require(checked.groupby("actual_group").partition.nunique().max() == 1,
                f"{problem}: identical inputs cross development/holdout")
        dev = checked[checked.partition.eq("development")]
        require(checked.partition.isin(["development", "holdout"]).all(), "Unknown partition")
        require(dev.validation_fold.isin(range(5)).all(), "Invalid development fold")
        require(dev.groupby("actual_group").validation_fold.nunique().max() == 1,
                f"{problem}: identical inputs cross original folds")

        final = ROOT / "outputs" / "final_validation"
        holdout = pd.read_csv(final / f"{problem}_holdout_predictions.csv")
        metrics = json.loads((final / f"{problem}_holdout_metrics.json").read_text())
        require(metrics["candidate"] == choice["candidate"], "Scores refer to another model")
        require(np.array_equal(holdout.row_position, checked.loc[checked.partition.eq("holdout"), "row_position"]),
                f"{problem}: holdout prediction rows differ from the reserved partition")
        require(np.allclose(holdout.actual_y, y.iloc[holdout.row_position], atol=1e-12, rtol=0),
                f"{problem}: saved holdout targets do not match original data")
        require(np.isclose(mean_squared_error(holdout.actual_y, holdout.predicted_y), metrics["holdout_mse"]),
                f"{problem}: holdout MSE does not reproduce")
        require(np.isclose(r2_score(holdout.actual_y, holdout.predicted_y), metrics["holdout_r2"]),
                f"{problem}: holdout R2 does not reproduce")

        robustness = ROOT / "outputs" / "robustness"
        summary = pd.read_csv(robustness / f"{problem}_summary.csv").set_index("candidate").loc[choice["candidate"]]
        folds = pd.read_csv(robustness / f"{problem}_fold_results.csv")
        selected_folds = folds[folds.candidate.eq(choice["candidate"])]
        require(len(selected_folds) == 25 and selected_folds.converged.all(),
                f"{problem}: selected repeated fits are incomplete or unconverged")
        with np.load(robustness / f"{problem}_predictions.npz") as saved:
            oof = saved[choice["candidate"]]
        dev_y = y.iloc[dev.row_position].to_numpy()
        require(oof.shape == (5, len(dev_y)) and np.isfinite(oof).all(), "Invalid repeated OOF predictions")
        require(np.isclose(np.mean([mean_squared_error(dev_y, p) for p in oof]), summary.validation_mse),
                f"{problem}: repeated CV MSE does not reproduce")
        require(np.isclose(np.mean([r2_score(dev_y, p) for p in oof]), summary.validation_r2),
                f"{problem}: repeated CV R2 does not reproduce")

        model_dir = ROOT / "outputs" / "final_models"
        formula = json.loads((model_dir / f"{problem}_polynomial.json").read_text())
        powers = np.asarray(formula["powers"])
        coefficients = np.asarray(formula["coefficients"])
        require(formula["features"] == list(X), f"{problem}: exported feature order differs")
        require(powers.ndim == 2 and powers.shape == (len(coefficients), len(X.columns)), "Invalid polynomial dimensions")
        require(np.issubdtype(powers.dtype, np.integer) and (powers >= 0).all(), "Invalid polynomial powers")
        require(np.isfinite(coefficients).all(), "Nonfinite coefficients")
        require(powers.sum(axis=1).max() == formula["degree"] == expected_degree <= limit,
                f"{problem}: total-degree constraint violated")
        path = ROOT / "submission" / f"BT2024060_pred_{problem}.csv"
        submission = pd.read_csv(path)
        require(submission.shape == sample.shape and list(submission) == list(sample), "Submission format mismatch")
        require(np.isfinite(submission.y).all(), "Nonfinite submitted predictions")
        codes = pd.MultiIndex.from_frame(test).factorize()[0]
        unique = test.drop_duplicates().reset_index(drop=True)
        reconstructed = predict(formula, unique)[codes]
        model = joblib.load(model_dir / f"{problem}_model.joblib")
        pipeline_predictions = model.predict(unique)[codes]
        require(np.allclose(submission.y, reconstructed, atol=1e-9, rtol=1e-9), "Formula does not reproduce submission row order/values")
        require(np.allclose(submission.y, pipeline_predictions, atol=1e-9, rtol=1e-9), "Pipeline does not reproduce submission")
        require(pd.DataFrame({"group": codes, "y": submission.y}).groupby("group").y.nunique().max() == 1,
                "Identical test inputs have different saved predictions")
        audit["problems"][problem] = {
            "passed": True, "rows": len(submission), "columns": list(submission),
            "degree": expected_degree, "allowed_degree": limit,
            "development_rows": len(dev), "holdout_rows": len(holdout),
            "repeated_cv_mse": float(summary.validation_mse),
            "repeated_cv_r2": float(summary.validation_r2),
            "holdout_mse": metrics["holdout_mse"], "holdout_r2": metrics["holdout_r2"],
            "formula_max_error": float(np.max(np.abs(submission.y - reconstructed))),
            "pipeline_max_error": float(np.max(np.abs(submission.y - pipeline_predictions))),
            "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{problem}: PASS — format, split isolation, convergence, metrics, degree and inference")
    pdf = ROOT / "submission" / "BT2024060_Report.pdf"
    require(pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-"), "Missing or invalid report PDF")
    audit["report"] = {"path": str(pdf.relative_to(ROOT)),
                       "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                       "native_pdf_review": "See docs/DELIVERABLE_AUDIT.md"}
    destination = ROOT / "outputs" / "deliverable_audit.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
