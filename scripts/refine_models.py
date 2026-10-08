"""Refine degree and Ridge alpha using development data only."""

from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from search_features import evaluate, load_development


from project_paths import ROOT
OUTPUT = ROOT / "outputs" / "refinement"
CONFIG = {
    "var1": ([4, 5, 6, 7], [3, 5, 10, 20, 30, 50, 100, 200]),
    "var2": (list(range(7, 14)), [0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20]),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for problem, (degrees, alphas) in CONFIG.items():
        X, y, folds = load_development(problem)
        path = OUTPUT / f"{problem}_results.csv"
        records = pd.read_csv(path).to_dict("records") if path.exists() else []
        completed = {(int(r["degree"]), float(r["alpha"])) for r in records}
        for degree in degrees:
            for alpha in alphas:
                if (degree, float(alpha)) not in completed:
                    records.append({"problem": problem,
                                    **evaluate(X, y, folds, tuple(X.columns), degree, alpha)})
            pd.DataFrame(records).to_csv(path, index=False)
            print(f"{problem}: completed degree {degree}", flush=True)
        table = pd.DataFrame(records).sort_values("validation_mse")
        print(table.drop(columns=[c for c in table if c.startswith("fold_")])
              .head(10).round(6).to_string(index=False), flush=True)
    print("Refinement complete. Reserved holdout remains unscored.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
