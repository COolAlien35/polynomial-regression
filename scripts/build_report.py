"""Build the five-page assignment report from the recorded analysis outputs."""

import json
import os
from pathlib import Path

from project_paths import ROOT
CACHE = ROOT / "outputs" / ".matplotlib_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


W, H = 8.27, 11.69
LEFT, WIDTH = 0.66, 6.95
INK, ACCENT, MUTED = "#142636", "#087f8c", "#526575"
PREVIEWS = ROOT / "outputs" / "report"
PDF = ROOT / "submission" / "BT2024060_Report.pdf"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#93a4af", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def read(relative):
    return pd.read_csv(ROOT / "outputs" / relative)


def axis(fig, left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def text(fig, value, top, *, left=LEFT, width=WIDTH, size=10.2,
         color=INK, weight="normal", family="DejaVu Sans", leading=1.32):
    """Wrap text using measured font widths; return its bottom coordinate."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=family, size=size, weight=weight)
    lines = []
    for paragraph in value.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            pixels = renderer.get_text_width_height_descent(candidate, prop, False)[0]
            if current and pixels > width * fig.dpi:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    pitch = size * leading / 72
    for i, line in enumerate(lines):
        fig.text(left / W, (top - i * pitch) / H, line, ha="left", va="top",
                 fontsize=size, color=color, fontweight=weight, fontfamily=family)
    bottom = top - len(lines) * pitch
    if bottom < 0.48:
        raise ValueError(f"Text overflow: {value[:70]} ends at {bottom:.2f} inches.")
    return bottom


def section(fig, title, top):
    return text(fig, title, top, size=12.5, weight="bold", color=ACCENT)


def table(fig, columns, rows, top, height, widths=None, size=9):
    ax = axis(fig, LEFT, top - height, WIDTH, height)
    ax.axis("off")
    widget = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                      loc="center", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    widget.auto_set_font_size(False)
    widget.set_fontsize(size)
    for (row, _), cell in widget.get_celld().items():
        cell.set_edgecolor("#d8e2e8")
        cell.set_linewidth(0.45)
        cell.PAD = 0.07
        cell.set_text_props(color=INK)
        if row == 0:
            cell.set_facecolor("#e6f1f3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#f5f8fa")
    return ax


def page(number, title, subtitle=None):
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor="white")
    text(fig, "ML ASSIGNMENT 1  /  POLYNOMIAL REGRESSION", 11.18,
         size=8.7, color=MUTED, weight="bold")
    fig.add_artist(plt.Line2D([LEFT / W, (LEFT + WIDTH) / W], [11.00 / H] * 2,
                             transform=fig.transFigure, color=ACCENT, linewidth=1.2))
    text(fig, title, 10.82, size=20, weight="bold", leading=1.15)
    if subtitle:
        text(fig, subtitle, 10.37, size=9.7, color=MUTED)
    fig.add_artist(plt.Line2D([LEFT / W, (LEFT + WIDTH) / W], [0.43 / H] * 2,
                             transform=fig.transFigure, color="#d8e2e8", linewidth=0.6))
    fig.text(LEFT / W, 0.28 / H, "Pulkit Pandey  ·  BT2024060  ·  Assignment report", fontsize=8, color=MUTED)
    fig.text((LEFT + WIDTH) / W, 0.28 / H, f"{number} / 5", ha="right", fontsize=8, color=MUTED)
    return fig


def correlations(fig, problem, rect):
    data = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_train_{problem}.csv").drop(columns="y")
    corr = data.corr()
    ax = axis(fig, *rect)
    plot = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(data.columns)), data.columns)
    ax.set_yticks(range(len(data.columns)), data.columns)
    ax.set_title(f"{problem}: input correlation", pad=9, fontweight="bold")
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.7 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return plot


def first_page():
    fig = page(1, "Polynomial regression: BT2024060",
               "Pulkit Pandey · personalised geothermal datasets · model selection and validation")
    section(fig, "1. Objective and supplied data", 9.96)
    text(fig, "Predict the continuous target y for two separate problems using polynomial regression. "
         "var1 maps six plant settings to Net Power Score; var2 maps three spatial coordinates to "
         "Thermal Anomaly Score. Only this roll number's datasets were used.", 9.66)
    table(fig, ["Problem", "Training", "Test", "Inputs / target"], [
        ["var1", "1,000 labelled rows", "1,000 unlabelled rows", "x1–x6 / y"],
        ["var2", "1,000 labelled rows", "1,000 unlabelled rows", "x1–x3 / y"],
    ], 8.84, 0.72, [0.12, 0.26, 0.29, 0.33], size=9.1)
    text(fig, "Inputs were already cleaned and bounded in [−1, 1]. Numeric and finite-value checks "
         "found no missing values. No observations were discarded or targets clipped.", 7.90)
    plot = correlations(fig, "var1", (1.03, 4.76, 2.29, 2.29))
    correlations(fig, "var2", (4.48, 4.93, 1.96, 1.96))
    cb = fig.colorbar(plot, cax=axis(fig, 6.92, 4.93, 0.12, 1.96))
    cb.set_label("Pearson r", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    text(fig, "Figure 1. var1 has correlated groups (x1–x4 and x5–x6); var2 has weak pairwise "
         "linear correlations. Neither input correlation nor correlation with y alone establishes "
         "feature usefulness: nonlinear and interaction effects require validation.", 4.30,
         size=9.0, color=MUTED)
    for problem, left, width in [("var1", 0.99, 2.85), ("var2", 4.69, 2.55)]:
        ax = axis(fig, left, 2.19, width, 1.25)
        for label, suffix, offset, color in [("Train", "train", -0.18, "#4589ac"),
                                            ("Test", "test", 0.18, "#efad63")]:
            data = pd.read_csv(ROOT / "data" / "BT2024060" / f"BT2024060_{suffix}_{problem}.csv").drop(columns="y", errors="ignore")
            counts = data.abs().eq(1).sum(axis=1)
            fractions = counts.value_counts(normalize=True).reindex(range(data.shape[1] + 1), fill_value=0)
            ax.bar(fractions.index + offset, fractions * 100, width=0.35, color=color, label=label)
        ax.set_title(f"{problem}: boundary coverage", fontweight="bold", pad=8)
        ax.set(xlabel="Number of inputs exactly at ±1", ylabel="Rows (%)", ylim=(0, 50))
        ax.set_xticks(fractions.index)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    text(fig, "Figure 2. var1 test data are more boundary-heavy: 64.8% of test rows have at least "
         "three boundary inputs, versus 28.0% of all training rows. var2 input coverage is similar "
         "between train and test.", 1.78, size=9.0, color=MUTED)
    text(fig, "Repeated input rows were retained. var2 contains nine repeated training feature "
         "rows, with different y values at some identical coordinates. Identical inputs were "
         "kept together in every fitting/validation split to prevent group leakage.", 1.13, size=9.6)
    return fig


def selection_heatmaps(fig):
    ridge = read("ridge_comparison/ridge_results.csv")
    arrays = []
    items = []
    for method in ["Ridge", "Lasso"]:
        for problem in ["var1", "var2"]:
            if method == "Ridge":
                data = ridge[ridge.problem.eq(problem) & ridge.alpha.gt(0)].copy()
                data["eligible"] = True
            else:
                data = read(f"lasso_comparison/{problem}_results.csv")
                data["eligible"] = data.converged_folds.eq(5)
                if problem == "var2":
                    verification = read("lasso_verification/var2_fold_results.csv")
                    mask = data.degree.eq(11) & np.isclose(data.alpha, 0.001)
                    if verification.converged.all():
                        data.loc[mask, "validation_mse"] = verification.validation_mse.mean()
                        data.loc[mask, "eligible"] = True
            eligible = data[data.eligible]
            best = eligible.loc[eligible.validation_mse.idxmin()]
            alphas = sorted(data.alpha.unique())
            degrees = list(range(1, 11 if problem == "var1" else 21))
            values = data.pivot(index="alpha", columns="degree", values="validation_mse").reindex(index=alphas, columns=degrees)
            eligibility = data.pivot(index="alpha", columns="degree", values="eligible").reindex(index=alphas, columns=degrees).fillna(False)
            values = values.where(eligibility)
            arrays.extend(values.to_numpy()[np.isfinite(values.to_numpy())].tolist())
            items.append((method, problem, values, best))
    norm = LogNorm(vmin=min(arrays), vmax=max(arrays))
    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("#d7dde2")
    for i, (method, problem, values, best) in enumerate(items):
        left = 1.13 if i % 2 == 0 else 4.74
        bottom = 6.93 if i < 2 else 4.84
        ax = axis(fig, left, bottom, 2.67, 1.33)
        image = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                          interpolation="nearest")
        degrees = values.columns.to_numpy()
        tick_positions = np.arange(len(degrees))
        if len(degrees) > 10:
            tick_positions = np.array([0, 3, 6, 9, 12, 15, 19])
        ax.set_xticks(tick_positions, degrees[tick_positions])
        ax.set_yticks(np.arange(len(values)), [f"{alpha:.3g}" for alpha in values.index])
        ax.tick_params(labelsize=7.8, pad=2)
        ax.set_xlabel("Total polynomial degree", fontsize=8.5, labelpad=3)
        ax.set_ylabel("Penalty α", fontsize=8.5, labelpad=4)
        ax.set_title(f"{method} / {problem} · min MSE {best.validation_mse:.4f}",
                     fontsize=9.4, fontweight="bold", pad=8)
        x = list(values.columns).index(int(best.degree))
        y = np.flatnonzero(np.isclose(values.index, best.alpha))[0]
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                               edgecolor="#fa8a35", linewidth=1.6))
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(image, cax=axis(fig, 1.14, 4.27, 6.22, 0.12), orientation="horizontal")
    cb.set_ticks([0.3, 1, 3, 10, 30], labels=["0.3", "1", "3", "10", "30"])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Mean validation MSE · logarithmic colour scale · lower is better", fontsize=8.5, labelpad=4)


def second_page():
    fig = page(2, "Validation design and model selection")
    text(fig, "A fixed grouped split reserved 200 var1 and 202 var2 rows. Initial searches used "
         "five folds on the remaining 800 / 798 development rows. Polynomial expansion, feature "
         "scaling and fitting were applied within each fold; no validation targets entered fitting.", 10.25)
    fig.text(0.5, 9.31 / H,
             r"$\hat{y}=\beta_0+\sum_{1\leq a_1+\cdots+a_m\leq d}\beta_{\mathbf{a}}\prod_{j=1}^{m}x_j^{a_j}$",
             ha="center", va="center", fontsize=14)
    text(fig, "Total degree d includes cross-feature interactions. Expanded terms were standardised "
         "before Ridge (L2) or Lasso (L1) penalties; their α values are not directly comparable.", 8.98,
         size=9.6)
    selection_heatmaps(fig)
    text(fig, "Figure 3. Full degree ranges: var1 1–10; var2 1–20. Outlines identify the minimum on "
         "each plotted grid. Grey cells failed convergence in at least one fold. The var2 Lasso "
         "degree-11, α=0.001 cell includes its successful longer verification run.", 3.79,
         size=9.0, color=MUTED)
    section(fig, "Refinement beyond the base grids", 3.10)
    table(fig, ["Problem", "Ridge base grid", "Ridge refined", "Lasso converged"], [
        ["var1", "d=5, α=10\nMSE 0.5412", "d=5, α=30\nMSE 0.5012", "d=5, α=0.01\nMSE 0.3256"],
        ["var2", "d=9, α=1\nMSE 0.2867", "d=11, α=3\nMSE 0.2789", "d=11, α=0.001\nMSE 0.2847"],
    ], 2.77, 0.99, [0.13, 0.29, 0.29, 0.29], size=9.2)
    text(fig, "Table 1. These are mean MSE values on the original development folds. Fine searches "
         "between the base penalties explain why the final Ridge degree differs from its grid winner.",
         1.60, size=9.0, color=MUTED)
    text(fig, "Screening all 63 var1 and seven var2 feature subsets near leading degrees favoured "
         "all inputs. Six inputs at degree 10 produce 8,007 terms; regularisation and validation "
         "control complexity. The search was staged, not exhaustive.", 1.10,
         size=9.4)
    return fig


def model_card(fig, left, title, description):
    fig.add_artist(Rectangle((left / W, 8.74 / H), 3.31 / W, 1.16 / H,
                             transform=fig.transFigure, facecolor="#edf5f6", edgecolor="none"))
    text(fig, title, 9.76, left=left + 0.13, width=3.04, size=11.2, weight="bold", color=ACCENT)
    text(fig, description, 9.43, left=left + 0.13, width=3.04, size=9.5, leading=1.24)


def third_page():
    fig = page(3, "Final choices and robustness checks")
    text(fig, "All supplied features were retained. Final choices were fixed before scoring the "
         "reserved holdout; the submitted polynomials were subsequently refitted on all training rows.",
         10.28, size=10)
    model_card(fig, 0.66, "var1 · degree 5 polynomial",
               "50% Lasso (α=0.01) + 50% OLS refit of terms selected by Lasso (α=0.03). "
               "The equal average remains a degree-5 polynomial.")
    model_card(fig, 4.30, "var2 · degree 11 polynomial",
               "Ridge α=3 on all three inputs. The expansion contains 363 terms plus "
               "an intercept. All fitting steps stay inside validation folds.")
    section(fig, "Additional techniques and fresh-split confirmation", 8.45)
    text(fig, "255 configurations/control settings compared Elastic Net, Legendre/Chebyshev "
         "bases, sparse coefficient refits and boundary weighting. Alternative bases did not "
         "improve the leaders; var1 boundary weighting worsened errors. Six shortlisted candidates "
         "per problem were tested on five fresh shuffled 5-fold group splits (25 fits each).", 8.16,
         size=9.7)
    aliases = {"current_lasso": "Lasso", "sparse_refit": "Sparse refit", "elastic_net": "Elastic Net",
               "stronger_lasso": "Stronger Lasso", "boundary_weighted": "Boundary weights",
               "average_lasso_sparse": "Fixed average", "current_ridge": "Ridge d=11",
               "degree9_ridge": "Ridge d=9", "average_ridge_sparse": "Ridge + sparse",
               "average_ridge_degrees": "Ridge d=9 + d=11"}
    for problem, left, width in [("var1", 1.80, 1.92), ("var2", 5.50, 1.90)]:
        ax = axis(fig, left, 4.95, width, 1.65)
        summary = read(f"robustness/{problem}_summary.csv").sort_values("validation_mse")
        chosen = "average_lasso_sparse" if problem == "var1" else "current_ridge"
        for row, (_, item) in enumerate(summary.iterrows()):
            ax.errorbar(item.validation_mse, row, xerr=item.split_mse_std, fmt="o", capsize=3,
                        markersize=4.5, color=ACCENT if item.candidate == chosen else "#8295a1")
        ax.set_yticks(range(len(summary)), [aliases[c] for c in summary.candidate])
        ax.invert_yaxis()
        ax.set(xlabel="Repeated-validation MSE", title=f"{problem}: shortlist stability")
        ax.tick_params(axis="y", labelsize=8.2)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#e5ebef", linewidth=0.6)
    text(fig, "Figure 4. Points are mean MSE; bars are SD across five split repetitions, not confidence "
         "intervals. The var1 average improved all five repetitions (0.3377 → 0.3333). var2's sparse "
         "refit gain was inconsistent; Ridge was retained for its stability and lower worst-split MSE.",
         4.52, size=9.0, color=MUTED)
    section(fig, "Learning curves at fixed settings", 3.82)
    text(fig, "Each curve uses five validation folds and progressively more fitting rows. Every "
         "selected-model fit converged. Degree and penalties were chosen through validation.", 3.54, size=9.5)
    for problem, left, width in [("var1", 1.02, 2.86), ("var2", 4.66, 2.70)]:
        ax = axis(fig, left, 1.19, width, 1.59)
        values = read(f"learning_curves/{problem}_summary.csv")
        ax.plot(values.training_rows, values.train_mse, "o-", color="#8295a1", markersize=3,
                label="Training")
        ax.errorbar(values.training_rows, values.validation_mse, yerr=values.validation_mse_std,
                    fmt="o-", color=ACCENT, markersize=3, capsize=3, label="Validation")
        ax.set(xlabel="Fitting rows per fold", ylabel="MSE", title=problem)
        ax.set_xticks([256, 384, 512, 640])
        ax.legend(fontsize=8, frameon=False)
    text(fig, "Figure 5. More fitting data lowers validation error and narrows the training–validation "
         "gap. The curves support using all 1,000 rows for the final fit; they do not prove zero overfitting.",
         0.82, size=8.8, color=MUTED)
    return fig


def fourth_page():
    fig = page(4, "Reserved-holdout results and limitations")
    text(fig, "Selected configurations and training-file hashes were saved before assessment. "
         "The holdout was excluded from fitting and tuning, and its scores did not change the "
         "choices. Initial descriptive EDA covered all supplied training rows. Hidden test labels "
         "remain unavailable.", 10.27, size=9.9)
    metrics = read("final_validation/validation_summary.csv")
    table(fig, ["Problem", "CV MSE", "CV R²", "Holdout n", "Holdout MSE", "Holdout R²"], [
        [r.problem, f"{r.repeated_cv_mse:.4f}", f"{r.repeated_cv_r2:.4f}", str(int(r.holdout_rows)),
         f"{r.holdout_mse:.4f}", f"{r.holdout_r2:.4f}"] for _, r in metrics.iterrows()
    ], 9.31, 0.72, [0.13, 0.17, 0.17, 0.17, 0.18, 0.18], size=9.1)
    text(fig, "Table 2. CV metrics are averages over complete out-of-fold predictions from five "
         "repetitions. Holdout metrics use one fixed model fit on the development partition.", 8.42,
         size=9.0, color=MUTED)
    text(fig, "Approximate 95% group-bootstrap intervals (2,000 resamples):\n"
         "var1: MSE [0.2616, 0.3769], R² [0.9570, 0.9748].\n"
         "var2: MSE [0.2489, 0.6194], R² [0.9911, 0.9950].", 7.90, size=9.3)
    for problem, bottom in [("var1", 5.43), ("var2", 3.10)]:
        points = read(f"final_validation/{problem}_holdout_predictions.csv")
        count_max = 6 if problem == "var1" else 3
        ax = axis(fig, 1.02, bottom, 2.52, 1.54)
        plot = ax.scatter(points.actual_y, points.predicted_y, c=points.boundary_feature_count,
                          cmap="viridis", vmin=0, vmax=count_max, s=11, alpha=0.8, edgecolors="none")
        low = min(points.actual_y.min(), points.predicted_y.min())
        high = max(points.actual_y.max(), points.predicted_y.max())
        ax.plot([low, high], [low, high], "--", color=INK, linewidth=0.8)
        ax.set(xlabel="Actual y", ylabel="Predicted y", title=f"{problem}: predictions")
        cb = fig.colorbar(plot, cax=axis(fig, 3.82, bottom, 0.09, 1.54))
        cb.ax.tick_params(labelsize=7)
        cb.set_ticks([0, count_max])
        ax = axis(fig, 4.83, bottom, 2.54, 1.54)
        ax.scatter(points.predicted_y, points.residual, c=points.boundary_feature_count,
                   cmap="viridis", vmin=0, vmax=count_max, s=11, alpha=0.8, edgecolors="none")
        ax.axhline(0, color=INK, linestyle="--", linewidth=0.8)
        ax.set(xlabel="Predicted y", ylabel="Actual − predicted", title=f"{problem}: residuals")
    text(fig, "Figure 6. Colour indicates the number of inputs exactly at ±1. Mean holdout residuals "
         "are +0.0062 (var1) and −0.0403 (var2). var2's six full-corner observations have MSE 3.4145, "
         "versus 0.2341 for 77 interior rows; a repeated corner accounts for several large errors.",
         2.62, size=9.0, color=MUTED)
    section(fig, "Interpretation and practical limits", 1.94)
    text(fig, "The results support generalisation within the sampled domain, but cannot establish "
         "globally optimal parameters or zero overfitting/underfitting. var1's test boundary shift "
         "and var2's rare corner groups limit certainty at unobserved combinations. Repeated splits "
         "share observations; their SD measures stability. The holdout, not further tuning to its "
         "scores, provides the final assessment.", 1.64, size=9.5)
    return fig


def fifth_page():
    fig = page(5, "Rationale, reproduction and deliverables")
    section(fig, "Why these polynomial degrees and techniques?", 10.23)
    text(fig, "var1: degree-1 regression had development MSE 8.9374. Ordinary degree 5 fitted "
         "training data closely (MSE 0.0733) but generalised poorly (validation MSE 4.5667). "
         "Lasso α=0.01 reduced the original-fold validation MSE to 0.3256. The fixed average with "
         "a sparse OLS refit further improved fresh-split consistency while remaining degree 5.",
         9.90)
    text(fig, "var2: degree-1 MSE was 30.8325; degree 4 gave 3.0693. Higher-degree regularised "
         "models captured the nonlinear structure, and degree-11 Ridge α=3 remained consistent "
         "across fresh splits. Adding complexity beyond this region did not reliably improve "
         "validation. Degrees and penalties were justified jointly using validation, convergence "
         "checks and learning curves. Final selection considered both error and stability.",
         8.92)
    text(fig, "Within the three repeated-input groups in var2 development data, pooled target "
         "variance was about 0.2253 (six residual degrees of freedom). This suggests noise, but is "
         "too limited to establish a global noise floor. Weak individual feature correlations "
         "were not used to remove inputs.", 7.72, size=9.8)
    section(fig, "Final fitting and prediction format", 6.78)
    text(fig, "After assessment, the fixed models were trained on all 1,000 labelled rows for each "
         "problem. Predictions preserve test row order. Repeated inputs receive identical values; "
         "all predictions were checked to be finite. The sample format contains one y column "
         "and no index column.", 6.48)
    table(fig, ["Prediction file", "Rows", "Columns"], [
        ["BT2024060_pred_var1.csv", "1,000", "y only"],
        ["BT2024060_pred_var2.csv", "1,000", "y only"],
    ], 5.48, 0.70, [0.66, 0.17, 0.17], size=9.4)
    section(fig, "Code and reproducibility", 4.48)
    text(fig, "Training, inference and analysis code are organised under scripts/ in the repository below. "
         "requirements.txt pins the Python packages. Saved configurations, original row splits, "
         "metrics and diagnostic figures are available under outputs/.", 4.18, size=9.9)
    fig.add_artist(Rectangle((LEFT / W, 2.83 / H), WIDTH / W, 0.62 / H,
                             transform=fig.transFigure, facecolor="#f0f4f7", edgecolor="none"))
    text(fig, "python -m pip install -r requirements.txt\n"
         "python scripts/final_model_check.py\n"
         "python scripts/predict_submission.py", 3.36, left=LEFT + 0.13, width=WIDTH - 0.26,
         size=9, family="DejaVu Sans Mono", leading=1.32)
    text(fig, "final_model_check.py refits the frozen choices and reuses the saved holdout "
         "assessment. predict_submission.py reproduces both CSVs from exported coefficients "
         "without retraining. JSON/CSV coefficient exports reproduce pipeline predictions to "
         "approximately 10⁻¹⁴, including the combined var1 polynomial.", 2.61, size=9.7)
    url = "https://github.com/COolAlien35/polynomial-regression"
    text(fig, "GitHub repository", 1.61, size=10.0, weight="bold", color=ACCENT)
    fig.text(LEFT / W, 1.31 / H, url, fontsize=9.3, color=ACCENT, va="top", url=url)
    text(fig, "Submission artifacts in submission/: this five-page PDF and the two prediction CSVs; "
         "also submit the "
         "repository containing training and inference code. Reported scores are internal "
         "validation and reserved-holdout scores; the assignment's hidden-test scores are unknown.",
         1.05, size=9.3)
    return fig


def main():
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "ML Assignment 1 — Polynomial Regression — BT2024060",
        "Author": "Pulkit Pandey (BT2024060)",
        "Subject": "Approach, polynomial selection, robustness, evaluation and reproduction",
        "Keywords": "polynomial regression, cross-validation, Ridge, Lasso, BT2024060",
    }
    builders = [first_page, second_page, third_page, fourth_page, fifth_page]
    temporary = PREVIEWS / "report_build.pdf"
    with PdfPages(temporary, metadata=metadata) as document:
        for number, builder in enumerate(builders, 1):
            fig = builder()
            document.savefig(fig)
            fig.savefig(PREVIEWS / f"page_{number}.png", dpi=150)
            plt.close(fig)
            print(f"Built page {number}/5", flush=True)
    temporary.replace(PDF)
    print(f"Saved {PDF} ({PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
