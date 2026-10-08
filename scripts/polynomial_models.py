"""Reusable polynomial bases and regularised regressors for the assignment."""

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class OrthogonalPolynomialFeatures(TransformerMixin, BaseEstimator):
    """Tensor-product Chebyshev/Legendre terms with total degree <= degree.

    These span the same polynomial space as ordinary monomials. Ridge/Lasso
    penalties act differently in each basis. Inputs must retain their supplied
    [-1, 1] coordinates; no data-dependent mapping is learned here.
    """

    def __init__(self, degree=3, basis="legendre"):
        self.degree = degree
        self.basis = basis

    def fit(self, X, y=None):
        values = check_array(X)
        self.n_features_in_ = values.shape[1]
        expansion = PolynomialFeatures(self.degree, include_bias=False).fit(values)
        self.powers_ = expansion.powers_
        self.n_output_features_ = len(self.powers_)
        if self.basis not in ["legendre", "chebyshev"]:
            raise ValueError("Unknown polynomial basis.")
        return self

    def transform(self, X):
        check_is_fitted(self)
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count changed.")
        vandermonde = legvander if self.basis == "legendre" else chebvander
        univariate = [vandermonde(values[:, j], self.degree)
                      for j in range(self.n_features_in_)]
        output = np.ones((len(values), self.n_output_features_))
        for j, basis_values in enumerate(univariate):
            output *= basis_values[:, self.powers_[:, j]]
        return output


class SparseRefit(RegressorMixin, BaseEstimator):
    """Select terms with Lasso, then refit their coefficients with Ridge/OLS.

    Selection occurs only on the fitting part of each validation fold.
    """

    def __init__(self, selection_alpha=0.01, refit_alpha=1.0):
        self.selection_alpha = selection_alpha
        self.refit_alpha = refit_alpha

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.selector_ = Lasso(alpha=self.selection_alpha, max_iter=150000,
                               tol=1e-5, selection="random", random_state=42,
                               precompute=True).fit(X, y)
        self.support_ = np.flatnonzero(np.abs(self.selector_.coef_) > 1e-8)
        self.mean_ = float(np.mean(y))
        if len(self.support_):
            self.regressor_ = (LinearRegression() if self.refit_alpha == 0
                               else Ridge(alpha=self.refit_alpha, solver="cholesky"))
            self.regressor_.fit(X[:, self.support_], y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        if not len(self.support_):
            return np.full(len(X), self.mean_)
        return self.regressor_.predict(X[:, self.support_])


def make_model(config):
    if config.get("boundary_weighting", False):
        return BoundaryWeightedPolynomial(config)
    if config["method"] == "average":
        return PolynomialAverage(config["members"])
    degree = int(config["degree"])
    basis = config.get("basis", "monomial")
    expansion = (PolynomialFeatures(degree, include_bias=False) if basis == "monomial"
                 else OrthogonalPolynomialFeatures(degree, basis))
    method = config["method"]
    if method == "ridge":
        estimator = Ridge(alpha=config["alpha"], solver="cholesky")
    elif method == "lasso":
        estimator = Lasso(alpha=config["alpha"], max_iter=150000, tol=1e-5,
                          selection="random", random_state=42, precompute=True)
    elif method == "elastic_net":
        estimator = ElasticNet(alpha=config["alpha"], l1_ratio=config["l1_ratio"],
                               max_iter=150000, tol=1e-5, selection="random",
                               random_state=42, precompute=True)
    elif method == "sparse_refit":
        estimator = SparseRefit(config["selection_alpha"], config["refit_alpha"])
    else:
        raise ValueError(f"Unknown method {method}")
    return make_pipeline(expansion, StandardScaler(), estimator)


class BoundaryWeightedPolynomial(RegressorMixin, BaseEstimator):
    """Weight fitting rows toward the test's boundary-count distribution.

    Uses test inputs only, with smoothing and a weight cap to limit the
    influence of rare boundary groups. Final predictions remain polynomials.
    """

    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        counts = np.sum(np.abs(np.asarray(X)) == 1, axis=1)
        fitting_counts = np.bincount(counts, minlength=X.shape[1] + 1)
        fitting_proportions = (fitting_counts + 2) / (len(X) + 2 * len(fitting_counts))
        target_proportions = np.asarray(self.config["test_boundary_proportions"])
        weights = np.clip(target_proportions[counts] / fitting_proportions[counts], 0.2, 5)
        weights /= weights.mean()
        inner = {k: v for k, v in self.config.items()
                 if k not in ["boundary_weighting", "test_boundary_proportions"]}
        self.model_ = make_model(inner)
        self.model_.fit(X, y, standardscaler__sample_weight=weights,
                       **{self.model_.steps[-1][0] + "__sample_weight": weights})
        self.weight_range_ = (float(weights.min()), float(weights.max()))
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model_.predict(X)


class PolynomialAverage(RegressorMixin, BaseEstimator):
    """Equal average of polynomial regressors; predictions remain polynomials."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.models_ = [make_model(config).fit(X, y) for config in self.members]
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.mean([model.predict(X) for model in self.models_], axis=0)
