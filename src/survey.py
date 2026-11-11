"""Estimates and standard errors for a stratified, clustered survey.

NHANES is not a simple random sample. People are sampled inside areas
(sampling units, PSUs), areas inside strata, and each person carries a
weight. Ignoring that gives the right-looking estimate with a standard
error that is too small.

The functions here use Taylor linearisation, the method behind SAS
SURVEYMEANS, Stata's svy prefix and SPSS Complex Samples:

1. write the estimate so that each person contributes one "linearised"
   value;
2. add those values up within each sampling unit;
3. measure how much the sampling-unit totals vary within each stratum.

Subgroups are handled as domains: people outside the subgroup stay in the
calculation with a contribution of zero, so every sampling unit is still
counted. Deleting them instead would understate the variance.

Degrees of freedom are (sampling units - strata), which is 15 for this
NHANES cycle, so intervals use the t distribution, not 1.96.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class Design:
    """The survey design: one weight, stratum and sampling unit per participant."""

    def __init__(self, data: pd.DataFrame, weight: str, stratum: str, psu: str) -> None:
        self.weight = data[weight].to_numpy(float)
        self.cluster = pd.MultiIndex.from_arrays([data[stratum].to_numpy(), data[psu].to_numpy()])
        self.n_strata = data[stratum].nunique()
        self.n_psu = len(self.cluster.unique())
        self.df = self.n_psu - self.n_strata
        self.t = stats.t.ppf(0.975, self.df)

    def variance(self, linearised: np.ndarray) -> np.ndarray:
        """Variance (or covariance matrix) of the total of the linearised values."""
        values = np.asarray(linearised, float)
        if values.ndim == 1:
            values = values[:, None]
        totals = pd.DataFrame(values).groupby([self.cluster.get_level_values(0), self.cluster.get_level_values(1)]).sum()
        k = values.shape[1]
        covariance = np.zeros((k, k))
        for _, group in totals.groupby(level=0):
            n = len(group)
            if n > 1:
                centred = group.to_numpy() - group.to_numpy().mean(axis=0)
                covariance += n / (n - 1) * centred.T @ centred
        return covariance

    # ------------------------------------------------------------------
    def proportion(self, y: np.ndarray, domain: np.ndarray) -> dict:
        """Weighted proportion (or mean) of y inside a domain, with a 95% interval."""
        y = np.asarray(y, float)
        inside = np.asarray(domain, bool) & ~np.isnan(y) & (self.weight > 0)
        w = np.where(inside, self.weight, 0.0)
        value = np.where(inside, y, 0.0)
        population = w.sum()
        estimate = (w * value).sum() / population
        se = float(np.sqrt(self.variance(w * (value - estimate) / population)[0, 0]))
        if 0 < estimate < 1 and se > 0:
            # interval on the logit scale, so it cannot leave 0-1
            logit = np.log(estimate / (1 - estimate))
            half = self.t * se / (estimate * (1 - estimate))
            low, high = 1 / (1 + np.exp(-(logit - half))), 1 / (1 + np.exp(-(logit + half)))
        else:
            low, high = estimate - self.t * se, estimate + self.t * se
        return {"estimate": estimate, "se": se, "low": low, "high": high,
                "n": int(inside.sum()), "population": population}

    def total(self, y: np.ndarray, domain: np.ndarray) -> dict:
        """Weighted number of people in the domain with y = 1, with a 95% interval."""
        y = np.asarray(y, float)
        inside = np.asarray(domain, bool) & ~np.isnan(y) & (self.weight > 0)
        contribution = np.where(inside, self.weight * np.where(inside, y, 0.0), 0.0)
        estimate = contribution.sum()
        se = float(np.sqrt(self.variance(contribution)[0, 0]))
        return {"estimate": estimate, "se": se, "low": estimate - self.t * se, "high": estimate + self.t * se}

    # ------------------------------------------------------------------
    def logistic(self, X: pd.DataFrame, y: np.ndarray, domain: np.ndarray) -> pd.DataFrame:
        """Weighted logistic regression with design-based standard errors.

        Coefficients come from Newton's method on the weighted likelihood.
        The covariance is the sandwich  A^-1 B A^-1 : A is the weighted
        information matrix, B the between-sampling-unit variance of the
        score contributions.
        """
        names = list(X.columns)
        matrix = X.to_numpy(float)
        y = np.asarray(y, float)
        use = np.asarray(domain, bool) & ~np.isnan(y) & ~np.isnan(matrix).any(axis=1) & (self.weight > 0)
        Xu, yu, wu = matrix[use], y[use], self.weight[use]
        wu = wu / wu.mean()  # rescaling the weights changes nothing but keeps the numbers small

        beta = np.zeros(Xu.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-Xu @ beta))
            gradient = Xu.T @ (wu * (yu - p))
            information = Xu.T @ (Xu * (wu * p * (1 - p))[:, None])
            step = np.linalg.solve(information, gradient)
            beta += step
            if np.abs(step).max() < 1e-10:
                break
        else:
            raise RuntimeError("logistic regression did not converge")

        p = 1 / (1 + np.exp(-Xu @ beta))
        information = Xu.T @ (Xu * (wu * p * (1 - p))[:, None])
        scores = np.zeros_like(matrix)
        scores[use] = (wu * (yu - p))[:, None] * Xu
        bread = np.linalg.inv(information)
        covariance = bread @ self.variance(scores) @ bread
        se = np.sqrt(np.diag(covariance))
        return pd.DataFrame({
            "term": names,
            "coefficient": beta,
            "se": se,
            "odds_ratio": np.exp(beta),
            "low": np.exp(beta - self.t * se),
            "high": np.exp(beta + self.t * se),
            "p_value": 2 * stats.t.sf(np.abs(beta / se), self.df),
        }).assign(n=int(use.sum()), events=int(yu.sum()))


def dummies(data: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Indicator columns for categorical predictors. The first level listed is the reference.

    A row with a missing value in any predictor gets NaN in that predictor's
    columns, so the regression leaves it out.
    """
    columns = {"Intercept": np.ones(len(data))}
    for variable, ordered in levels.items():
        for level in ordered[1:]:
            indicator = (data[variable] == level).astype(float)
            columns[f"{variable}: {level}"] = indicator.where(data[variable].notna())
    return pd.DataFrame(columns, index=data.index)
