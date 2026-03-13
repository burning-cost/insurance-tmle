"""
Naive GLM baseline for ATE estimation.

The naive approach: ignore the treatment assignment mechanism and simply
include the treatment indicator as a covariate in a GLM. This is what most
pricing actuaries would do if asked "estimate the effect of telematics on claims".

The naive GLM is biased when the treatment is confounded — i.e. when the
probability of receiving treatment depends on features that also predict the outcome.
In insurance, this is almost always the case: telematics devices are more likely to
be adopted by careful young drivers; loyalty discounts are offered to low-risk
long-term customers; rate changes are applied to selected segments.

The bias is the omitted variable bias in econometrics:
    bias = (X'X)^{-1} X'A * gamma
where gamma is the unobserved confounder effect. TMLE and DML remove this bias
by explicitly modelling the treatment assignment mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class NaiveResult:
    """Results from a naive GLM ATE estimation."""

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_obs: int
    n_treated: int

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Naive GLM: ATE (biased, ignores confounding)",
            "=" * 55,
            f"  ATE:               {self.ate:+.6f}",
            f"  SE:                 {self.se:.6f}",
            f"  95% CI:            [{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]",
            f"  p-value:            {self.pvalue:.4f}",
            "-" * 55,
            f"  N obs:              {self.n_obs:,}",
            f"  N treated:          {self.n_treated:,}  ({100*self.n_treated/self.n_obs:.1f}%)",
            "  WARNING: This estimate is biased under confounding.",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ate": self.ate,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "pvalue": self.pvalue,
        }


class NaiveGLM:
    """Naive GLM estimator for ATE: ignores confounding.

    Fits a linear regression of Y on (A, W), treating the coefficient on A
    as the ATE. Biased when A is confounded by W.

    Parameters
    ----------
    outcome_type : 'linear' or 'poisson'
        Model family for the outcome. 'linear' uses OLS. 'poisson' uses Poisson GLM.
    """

    def __init__(self, outcome_type: Literal["linear", "poisson"] = "linear"):
        self.outcome_type = outcome_type
        self._result: NaiveResult | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> NaiveResult:
        """Fit naive GLM.

        Parameters
        ----------
        X : array-like (n, p)
        y : array-like (n,)
        treatment : array-like (n,)

        Returns
        -------
        NaiveResult
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.asarray(treatment, dtype=float).reshape(-1, 1)
        n = len(y)

        # Augment features with treatment indicator
        X_aug = np.column_stack([A, X])

        if self.outcome_type == "poisson":
            model = PoissonRegressor(max_iter=1000, alpha=0)
            scaler = StandardScaler()
            X_aug_s = scaler.fit_transform(X_aug)
            model.fit(X_aug_s, y)
            coef = model.coef_  # on scaled features
            # Unscale: coef_unscaled[j] = coef[j] / scale[j]
            coef_unscaled = coef / scaler.scale_
            # For Poisson log-link, the coefficient on A is not directly the ATE
            # We compute it via marginal effect: dE[Y]/dA = exp(mu) * coef_A
            mu = model.predict(X_aug_s)
            ate = float(np.mean(mu * coef[0] / scaler.scale_[0]))
            # Bootstrap SE would be needed for proper inference; use OLS approximation
            se = float(np.std(mu * coef[0] / scaler.scale_[0]) / np.sqrt(n))
        else:
            # OLS with heteroscedasticity-robust (HC1) SE
            model = LinearRegression(fit_intercept=True)
            model.fit(X_aug, y)
            # Coefficient on A is the first covariate
            ate = float(model.coef_[0])

            # HC1 robust SE
            y_hat = model.predict(X_aug)
            resid = y - y_hat
            # X matrix for HC1 (with intercept)
            X_int = np.column_stack([np.ones(n), X_aug])
            H = X_int.T @ X_int
            try:
                H_inv = np.linalg.pinv(H)
            except np.linalg.LinAlgError:
                H_inv = np.diag(1.0 / np.diag(H))
            meat = (X_int * resid[:, None]).T @ (X_int * resid[:, None]) * n / (n - X_int.shape[1])
            vcov = H_inv @ meat @ H_inv
            # Index of A in X_int is 1 (after intercept)
            se = float(np.sqrt(vcov[1, 1]))

        z      = stats.norm.ppf(0.975)
        ci_lo  = ate - z * se
        ci_hi  = ate + z * se
        pvalue = float(2 * stats.norm.sf(abs(ate / max(se, 1e-12))))

        self._result = NaiveResult(
            ate=float(ate),
            se=se,
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            pvalue=pvalue,
            n_obs=n,
            n_treated=int(A.sum()),
        )
        return self._result
