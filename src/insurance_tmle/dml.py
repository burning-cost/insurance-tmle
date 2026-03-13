"""
Double Machine Learning (DML / Partially Linear Model) for ATE estimation.

DML uses the Robinson (1988) partialling-out approach:
    1. Residualise Y on W: tilde_Y = Y - E[Y|W]
    2. Residualise A on W: tilde_A = A - E[A|W]
    3. Regress tilde_Y on tilde_A to get ATE

Cross-fitting is used for both nuisance estimates to avoid regularisation bias.
This is the approach from Chernozhukov et al. (2018) "Double/Debiased ML".

DML is more transparent than TMLE but does not use the propensity score
directly — it uses it implicitly via the residualisation. The confidence
interval comes from OLS variance on the residualised regression.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from insurance_tmle.estimator import _get_outcome_learner, _get_propensity_learner

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DMLResult:
    """Results from a Double ML estimation."""

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
            "Double ML (Partialling-Out): ATE",
            "=" * 55,
            f"  ATE:               {self.ate:+.6f}",
            f"  SE:                 {self.se:.6f}",
            f"  95% CI:            [{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]",
            f"  p-value:            {self.pvalue:.4f}",
            "-" * 55,
            f"  N obs:              {self.n_obs:,}",
            f"  N treated:          {self.n_treated:,}  ({100*self.n_treated/self.n_obs:.1f}%)",
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


class DoubleMLE:
    """Double Machine Learning estimator (partialling-out / Robinson's method).

    Parameters
    ----------
    outcome_learner : str or estimator
        Model for E[Y | W]. Same options as TMLE.
    treatment_learner : str or estimator
        Model for E[A | W]. Options: 'logistic', 'gbm', or classifier.
    n_folds : int
        Cross-fitting folds.
    random_state : int

    References
    ----------
    Chernozhukov, V., et al. (2018). Double/Debiased machine learning for treatment
    and structural parameters. The Econometrics Journal, 21(1), C1-C68.
    """

    def __init__(
        self,
        outcome_learner: str | object = "glm",
        treatment_learner: str | object = "logistic",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.outcome_learner = outcome_learner
        self.treatment_learner = treatment_learner
        self.n_folds = n_folds
        self.random_state = random_state
        self._result: DMLResult | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> DMLResult:
        """Fit Double ML and return ATE estimate.

        Parameters
        ----------
        X : array-like (n, p)
        y : array-like (n,)
        treatment : array-like (n,) — binary 0/1

        Returns
        -------
        DMLResult
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.asarray(treatment, dtype=float)
        n = len(y)

        y_resid = np.zeros(n)
        a_resid = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(X):
            # Step 1: Residualise Y on W
            m_y = _get_outcome_learner(self.outcome_learner)
            m_y.fit(X[train_idx], y[train_idx])
            y_resid[val_idx] = y[val_idx] - m_y.predict(X[val_idx])

            # Step 2: Residualise A on W
            m_a = _get_propensity_learner(self.treatment_learner)
            m_a.fit(X[train_idx], A[train_idx].astype(int))
            a_resid[val_idx] = A[val_idx] - m_a.predict_proba(X[val_idx])[:, 1]

        # Step 3: OLS of y_resid on a_resid (no intercept)
        theta = np.sum(a_resid * y_resid) / np.sum(a_resid ** 2)

        # Heteroscedasticity-robust SE (HC1)
        residuals = y_resid - theta * a_resid
        hc_meat   = np.sum((a_resid * residuals) ** 2)
        hc_bread  = np.sum(a_resid ** 2) ** 2
        var_theta = hc_meat / hc_bread
        se        = float(np.sqrt(var_theta))

        z      = stats.norm.ppf(0.975)
        ci_lo  = theta - z * se
        ci_hi  = theta + z * se
        pvalue = float(2 * stats.norm.sf(abs(theta / se)))

        self._result = DMLResult(
            ate=float(theta),
            se=se,
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            pvalue=pvalue,
            n_obs=n,
            n_treated=int(A.sum()),
        )
        return self._result
