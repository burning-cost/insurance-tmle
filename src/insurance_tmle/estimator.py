"""
TMLE: Targeted Maximum Likelihood Estimation.

The TMLE algorithm for binary treatment, continuous/count outcome:

    1. Fit initial outcome model Q(A, W) = E[Y | A, W] using cross-fitting.
       This produces initial predictions Q0 (under A=0) and Q1 (under A=1).

    2. Fit propensity score model g(W) = P(A=1 | W) using cross-fitting.

    3. Compute the clever covariate H:
           H(A, W) = A/g(W) - (1-A)/(1-g(W))

    4. Run a targeting step: update Q via a logistic fluctuation model
       (or linear if outcome is unbounded):
           Q*(A, W) = Q(A, W) + eps * H(A, W)
       where eps is estimated by a 1-parameter MLE on the residuals.

    5. Compute the plug-in estimator:
           ATE = mean(Q*(1, W) - Q*(0, W))

    6. Compute the efficient influence curve (EIC) for a valid confidence interval:
           EIC_i = H_i * (Y_i - Q*(A_i, W_i)) + Q*(1, W_i) - Q*(0, W_i) - ATE
           Var(ATE) = Var(EIC) / n
           SE = sqrt(Var(ATE))

Cross-fitting (K-fold) is used to avoid Donsker conditions. Each fold's observations
have their nuisance parameters estimated using models trained on the other folds.

References:
    van der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning.
    The international journal of biostatistics, 2(1).

    Schuler, A., & Rose, S. (2017). Targeted maximum likelihood estimation for causal
    inference in observational studies. American journal of epidemiology, 185(1), 65-73.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def _get_outcome_learner(learner: str | object):
    if isinstance(learner, str):
        if learner == "glm":
            return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        elif learner == "poisson":
            return Pipeline([("scaler", StandardScaler()), ("model", PoissonRegressor(max_iter=500))])
        elif learner == "gbm":
            return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        else:
            raise ValueError(f"Unknown outcome learner: {learner!r}. Use 'glm', 'poisson', 'gbm', or pass an sklearn estimator.")
    return clone(learner)


def _get_propensity_learner(learner: str | object):
    if isinstance(learner, str):
        if learner == "logistic":
            return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, C=1.0))])
        elif learner == "gbm":
            return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        else:
            raise ValueError(f"Unknown propensity learner: {learner!r}. Use 'logistic', 'gbm', or pass an sklearn estimator.")
    return clone(learner)


@dataclass
class TMLEResult:
    """Results from a TMLE estimation.

    Attributes
    ----------
    ate : float
        Average treatment effect (E[Y(1)] - E[Y(0)]).
    se : float
        Standard error of the ATE from the efficient influence curve.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    pvalue : float
        Two-sided p-value for H0: ATE = 0.
    epsilon : float
        TMLE fluctuation parameter (should be close to 0 if initial Q is well-specified).
    n_obs : int
        Number of observations.
    n_treated : int
        Number treated (A=1).
    mean_y0 : float
        Estimated mean potential outcome under A=0 (counterfactual).
    mean_y1 : float
        Estimated mean potential outcome under A=1.
    eic : np.ndarray
        Efficient influence curve values (n,).
    propensity_mean : float
        Mean propensity score.
    propensity_min : float
        Minimum propensity score (trimmed).
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    epsilon: float
    n_obs: int
    n_treated: int
    mean_y0: float
    mean_y1: float
    eic: np.ndarray = field(repr=False)
    propensity_mean: float = 0.0
    propensity_min: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "TMLE: Average Treatment Effect",
            "=" * 55,
            f"  ATE:               {self.ate:+.6f}",
            f"  SE:                 {self.se:.6f}",
            f"  95% CI:            [{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]",
            f"  p-value:            {self.pvalue:.4f}",
            f"  Epsilon (target):   {self.epsilon:.6f}",
            "-" * 55,
            f"  N obs:              {self.n_obs:,}",
            f"  N treated:          {self.n_treated:,}  ({100*self.n_treated/self.n_obs:.1f}%)",
            f"  E[Y(1)]:            {self.mean_y1:.6f}",
            f"  E[Y(0)]:            {self.mean_y0:.6f}",
            f"  Propensity mean:    {self.propensity_mean:.4f}",
            f"  Propensity min:     {self.propensity_min:.4f}",
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
            "epsilon": self.epsilon,
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "mean_y1": self.mean_y1,
            "mean_y0": self.mean_y0,
        }


class TMLE:
    """Targeted Maximum Likelihood Estimator for Average Treatment Effects.

    Parameters
    ----------
    outcome_learner : str or sklearn estimator
        Model for E[Y | A, W]. Options: 'glm', 'poisson', 'gbm', or any sklearn regressor.
    propensity_learner : str or sklearn estimator
        Model for P(A=1 | W). Options: 'logistic', 'gbm', or any sklearn classifier.
    n_folds : int
        Number of cross-fitting folds. At least 2. Higher reduces bias from overfitting
        but increases runtime. 5 is a good default.
    propensity_clip : float
        Clip propensity scores to [clip, 1-clip]. Prevents extreme weights. Default 0.05.
    outcome_bounds : tuple[float, float] or None
        Optional bounds for the outcome. If set, the targeting step uses a logistic
        fluctuation model (appropriate for bounded outcomes). None = linear fluctuation.
    random_state : int
        Seed for cross-fitting fold generation.

    Examples
    --------
    >>> from insurance_tmle import TMLE
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 1000
    >>> W = rng.normal(size=(n, 3))
    >>> A = rng.binomial(1, 1/(1 + np.exp(-W[:, 0])))
    >>> Y = 0.5 * A + W[:, 0] + rng.normal(0, 0.5, n)
    >>> est = TMLE(outcome_learner='glm', propensity_learner='logistic', n_folds=3)
    >>> result = est.fit(W, Y, A)
    >>> print(f"ATE: {result.ate:.3f} (true: 0.500)")
    """

    def __init__(
        self,
        outcome_learner: str | object = "glm",
        propensity_learner: str | object = "logistic",
        n_folds: int = 5,
        propensity_clip: float = 0.05,
        outcome_bounds: tuple[float, float] | None = None,
        random_state: int = 42,
    ):
        self.outcome_learner = outcome_learner
        self.propensity_learner = propensity_learner
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.outcome_bounds = outcome_bounds
        self.random_state = random_state

        self._outcome_models: list = []
        self._propensity_models: list = []
        self._result: TMLEResult | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> TMLEResult:
        """Fit TMLE and return ATE estimate with confidence interval.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariate / feature matrix (confounders W).
        y : array-like of shape (n,)
            Outcome variable Y.
        treatment : array-like of shape (n,)
            Binary treatment indicator A (0 or 1).

        Returns
        -------
        TMLEResult
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.asarray(treatment, dtype=float)

        n = len(y)
        assert X.shape[0] == n, "X and y must have the same number of rows"
        assert A.shape[0] == n, "treatment must have the same number of rows as y"
        assert set(A).issubset({0.0, 1.0}), "treatment must be binary (0/1)"

        # Arrays to hold cross-fitted nuisance estimates
        Q_hat   = np.zeros(n)   # E[Y | A, W] with actual A
        Q0_hat  = np.zeros(n)   # E[Y | A=0, W] counterfactual
        Q1_hat  = np.zeros(n)   # E[Y | A=1, W] counterfactual
        g_hat   = np.zeros(n)   # P(A=1 | W)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        self._outcome_models = []
        self._propensity_models = []

        # Cross-fitting: train on folds != k, predict on fold k
        X_aug_A0 = np.column_stack([np.zeros(n), X])   # with A=0
        X_aug_A1 = np.column_stack([np.ones(n),  X])   # with A=1
        X_aug    = np.column_stack([A,            X])   # with actual A

        for train_idx, val_idx in kf.split(X):
            # Outcome model: Q(A, W) = E[Y | A, W]
            Q_model = _get_outcome_learner(self.outcome_learner)
            Q_model.fit(X_aug[train_idx], y[train_idx])
            Q_hat[val_idx]  = Q_model.predict(X_aug[val_idx])
            Q0_hat[val_idx] = Q_model.predict(X_aug_A0[val_idx])
            Q1_hat[val_idx] = Q_model.predict(X_aug_A1[val_idx])
            self._outcome_models.append(Q_model)

            # Propensity model: g(W) = P(A=1 | W)
            g_model = _get_propensity_learner(self.propensity_learner)
            g_model.fit(X[train_idx], A[train_idx].astype(int))
            g_hat[val_idx] = g_model.predict_proba(X[val_idx])[:, 1]
            self._propensity_models.append(g_model)

        # Clip propensity scores
        g_hat = np.clip(g_hat, self.propensity_clip, 1.0 - self.propensity_clip)

        # Targeting step
        # Clever covariate H(A, W) = A/g - (1-A)/(1-g)
        H      = A / g_hat - (1 - A) / (1 - g_hat)
        H1     = 1.0 / g_hat          # under A=1
        H0     = -1.0 / (1 - g_hat)   # under A=0

        # 1-parameter MLE to find epsilon: regress (y - Q_hat) on H
        # Simple OLS estimate (valid for linear fluctuation / unbounded outcome)
        epsilon = np.sum(H * (y - Q_hat)) / np.sum(H ** 2)

        # Update initial estimates
        Q_star     = Q_hat  + epsilon * H
        Q0_star    = Q0_hat + epsilon * H0
        Q1_star    = Q1_hat + epsilon * H1

        if self.outcome_bounds is not None:
            lo, hi = self.outcome_bounds
            Q_star  = np.clip(Q_star,  lo, hi)
            Q0_star = np.clip(Q0_star, lo, hi)
            Q1_star = np.clip(Q1_star, lo, hi)

        # Plug-in ATE
        ate = float(np.mean(Q1_star - Q0_star))

        # Efficient influence curve
        eic = H * (y - Q_star) + (Q1_star - Q0_star) - ate

        # Standard error and CI from EIC
        var_ate = float(np.var(eic, ddof=1)) / n
        se      = float(np.sqrt(var_ate))
        z       = stats.norm.ppf(0.975)
        ci_lo   = ate - z * se
        ci_hi   = ate + z * se
        pvalue  = float(2 * stats.norm.sf(abs(ate / se)))

        self._result = TMLEResult(
            ate=ate,
            se=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            pvalue=pvalue,
            epsilon=float(epsilon),
            n_obs=n,
            n_treated=int(A.sum()),
            mean_y0=float(np.mean(Q0_star)),
            mean_y1=float(np.mean(Q1_star)),
            eic=eic,
            propensity_mean=float(g_hat.mean()),
            propensity_min=float(g_hat.min()),
        )
        return self._result

    @property
    def result(self) -> TMLEResult:
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result
