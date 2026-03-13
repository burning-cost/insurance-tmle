"""
insurance-tmle: Targeted Maximum Likelihood Estimation for insurance causal inference.

TMLE is a doubly-robust semiparametric estimator for average treatment effects (ATE).
It combines an outcome model (Q) and a propensity score model (g) to produce a
bias-corrected estimate with valid confidence intervals even when one of the two
nuisance models is misspecified.

In insurance pricing, TMLE is used to measure the causal effect of a treatment
(e.g. a telematics device, a loyalty discount, a rate change) on an outcome
(claims frequency, pure premium, lapse rate) from observational data.

Key classes:
    TMLE           — Targeted MLE estimator for binary treatment, continuous/count outcome
    DoubleMLE      — Double Machine Learning (cross-fitting) for comparison
    TMLEResult     — Named result container with ATE, CI, influence curve diagnostics

Typical usage::

    from insurance_tmle import TMLE

    est = TMLE(outcome_learner="glm", propensity_learner="logistic", n_folds=5)
    result = est.fit(X, y, treatment)
    print(result.summary())
"""

from insurance_tmle.estimator import TMLE, TMLEResult
from insurance_tmle.dml import DoubleMLE, DMLResult
from insurance_tmle.naive import NaiveGLM, NaiveResult

__version__ = "0.1.0"

__all__ = [
    "TMLE",
    "TMLEResult",
    "DoubleMLE",
    "DMLResult",
    "NaiveGLM",
    "NaiveResult",
    "__version__",
]
