"""Tests for insurance-tmle."""

import numpy as np
import pytest
from insurance_tmle import TMLE, DoubleMLE, NaiveGLM


def make_confounded_data(n=2000, true_ate=0.5, seed=42):
    """Generate confounded data where naive GLM is biased."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 3))
    # Confounded treatment: high-risk (W[:,0] high) less likely treated
    prob_treat = 1 / (1 + np.exp(-(-W[:, 0] + 0.5 * W[:, 1])))
    A = rng.binomial(1, prob_treat)
    # Outcome depends on both W and A
    Y = true_ate * A + 0.8 * W[:, 0] + 0.3 * W[:, 1] + rng.normal(0, 0.5, n)
    return W, Y, A


def test_tmle_recovers_ate():
    """TMLE should recover the true ATE with low bias."""
    W, Y, A = make_confounded_data(n=3000, true_ate=0.5)
    est = TMLE(outcome_learner="glm", propensity_learner="logistic", n_folds=3)
    result = est.fit(W, Y, A)
    assert abs(result.ate - 0.5) < 0.15, f"ATE bias too large: {result.ate:.3f} vs 0.5"


def test_tmle_ci_contains_true():
    """95% CI from TMLE should contain the true ATE."""
    W, Y, A = make_confounded_data(n=3000, true_ate=0.5)
    est = TMLE(outcome_learner="glm", propensity_learner="logistic", n_folds=3)
    result = est.fit(W, Y, A)
    assert result.ci_lower < 0.5 < result.ci_upper, (
        f"True ATE not in CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
    )


def test_dml_recovers_ate():
    """DML should recover the true ATE with low bias."""
    W, Y, A = make_confounded_data(n=3000, true_ate=0.5)
    est = DoubleMLE(n_folds=3)
    result = est.fit(W, Y, A)
    assert abs(result.ate - 0.5) < 0.15, f"DML ATE bias too large: {result.ate:.3f}"


def test_naive_is_biased():
    """Naive GLM should be biased when a confounder is omitted.

    We construct a scenario where an unobserved variable U drives both treatment
    assignment and outcome. The analyst only observes W (not U), so the naive GLM
    is biased. True ATE = 0.5, but the naive estimate picks up the confounder effect.
    """
    rng = np.random.default_rng(7)
    n = 5000
    # Unobserved confounder U: affects both treatment and outcome
    U = rng.normal(size=n)
    # Observed covariates: unrelated to U
    W = rng.normal(size=(n, 2))
    # Treatment probability strongly driven by U (unobserved confounder)
    prob_treat = 1 / (1 + np.exp(-(2.0 * U)))
    A = rng.binomial(1, prob_treat)
    # Outcome: depends on true ATE, W, and U (unobserved)
    Y = 0.5 * A + 0.5 * W[:, 0] + 3.0 * U + rng.normal(0, 0.5, n)
    # Naive GLM only sees W, not U — so it's biased
    est = NaiveGLM()
    result = est.fit(W, Y, A)
    # The naive estimate should be substantially above 0.5 because positive U
    # increases both treatment probability and outcome (upward bias)
    assert result.ate > 0.5 + 0.2, (
        f"Naive GLM should be upward-biased (omitted U), got ATE={result.ate:.3f}"
    )


def test_result_summary():
    W, Y, A = make_confounded_data(n=500)
    result = TMLE(n_folds=2).fit(W, Y, A)
    summary = result.summary()
    assert "ATE" in summary
    assert "95% CI" in summary


def test_tmle_zero_ate():
    """TMLE should return near-zero ATE when treatment has no effect."""
    rng = np.random.default_rng(99)
    n = 2000
    W = rng.normal(size=(n, 2))
    A = rng.binomial(1, 0.5, n)
    Y = W[:, 0] + rng.normal(0, 1, n)  # no treatment effect
    result = TMLE(n_folds=3).fit(W, Y, A)
    assert abs(result.ate) < 0.2, f"Expected near-zero ATE, got {result.ate:.3f}"
