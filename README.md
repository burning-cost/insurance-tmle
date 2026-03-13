# insurance-tmle

Targeted Maximum Likelihood Estimation (TMLE) for causal inference in insurance pricing.

## The problem

You want to know whether a telematics device actually reduces claims, or whether
telematics adopters are just inherently safer drivers. Or whether a loyalty discount
causes retention, or whether loyal customers would have stayed anyway. Or whether
a rate change in a segment caused the observed volume shift, or whether it was
correlated market movement.

These are all causal questions. Standard GLMs — even with the treatment as a covariate —
give biased answers when treatment assignment is confounded by risk profile. And in
insurance, it almost always is.

The standard actuary approach (treatment as a covariate in a Poisson GLM) has a name
in econometrics: omitted variable bias. The bias is proportional to how strongly the
treatment predicts the outcome *through confounders*. On telematics uptake, where
adoption rate varies by 40+ points across risk groups, the bias is not small.

## The solution

TMLE is a doubly-robust semiparametric estimator for average treatment effects (ATE).
It combines two nuisance models — the outcome model Q(A, W) and the propensity model
g(W) = P(A=1|W) — and then applies a targeting step that solves the efficient score
equation. The result is:

- **Doubly robust**: consistent if *either* Q or g is correctly specified
- **Efficient**: achieves the semiparametric efficiency bound (smallest asymptotic variance)
- **Valid CI**: from the efficient influence curve, not parametric assumptions
- **Cross-fitted**: uses K-fold to avoid Donsker conditions on the nuisance models

Also included: `DoubleMLE` (Robinson's partialling-out) and `NaiveGLM` for comparison.

## Installation

```bash
pip install git+https://github.com/burning-cost/insurance-tmle.git
```

## Usage

```python
from insurance_tmle import TMLE

# X: covariate matrix (confounders)
# y: outcome (claim rate, lapse rate, etc.)
# treatment: binary treatment indicator (0/1)

est = TMLE(
    outcome_learner="glm",        # or 'gbm', or any sklearn regressor
    propensity_learner="logistic", # or 'gbm', or any sklearn classifier
    n_folds=5,
    propensity_clip=0.05,
)
result = est.fit(X, y, treatment)
print(result.summary())
# ATE:         -0.0482
# SE:           0.0031
# 95% CI:      [-0.0543, -0.0421]
# p-value:      0.0000
```

## When to use it

Use TMLE when:
- Treatment uptake varies by risk segment (telematics, loyalty, affinity schemes)
- You need valid confidence intervals on causal claims for regulatory review
- The treatment is not randomised (observational data from operations, not an RCT)
- You want doubly-robust protection against model misspecification

Use naive GLM when:
- Treatment is randomised (A/B test, RCT, randomised rate filing)
- You need speed over correctness and bias is acceptable
- Confounding is weak (propensity scores near 0.5 everywhere)

## Performance

Benchmarked against naive GLM and Double ML on synthetic confounded motor data
(20,000 policies, known DGP, confounder strength 0.8). See `notebooks/benchmark_tmle.py`
for full methodology.

- **TMLE recovers the true ATE with near-zero bias** under strong confounding; naive GLM
  bias grows linearly with confounder strength and can be 2-5x the true effect size.
- **TMLE achieves ~95% CI coverage** across 100 simulations; naive GLM CI coverage can
  fall to 30-60% under strong confounding because the bias dominates the interval.
- **DML is also consistent** and close to TMLE in performance, but lacks the doubly-robust
  targeting step — misspecified outcome model is not corrected by propensity.
- **Fit time is 2-10x naive GLM** depending on learner and folds (GLM: ~2s for n=5k, 5-fold).
  Comfortably within a monthly pricing cycle; not suitable for real-time scoring.
- **GBM nuisance models** substantially reduce bias when the true Q and g are non-linear,
  at the cost of longer fit times (30-60s for n=20k, 5-fold).

## References

- van der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. *IJB*, 2(1).
- Chernozhukov, V., et al. (2018). Double/Debiased ML. *Econometrics Journal*, 21(1).
- Schuler, A., & Rose, S. (2017). TMLE for causal inference in observational studies. *AJE*, 185(1).
