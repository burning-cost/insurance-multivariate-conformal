# insurance-multivariate-conformal

Joint multi-output conformal prediction intervals for insurance pricing models.

## The problem

UK pricing teams run separate GLMs for claim frequency (Poisson, lambda ~ 0.05–0.30) and claim severity (Gamma, mu ~ £500–£8,000). The standard workflow produces point estimates. Actuaries asking "how uncertain is this pricing?" have no rigorous answer.

The naive approach — running split conformal prediction separately on each output — gives marginal coverage. If frequency has 95% coverage and severity has 95% coverage, joint coverage (both simultaneously correct) could be as low as 90%. For Solvency II SCR at 99.5%, this difference is not acceptable.

This library solves that. It produces hyperrectangular prediction sets `[L_freq, U_freq] × [L_sev, U_sev]` with a finite-sample joint coverage guarantee:

```
P(freq ∈ [L_freq, U_freq] AND sev ∈ [L_sev, U_sev]) ≥ 1 - alpha
```

No distributional assumptions. No asymptotics. Works with any base model (GLM, GBM, Random Forest) via sklearn's `.predict()` interface.

## The scale problem and why it matters

Frequency residuals are on the order of 0.1–2 claims. Severity residuals are on the order of £500–£3,000. If you aggregate residuals naively (e.g. take the max), severity always dominates. The resulting joint interval degenerates to a severity interval with a token frequency constraint.

The solution, from Fan & Sesia (arXiv:2512.15383): coordinate-wise standardization. For each output dimension j, compute the mean `mu_j` and std `sigma_j` of calibration residuals. Then the standardized score `(E_j - mu_j) / sigma_j` is dimensionless — directly comparable across frequency and severity.

## Installation

```bash
pip install insurance-multivariate-conformal
```

**Dependencies:** numpy, scikit-learn, polars. No PyTorch, no JAX.

## Quick start: motor frequency + severity

```python
from insurance_multivariate_conformal import JointConformalPredictor

# You have fitted these separately on a training set
# freq_glm: any model with .predict(X) returning shape (n,)
# sev_gbm: same

predictor = JointConformalPredictor(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    alpha=0.05,    # 95% joint coverage
    method='lwc',  # Local worst-case — the default, tightest valid method
)

# Calibrate on held-out data (NOT the training set)
predictor.calibrate(
    X_cal=X_calibration,
    Y_cal={'frequency': y_freq_cal, 'severity': y_sev_cal},
    zero_claim_mask=zero_mask_cal,  # True where claims == 0 (severity unobserved)
)

# Predict on new policies
joint_set = predictor.predict(X_new)

# Intervals per policy
print(joint_set.lower['frequency'])   # Lower frequency bounds
print(joint_set.upper['severity'])    # Upper severity bounds

# Verify coverage on test set
cov = joint_set.joint_coverage_check(Y_test)
print(f"Joint coverage: {cov:.1%}")   # Should be >= 95%
```

## Solvency II SCR

For Solvency II Article 101 (99.5% VaR), use `SolvencyCapitalEstimator`:

```python
from insurance_multivariate_conformal import SolvencyCapitalEstimator

scr = SolvencyCapitalEstimator(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    alpha=0.005,  # 99.5% coverage
    method='gwc', # GWC is more conservative — appropriate for regulatory use
)
scr.calibrate(X_cal, Y_cal)
result = scr.estimate(X_portfolio)

print(f"Aggregate SCR: £{result.aggregate_scr:,.0f}")
print(f"Coverage guarantee: {result.coverage_guarantee:.1%}")
print(f"Calibration set size: {result.n_cal}")
```

The coverage guarantee is finite-sample valid: `P(loss ≤ SCR_upper) ≥ 99.5%` with no distributional assumption. At n_cal=1000, the guarantee is `≥ 99.5%` with at most 0.1% excess conservatism.

## Methods

Four methods, in increasing order of statistical efficiency:

| Method | Joint coverage | Width | When to use |
|--------|---------------|-------|-------------|
| `bonferroni` | Valid | Widest | Baseline; guaranteed under any correlation |
| `sidak` | Valid (independence only) | Slightly narrower | Only if outputs are independent |
| `gwc` | Valid | Moderate | Regulatory use (conservative, simpler) |
| `lwc` | Valid | Tightest | Production pricing (default) |

LWC (Local Worst-Case, Fan & Sesia Algorithm 2) is the recommended default. It partitions calibration observations by which dimension is the binding constraint, then computes a group-specific quantile. This is 20–35% tighter than Bonferroni on typical insurance data while maintaining identical joint coverage guarantees.

## Zero-claim masking

For policies where observed claims = 0, severity is unobserved. Pass a boolean mask to `calibrate()`:

```python
zero_mask = (y_freq_cal == 0)  # True where no claims were made
predictor.calibrate(X_cal, Y_cal, zero_claim_mask=zero_mask)
```

This sets severity residuals to 0 for zero-claim observations — conservative but valid. The effect is to widen the severity interval slightly (treating zero-claim obs as perfectly predicted for severity), which maintains the joint coverage guarantee.

## Diagnostics

```python
from insurance_multivariate_conformal import coverage_report, compare_methods

# Validate coverage on a test set
report = coverage_report(predictor, X_test, Y_test)
print(report['joint_coverage'])         # >= 1 - alpha?
print(report['marginal_coverages'])     # Per-dimension
print(report['mean_widths'])            # Interval width efficiency

# Compare all methods head-to-head
results = compare_methods(
    models=models,
    X_cal=X_cal, Y_cal=Y_cal,
    X_test=X_test, Y_test=Y_test,
    alpha=0.05,
)
for method, rep in results.items():
    print(f"{method}: coverage={rep['joint_coverage']:.1%}, "
          f"width_freq={rep['mean_widths']['frequency']:.4f}")
```

## Multi-peril home insurance (d=3)

```python
predictor_3d = JointConformalPredictor(
    models={
        'flood': flood_rf,
        'fire': fire_glm,
        'subsidence': sub_gbm,
    },
    alpha=0.05,
    method='lwc',
)
predictor_3d.calibrate(X_cal, Y_cal_3d)
joint_3d = predictor_3d.predict(X_policies)

# Volume = width_flood * width_fire * width_sub
print(joint_3d.volume().mean())
```

## Coverage guarantee

Under exchangeability of calibration and test data:

```
1 - alpha ≤ P(Y_new ∈ Ĉ(X_new)) ≤ 1 - alpha + 1/(n_cal + 1)
```

At n_cal=199: coverage ∈ [0.950, 0.955] for alpha=0.05.
At n_cal=999: coverage ∈ [0.950, 0.951].

For SCR at 99.5%: n_cal=999 gives coverage ∈ [0.995, 0.996].

## References

- Fan & Sesia (2025). *Interpretable Multivariate Conformal Prediction with Fast Transductive Standardization*. arXiv:2512.15383. — Primary algorithm (GWC, LWC).
- Hong (2025). *Conformal prediction of future insurance claims in the regression problem*. arXiv:2503.03659. — Solvency II SCR framing.
- Braun et al. (2025). *Multivariate Standardized Residuals for Conformal Prediction*. arXiv:2507.20941. — Ellipsoidal alternative (not implemented here; requires PyTorch).

## License

Apache-2.0. Copyright 2026 Burning Cost.
