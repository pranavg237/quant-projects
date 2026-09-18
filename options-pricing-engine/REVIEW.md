# Skeptical review

Written as if reviewing someone else's repo before putting it in front of a desk. Every
item is a real weakness; the ones marked **FIXED** were addressed after the first pass, and
the rest are open with an honest reason.

---

## Things that were wrong and are now fixed

### 1. The Monte Carlo control variate was fitted on the wrong sample — **FIXED**
The first version estimated `beta` on the raw `Z` / `-Z` legs and *then* folded antithetic
pairs. That minimises the variance of an estimator that is never reported. It was not
obviously broken — prices were right and the standard error looked plausible — but stacking
antithetic and control variates came out *worse* than the control alone (SE 0.0175 vs
0.0125), which is the tell. Fitting after the fold gives SE 0.0043, a 57x variance
reduction. A reviewer who only checked that the price was right would have missed it.

### 2. Implied vol converged on price, not volatility — **FIXED**
A price tolerance of 1e-10 sounds strict. On a five-day 10%-OTM call with vega ~1e-6 it
leaves 1e-4 of *volatility* error. Adding a bracket-width criterion took the worst case
from 5e-5 to 1.7e-11. The original would not have been visibly wrong on any chart; it
would just have quietly added noise to the wings of every surface.

### 3. The forward curve was extracted by a badly conditioned regression — **FIXED**
Fitting `C - P = D(F - K)` for both `D` and `F` gave implied rates of +42% and −33% on
adjacent SPY expiries with `R² = 0.9998`. The fix is to take `D` from the Treasury curve
and solve parity for `F` alone. Both methods are kept and both are tested, so the failure
is demonstrable rather than asserted.

### 4. The butterfly arbitrage test was measuring the quote tick — **FIXED**
26% of SPY strike triples failed a zero-tolerance convexity test. At \$1 strike spacing the
theoretical butterfly value is comparable to the \$0.01 tick, so that number is rounding,
not arbitrage. Netting off the bid-ask cost of all three legs leaves 2.9%. Reporting the
26% alone would have been alarming and meaningless.

### 5. Negative time to expiry produced nonsense — **FIXED**
`tau = -1` was silently *inflating* the price via `exp(-r·tau)` rather than returning
intrinsic. Only a test found it. Chains do contain contracts that expire between the
snapshot and the call.

### 6. Charts that did not show what their titles claimed — **FIXED**
The convergence chart sampled only even step counts, which hides the CRR sawtooth the
title advertises. The Monte Carlo chart's second panel plotted single-seed realised error,
which is pure noise; it now plots realised RMSE over replications divided by the reported
standard error, which is a real diagnostic. The 3-D surface used a linear expiry axis that
crushed half the expiries into the front edge.

---

## Open weaknesses

### A. One snapshot, one underlying
Everything here is SPY on 2026-09-18. The single most damning test of a Heston
implementation is **parameter stability across days**: calibrate on consecutive dates and
see how far `(v0, kappa, theta, xi, rho)` moves when the surface barely does. Practitioners
report it moves a lot. Measuring that needs a time series of chains, which needs either a
paid feed or weeks of daily collection. This is the largest gap in the repo and it is
stated in the README rather than papered over.

### B. American exercise is ignored when inverting SPY quotes
SPY options are American. The surface treats them as European. For an index ETF with a
~1% dividend the early-exercise premium on OTM options is small, but it is not zero, and
the pipeline would be materially wrong on a single stock around a dividend. A proper fix
means inverting through the binomial tree (which is implemented, just not wired into
`build_surface`) or applying a de-Americanisation step. Cost: roughly 1000x slower
inversion. Not done.

### C. No confidence intervals on the calibrated parameters — **FIXED, with a caveat**
Now computed from the Jacobian at the optimum. The relative standard errors are small
(0.8%–3.8%), and that *understates* the uncertainty because the calculation assumes
independent residuals while adjacent strikes are strongly dependent. The genuinely useful
output turned out to be the correlation matrix: `corr(kappa, xi) = +0.83` and
`corr(kappa, theta) = -0.79`. My guess before measuring was "above 0.9"; the measured
value is 0.83, so kappa and xi are strongly but not degenerately linked. It remains a
*local* quantity — it describes the basin the optimiser landed in, not global uncertainty
on a multimodal objective.

### D. Model comparison is in-sample — **FIXED**
Now cross-validated: fit on alternate strikes within each expiry, score on the rest.
Heston goes 2.55 in-sample to 2.38 out-of-sample (ratio 0.93); both Black-Scholes
benchmarks sit at 0.96. Nothing overfits, which is what five parameters against 234
training quotes should do — but it is now measured rather than assumed. Note this tests
interpolation across strikes, **not** extrapolation in maturity, which is the harder
question and is still untested (and is really item A in disguise).

### E. The cleaning thresholds are not tuned, just chosen
`max_relative_spread = 0.40`, `min_price = 0.05` and the 5-day minimum are defensible
conventions, not optimised values. A sensitivity analysis — how much do the calibrated
parameters move if the spread filter goes from 20% to 60%? — would say whether the results
are robust to them. Not run.

### F. The Feller condition is violated and only noted
`2·kappa·theta/xi² = 0.16`. That is fine for the characteristic function (the variance
process just touches zero) and typical of equity calibrations, but the repo's own Heston
Monte Carlo uses full-truncation Euler, which is *known* to be biased in exactly that
regime. The MC is only used as a validation cross-check at parameters where Feller holds
comfortably, so nothing in the results depends on it — but if anyone reused `mc_price` at
the calibrated parameters they would get a biased answer, and only a docstring warns them.
An Andersen QE scheme would fix it properly.

### G. No performance work
The calibration takes ~90 seconds for four seeds over 468 quotes. Most of that is 512-node
Gauss-Legendre quadrature evaluated inside `least_squares`' finite-difference Jacobian.
Analytic gradients of the characteristic function, or the COS method instead of direct
inversion, would be 10-50x faster. Irrelevant for one surface; a blocker for calibrating
every day across a universe.

### H. The risk-free curve is Treasury, not OIS
A few basis points at these maturities, far inside the bid-ask. But it is visibly the
reason the implied dividend yield comes out at ~0.3% against SPY's actual ~1.1%, so it is
not invisible in the output. The README says so.

### I. `surface_grid` interpolates linearly in total variance
Linear interpolation in `k` is safe and monotone-preserving but not smooth, so the 3-D
surface has visible kinks and the calendar-arbitrage check inherits a piecewise-linear
view of the surface. A proper parameterisation (SVI, or a monotone spline in total
variance) would be smoother and would let the arbitrage check test the *fitted* surface
rather than the interpolant. SVI is the obvious next module.

### J. Three uncovered branches
Coverage is 99% with branch coverage on. The remainder is defensive: a `slope >= 0` guard
in the parity regression that would mean parity slopes upward in strike, and two
short-circuit branches in the Heston integration limit. They are marked `pragma: no cover`
where they are genuinely unreachable and left uncovered where they are merely very
unlikely.
