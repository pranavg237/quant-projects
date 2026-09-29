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
A price tolerance of 1e-10 sounds strict. On a five-day 10%-OTM call at 15% vol (vega
2.1e-6) it leaves 4.7e-5 of *volatility* error. The solver now also requires the vol
uncertainty or the bracket width to be below 1e-9; on the round-trip stress grid in
`results/validation.md` the worst vol error is 2.5e-10 for quotes with vega >= 1e-6 (the
before/after figures quoted in an earlier version of this item were not reproducible from
a committed output and have been dropped). The original would not have been visibly wrong
on any chart; it would just have quietly added noise to the wings of every surface.

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

### 7. The Greeks check was thinner than the README claimed — **FIXED**
"Greeks verified against central finite differences" was true at exactly two parameter
points. Charm was checked for calls only, vanna/volga/charm were differences of the
*analytic* delta and vega rather than of the price, and nothing was closer than 0.7 years
to expiry -- where step size actually matters. `greeks_check` now sweeps calls and puts,
1 day to 3 years, +/-3 standard deviations, two vols, differencing the price only, and
commits the worst errors per Greek to `results/greeks_fd.md`. No formula turned out to be
wrong. The step-size analysis did find that the textbook `eps^(1/3)` / `eps^(1/4)` steps
lose ~10x on one-day gamma and volga, because the price's round-off is `eps * S`, not
`eps * V`. The README also said "10 Greeks"; there are nine, plus the price.

### 8. The call/put gap at the forward was blamed on quote noise — **MEASURED, not yet fixed**
The README said call and put vols meet at the forward with a ~0.5 vol point gap that was
"a hard floor from free data". The parity analysis shows it is mostly American exercise:
deep-ITM American puts in the forward-fitting window pull the parity forward down, by 2bp
at a month and 85bp at 21 months. Pricing the early-exercise premium on a tree cuts the
scatter of the per-strike forwards 2.4-5.6x beyond six months, cuts in-window parity
violations beyond the spread from 44.5% to 10.6%, and closes the call/put gap at 637 days
from -1.08 to +0.18 vol points. The surface and Heston numbers still use the biased
forward (see B).

---

## Open weaknesses

### A. One snapshot, one underlying
Everything here is SPY on 2026-09-18. The single most damning test of a Heston
implementation is **parameter stability across days**: calibrate on consecutive dates and
see how far `(v0, kappa, theta, xi, rho)` moves when the surface barely does. Practitioners
report it moves a lot. Measuring that needs a time series of chains, which needs either a
paid feed or weeks of daily collection. This is the largest gap in the repo and it is
stated in the README rather than papered over.

### B. American exercise is ignored when inverting SPY quotes — **now measured**
SPY options are American. The surface treats them as European. For the OTM quotes the
surface inverts, the early-exercise premium is small. The first version of this item
stopped there, and missed the bigger effect: the *forward* is fitted by parity over
strikes within 10% of spot, which includes in-the-money puts whose early-exercise premium
at 4% rates is roughly `r K tau` -- dollars, not cents, at long maturities. `parity.py`
now prices that premium (American minus European on one CRR lattice) and re-solves the
forward: +2bp at a month, +85bp at 21 months (item 8). Not yet done: using the adjusted
forward and de-Americanised OTM prices in `build_surface`, and re-running the calibration.
That changes every downstream number and deserves its own review. The tree also uses a
continuous dividend yield, which cannot represent the pre-ex-date exercise of deep-ITM
calls under SPY's discrete dividends; the pipeline would still be materially wrong on a
single stock around a dividend.

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
benchmarks sit at 0.96 and 0.95. Nothing overfits, which is what five parameters against 234
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
The calibration takes ~105-110 seconds for four seeds over 468 quotes (106.8 s in the
committed run, `results/results.json` `heston.seconds`). Most of that is 512-node
Gauss-Legendre quadrature evaluated inside `least_squares`' finite-difference Jacobian.
Analytic gradients of the characteristic function, or the COS method instead of direct
inversion, would be 10-50x faster. Irrelevant for one surface; a blocker for calibrating
every day across a universe.

### H. The risk-free curve is Treasury, not OIS
A few basis points at these maturities, far inside the bid-ask. The first version of this
item called it "visibly the reason" the implied dividend yield comes out at ~0.3% against
SPY's actual ~1.1%. That was asserted, not measured, and the American adjustment makes the
gap wider (implied q about -0.1% to -0.2% beyond six months). Parity only identifies the
carry `r - q`; splitting it needs a dividend forecast and the funding rate, neither of
which is in the data. Funding is the leading hypothesis, stated as a hypothesis.

### K. Stale quotes in the snapshot
72 quotes are offered below intrinsic (on American options, an instant arbitrage), 100
adjacent-strike pairs are not monotone, and 128 call/put pairs breach the American bound
`C - P <= S - K e^(-r tau)` -- nearly all deep in the money. They are what drive the
out-of-sample parity violations (33% of pairs, unchanged by the American adjustment). The
surface avoids most of them by using OTM quotes only, but nothing filters them out of the
forward fit or the butterfly check explicitly.

### I. `surface_grid` interpolates linearly in total variance
Linear interpolation in `k` is safe and monotone-preserving but not smooth, so the 3-D
surface has visible kinks and the calendar-arbitrage check inherits a piecewise-linear
view of the surface. A proper parameterisation (SVI, or a monotone spline in total
variance) would be smoother and would let the arbitrage check test the *fitted* surface
rather than the interpolant. SVI is the obvious next module.

### J. A few uncovered branches
Coverage is 98% with branch coverage on (99% before the parity and Greeks-check modules
were added; their uncovered lines are empty-input guards). The rest is defensive: a `slope >= 0` guard
in the parity regression that would mean parity slopes upward in strike, and two
short-circuit branches in the Heston integration limit. They are marked `pragma: no cover`
where they are genuinely unreachable and left uncovered where they are merely very
unlikely.
