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
26% alone would have been alarming and meaningless. The American correction then made the
point again: it moved the zero-tolerance count to 37.6% (746/1986) while the tradeable count
*fell* to 2.1% (41). 209 of the 229 extra triples are runs of identical tick-quantised mids
whose butterfly is exactly zero, tipped negative by millionths of a dollar; at a tenth of a
cent the counts are 497 before and 501 after (`results/exercise_comparison.md`).

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

### 8. The call/put gap at the forward was blamed on quote noise — **FIXED, with an overshoot**
The README said call and put vols meet at the forward with a ~0.5 vol point gap that was
"a hard floor from free data". The parity analysis showed it is mostly American exercise:
deep-ITM American puts in the forward-fitting window pull the parity forward down, by 2bp
at a month and 86bp at 21 months. The surface now removes every quote's early-exercise
premium and uses the de-Americanised parity forward (item B). On the surface itself the
call/put gap at the forward goes from -0.24 / -0.26 / -0.51 / -1.08 vol points at 73 / 272
/ 455 / 637 days to -0.01 / +0.03 / +0.20 / +0.37. So it is fixed out to nine months (within ±0.03, apart from +0.07 at 104 days on 3-4 strikes, which was +0.06 before) and
over-corrected at 15-21 months, on three strikes per expiry, against a bid-ask band of
0.07-0.09. The leading suspect is the tree's continuous dividend yield (below); it is not
tested.

---

## Open weaknesses

### A. One snapshot, one underlying
Everything here is SPY on 2026-09-18. The single most damning test of a Heston
implementation is **parameter stability across days**: calibrate on consecutive dates and
see how far `(v0, kappa, theta, xi, rho)` moves when the surface barely does. Practitioners
report it moves a lot. Measuring that needs a time series of chains, which needs either a
paid feed or weeks of daily collection. This is the largest gap in the repo and it is
stated in the README rather than papered over.

### B. American exercise in the surface — **FIXED for the continuous-yield model; discrete dividends open**
SPY options are American; the surface used to treat them as European. The first version
of this item said the premium on the OTM quotes the surface inverts is small. That is
true short-dated and wrong long-dated: the 21-month puts near the forward carry up to
\$11.81 (median \$0.75), because the ATM-forward strike is 8% in the money against spot.
The bigger effect was the *forward*, fitted by parity over strikes that include ITM puts
whose premium is roughly `r K tau`.

`build_surface` now de-Americanises every quote (201-step Leisen-Reimer lattice, a
per-quote fixed point for the vol-premium circularity, contraction <= 0.11, <= 8
iterations) and iterates the forward with the surface (5 passes to 6e-8). The old
pipeline is `exercise="european"` and both are calibrated in every run
(`results/exercise_comparison.md`). Heston RMSE 2.51 -> 2.26, holdout 2.38 -> 2.14,
body 1.09 -> 1.10, short-dated put wing 5.92 -> 5.26, one-vol-per-expiry Black-Scholes
10.61 -> 10.66. The Heston gain looked suspicious, because the wing quotes barely moved
(at most 0.018 vol points), so it was checked by cross-scoring: the new parameters score
5.26 on the *old* wing quotes too. The gain is a different compromise across maturities
(a lower, flatter long end lets the optimiser raise xi from 1.96 to 2.08), not a better
fit to changed data.

Two findings from doing it. (1) The first run priced the premium on CRR, whose premium
has a sawtooth in strike about five times the true curvature on \$1 strikes; Leisen-Reimer
removes it and is now the default. That was blamed at the time for a jump in the
zero-tolerance butterfly count, and wrongly: the jump survived the switch and is the
tick-run effect in item 4. (2) Pricing the parity module's American theory at the
de-Americanised vols, which is what self-consistency requires, fits parity slightly
*worse* than the earlier run that used the uncorrected vols. In-window violations rise
from 10.6% to 11.5%, and the long-dated forward scatter from 1.38/1.76 to 1.63/2.18 at
15/21 months.

Still open: the lattice uses a continuous dividend yield, and the implied yield here is
not positive, so every call premium is zero. SPY's discrete dividends give deep-ITM calls a
pre-ex-date premium the model cannot see, which would pull the forward back down. That is
the leading suspect for the long-dated overshoot in item 8. The premium also depends on
`r` itself, not just the carry, so it inherits the Treasury-for-funding assumption (H).
The pipeline would still be materially wrong on a single stock around a dividend.

### C. No confidence intervals on the calibrated parameters — **FIXED, with a caveat**
Now computed from the Jacobian at the optimum. The relative standard errors are small
(0.9%–4.2% on the corrected surface), and that *understates* the uncertainty because the
calculation assumes independent residuals while adjacent strikes are strongly dependent.
The genuinely useful output turned out to be the correlation matrix: `corr(kappa, xi) =
+0.84` and `corr(kappa, theta) = -0.79` (+0.83 / -0.79 on the uncorrected surface). My
guess before measuring was "above 0.9"; the measured value is 0.84, so kappa and xi are
strongly but not degenerately linked. It remains a
*local* quantity — it describes the basin the optimiser landed in, not global uncertainty
on a multimodal objective.

### D. Model comparison is in-sample — **FIXED**
Now cross-validated: fit on alternate strikes within each expiry, score on the rest.
Heston goes 2.30 in-sample to 2.14 out-of-sample (ratio 0.93) on the corrected surface
(2.55 to 2.38 before); both Black-Scholes benchmarks sit at 0.96 and 0.95. Nothing overfits, which is what five parameters against 234
training quotes should do — but it is now measured rather than assumed. Note this tests
interpolation across strikes, **not** extrapolation in maturity, which is the harder
question and is still untested (and is really item A in disguise).

### E. The cleaning thresholds are not tuned, just chosen
`max_relative_spread = 0.40`, `min_price = 0.05` and the 5-day minimum are defensible
conventions, not optimised values. A sensitivity analysis — how much do the calibrated
parameters move if the spread filter goes from 20% to 60%? — would say whether the results
are robust to them. Not run.

### F. The Feller condition is violated and only noted
`2·kappa·theta/xi² = 0.13` (0.16 before the American correction). That is fine for the characteristic function (the variance
process just touches zero) and typical of equity calibrations, but the repo's own Heston
Monte Carlo uses full-truncation Euler, which is *known* to be biased in exactly that
regime. The MC is only used as a validation cross-check at parameters where Feller holds
comfortably, so nothing in the results depends on it — but if anyone reused `mc_price` at
the calibrated parameters they would get a biased answer, and only a docstring warns them.
An Andersen QE scheme would fix it properly.

### G. No performance work
The calibration takes ~105-110 seconds for four seeds over 468 quotes (109.5 s in the
committed run, `results/results.json` `heston.seconds`), and the pipeline now runs it
twice (both surfaces) plus the holdout, so a full run is ~9 minutes. Most of that is 512-node
Gauss-Legendre quadrature evaluated inside `least_squares`' finite-difference Jacobian.
Analytic gradients of the characteristic function, or the COS method instead of direct
inversion, would be 10-50x faster. Irrelevant for one surface; a blocker for calibrating
every day across a universe.

### H. The risk-free curve is Treasury, not OIS
A few basis points at these maturities, far inside the bid-ask. The first version of this
item called it "visibly the reason" the implied dividend yield comes out at ~0.3% against
SPY's actual ~1.1%. That was asserted, not measured, and the American adjustment, now in
the surface, makes the gap wider (implied q -0.13% to -0.19% beyond six months). Parity only identifies the
carry `r - q`; splitting it needs a dividend forecast and the funding rate, neither of
which is in the data. Funding is the leading hypothesis, stated as a hypothesis.

### K. Stale quotes in the snapshot
72 quotes are offered below intrinsic (on American options, an instant arbitrage), 100
adjacent-strike pairs are not monotone, and 128 call/put pairs breach the American bound
`C - P <= S - K e^(-r tau)` -- nearly all deep in the money. They are what drive the
out-of-sample parity violations (33.1% of pairs before the American adjustment, 34.1% after). The
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
