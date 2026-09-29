# Design decisions

Running log of the non-obvious choices made while building this engine, and why.

---

## Numerics and API shape

**Everything is vectorised over NumPy arrays, and pricers are total functions.**
Real option chains have thousands of rows and a meaningful fraction of them are
degenerate (zero time value, expiry today, crossed quotes). Raising on those would force
every caller to write the same `try/except` loop. Instead, degenerate inputs return the
correct limiting value: `tau <= 0` or `sigma <= 0` collapses to the discounted forward
intrinsic `max(phi*(S e^{-q tau} - K e^{-r tau}), 0)`, which is the true no-uncertainty
price, not a fudge. Implied vol returns `nan` for quotes outside the no-arbitrage bounds.

**Greeks are quoted in "mathematical" units, not trader units.** Vega is
`dV/dsigma` per *one* unit of vol (so 100 vol points), theta is per *year*. Conversions to
per-vol-point and per-day are one division and belong in the reporting layer, not baked
into the model. This is stated in every docstring because getting it wrong by 100x is the
most common Greeks bug.

**`StrEnum` for `OptionType` / `ExerciseStyle` / `TreeMethod`.** Callers can pass
`"call"` or `OptionType.CALL` interchangeably, which keeps notebook use ergonomic while
still giving mypy something to check.

**`OptionType.sign` (+1 / -1).** Nearly every Black-Scholes formula can be written once
with `phi = +/-1` instead of twice with a call branch and a put branch. Halves the surface
area for sign errors and means the put path is exercised by the same tests as the call.

---

## Black-Scholes

**`d1_d2` returns `+/-inf` in degenerate cells rather than `nan`.** `N(+inf) = 1` and
`N(-inf) = 0` reproduce exactly the intrinsic-value limit, so the main price formula needs
no special-casing beyond the final `maximum(., 0)`.

**Second-order Greeks included (vanna, volga, charm), plus dual delta.** Nine Greeks in
all (an earlier README said ten; it counted the price). Vanna and volga are what a vol
trader actually risk-manages. Dual delta is `dV/dK`: undiscounted, the put's is the
risk-neutral CDF `Q(S_T < K)` and minus the call's is `Q(S_T > K)`, which is why call
prices must fall and be convex in strike. (An earlier version of this entry said the
negative was the CDF and that it was used in the surface checks; the first is only true
for the call's survival function, and the surface's butterfly check tests convexity of
prices directly rather than calling `dual_delta`.)

---

## Binomial trees

**Three parameterisations: CRR, Jarrow-Rudd, Leisen-Reimer.** CRR is the reference
everyone knows and shows the classic `O(1/n)` sawtooth convergence. Leisen-Reimer inverts
the Peizer-Pratt normal approximation so the tree is centred on the strike; it converges
`O(1/n^2)` with no oscillation: at 101 steps its error on the benchmark call is 3.4e-5,
against 8.8e-4 for CRR at 2,001 steps (`results/results.json`).
Having both in the repo makes the convergence chart in the README actually interesting
rather than a formality.

**American exercise by `max(continuation, intrinsic)` at every node, recomputing the node
spot lattice each step.** Slightly more arithmetic than caching the full lattice, but it
keeps memory at `O(n)` instead of `O(n^2)`, which is what lets the tests run CRR to 5,000
steps.

**Early-exercise boundary is reported as `nan` where no node exercises.** At small `t` a
CRR tree spans only `S_0 d^k` to `S_0 u^k`, so if the true boundary is below the lowest
node the honest answer is "not resolved", not an extrapolation.

---

## Monte Carlo

**European payoffs are priced from one-shot exact lognormal draws, not time-stepped.**
There is no discretisation bias to discuss, so the error bars are purely statistical and
the convergence plot is a clean `O(n^{-1/2})`.

**Antithetic pairs are folded into a single sample *before* the standard error is
computed.** Treating the `Z` and `-Z` legs as `2n` independent draws understates the
standard error by up to `sqrt(2)` and is a very common bug in teaching code.

**The control-variate coefficient is fitted on the folded samples, not the raw legs.**
This was a real bug in the first version: fitting `beta` on the legs minimises the
variance of an estimator we do not report. Fixing it changed the combined
antithetic+control variance reduction on the benchmark case from *worse than the control
alone* (SE 0.0175) to **57x better than plain MC** (SE 0.0043; 57.4x in `results/results.json`). Numbers are in the README.

**Arithmetic Asian options use the discretely monitored geometric Asian as a control.**
The geometric average is lognormal so it has a closed form; its payoff correlates with the
arithmetic payoff at ~0.999, and the standard error drops ~575x. This is a far better
demonstration of what a control variate is worth than the `S_T` control on a vanilla.

**Estimating `beta` in-sample introduces an `O(1/n)` bias.** It is negligible at the path
counts used and is standard practice, but `beta` can be passed explicitly to remove it.
Noted rather than hidden.

---

## Implied volatility

**Safeguarded Newton (`rtsafe`), not bare Newton.** Bare Newton-Raphson with analytic vega
is quadratically convergent and *globally unsafe*: vega collapses for deep ITM/OTM strikes
and short expiries, which is exactly where a real chain has most of its rows. The
implementation keeps a bracket at all times, takes a Newton step when it lands inside the
bracket and is at least halving the interval, and bisects otherwise. It cannot diverge.

**Manaster-Koehler seed.** `sigma_0 = sqrt(2|ln(F/K)|/tau)` is the vol that maximises vega
for the given moneyness, i.e. the best-conditioned starting point. On the round-trip
stress grid in `results/validation.md` no quote with vega >= 1e-6 needs more than 16
iterations.

**Convergence is required in volatility space, not just on the price residual.** A 1e-10
price tolerance alone leaves `tol / vega` of vol error -- 4.7e-5 for a five-day 10%-OTM
call at 15% vol. So a quote only counts as converged when the price residual is below
`tol` *and* the implied vol uncertainty `|residual| / vega` is below `vol_tol = 1e-9`, or
when the bracket itself is narrower than `vol_tol`. What remains is round-off in the
*price*: a price is only known to about `eps * price`, which is `eps * price / vega` of vol.
For a deep in-the-money 3-year call with vega 2e-8 that floor is 5e-7, and the round-trip
grid in `results/validation.md` hits it; with vega >= 1e-6 the worst error is 2.5e-10.
The `ImpliedVolResult` diagnostics expose iterations and convergence per quote.

---

## Heston

**"Little trap" (Albrecher et al. 2007) formulation of the characteristic function.**
Heston's original grouping of the `g` term crosses a branch cut of the complex logarithm
for longer maturities and returns silently wrong prices. Using `g = (beta - d)/(beta + d)`
keeps `|g| <= 1` so the log stays on the principal sheet.

**Two independent pricing formulas: Gil-Pelaez (`P1`/`P2`) and Lewis' single integral.**
They are algebraically equivalent, so agreement between them (checked to <1e-6 in the
tests) catches quadrature and sign errors that testing a formula against itself cannot.
A third, completely separate check is the full-truncation Euler Monte Carlo, which agrees
with the Fourier price to within one standard error across strikes.

**The Fourier upper limit is chosen from `tau` and the parameters, not fixed.**
`|psi(u)|` decays like `exp(-v_bar tau u^2 / 2)`, so short expiries need a much wider
domain. A hard-coded `u_max = 200` is accurate at 1y and wrong at 1 week. The adaptive
limit is `clip(40/sqrt(v_bar*tau), 100, 5000)`.

**Default `n_quad = 512` Gauss-Legendre nodes.** 256 is enough for Gil-Pelaez but Lewis
needs 512 at short expiries; 512 costs ~10ms for a whole expiry slice, which is
irrelevant next to the calibration loop, so both formulations get to be accurate.

**Puts are obtained from calls by put-call parity, not by a second integral.** Parity is
model-free, so this is exact and halves the quadrature cost.

---

## Put-call parity

**Say what is being tested, because the forward comes from the same quotes.** The pipeline
infers each expiry's forward as the median of `K + (C - P)/D` over strikes within 10% of
spot, so the median parity residual inside that window is zero by construction. Checking
parity against that forward tests the *shape*, not the level: one number per expiry must
explain 30-230 strikes, the residuals must be flat in strike and inside each pair's
bid-ask range, and strikes outside the window are out of sample. The *level* -- whether
the implied carry matches SPY's dividends and funding -- needs a dividend forecast and a
funding rate the data does not contain, and is not claimed.

**Pairs come from the raw chain, not the cleaned one.** Cleaning drops mids under 5 cents
and spreads over 40%, which removes exactly the deep-ITM legs where parity is most
informative (and most often stale). Whether both legs survived cleaning is recorded.

**A violation is "beyond the spread", not "non-zero".** A pair violates parity only if the
theoretical synthetic forward lies outside `[C_bid - P_ask, C_ask - P_bid]`, i.e. you could
actually trade against it at the quoted prices. Mid-price residuals are reported as well.

**American exercise is priced, not waved away.** The residuals on SPY fall with strike in a
way noise does not: at 4% rates a deep-ITM American put is worth about intrinsic while a
European one is worth about `r K tau` less. `early_exercise_premia` prices American minus
European on the *same* CRR lattice, so the lattice's own error cancels, at the surface vol
for each strike. The adjusted forward then solves parity with that premium removed. Two
things keep this honest: the adjustment has no parameter fitted to the residuals (the vol
comes from the OTM surface), and a test applies it to a chain that really is European and
requires it to make the fit *worse*.

**Wired into the surface, and the default.** `build_surface(exercise="american")` removes
each quote's early-exercise premium and fits the forward to de-Americanised parity;
`exercise="european"` is the old pipeline. The run calibrates both and writes
`results/exercise_comparison.md`. The design choices:

* *The circularity is a fixed point, not a guess.* The premium depends on the vol being
  solved for. Each quote solves `sigma = BS^-1(V_mkt - e(sigma))` by iteration from the
  uncorrected vol. The map's slope is the premium's vega over the option's vega; the
  largest seen on SPY is 0.11, so it converges geometrically in at most 8 iterations to
  1e-8. The simpler alternative, pricing the premium once at the uncorrected vol, is one
  iteration of the same map: its error is the whole correction times the map's slope,
  about a tenth of the correction, and the correction is largest (dollars of premium) on
  exactly the long-dated puts near the forward that pin the long end. Iterating costs
  seconds. The forward is a second fixed point (forward -> surface -> forward, 5
  passes to 6e-8), because the ITM parity legs need vols from the surface.
* *Leisen-Reimer, not CRR, for the premium.* On a fixed CRR lattice the premium has a
  sawtooth in strike (each strike sits differently relative to the nodes), about five
  times the true curvature on \$1 strikes. Leisen-Reimer centres each tree on its own
  strike; its premium is smooth and closer to a 2,001-step answer at 201 steps. Doubling
  the steps moves no surface vol by more than 0.01 vol points.
* *Bid and ask carry the mid's premium.* The premium moves with vol by far less than a
  cent across a quote's bid-ask, so this keeps the tradeable spread exactly as quoted,
  which the spread-aware butterfly test relies on.
* *`mid`/`bid`/`ask` on the surface are European-equivalent prices*, with the quote as
  observed kept in `mid_market`. The calibration compares Heston, a European model, with
  `mid`; leaving the American price there would have been the obvious bug.
* *Controls.* A synthetic American chain is recovered exactly (forward to 1e-6, smile to
  1e-7) while the old pipeline misses by 1.6 vol points. With `r = q = 0` early exercise is
  never optimal and the correction is < 1e-10. On a European chain with carry the
  correction must make recovery worse. Given the corrected surface, the parity module's
  independent American forward reproduces the surface's to 6e-8.

**Why the corrected surface is the default.** The evidence is about the *data*, not the
fit. On SPY the call/put vol gap at the forward goes from -0.24 / -0.26 vol points at 73 /
272 days to -0.01 / +0.03, and the European-parity forward's per-strike scatter falls
1.9-6.9x beyond six months once the premium is priced. SPY options *are* American, so the
European surface is a known misspecification of every quote. The Heston RMSE improvement
(2.51 -> 2.26) is *not* part of the case: cross-scoring shows the new parameters fit the
old wing quotes just as well, so it is a different compromise, not better data. Against
it: the gap overshoots to +0.20 / +0.37 at 15 / 21 months, and the self-consistent
parity check is slightly worse than the non-self-consistent one (in-window violations
10.6% -> 11.5%). Both point at what the lattice leaves out, not at a reason to go back to
a model that ignores exercise altogether.

**What it still leaves out.** A continuous dividend yield, so no pre-ex-date exercise of
deep-ITM calls under SPY's discrete dividends. With the implied yield not positive here,
the model prices every call premium at zero; the real ones would pull the long-dated
forward back down, which is the leading suspect for the overshoot. And the premium depends
on `r` itself, not only on the carry, so it inherits the Treasury-rate assumption.

**Stale quotes are identified by arbitrage, not by timestamps.** The snapshot has last-trade
times but no quote times. A quote offered below intrinsic (on an American option) or a call
bid above a lower strike's call ask cannot survive in a live market, so those two
model-free checks are what "stale" means here.

---

## Tooling

**`mypy` runs without a `python_version` pin.** numpy >= 2.3 ships stubs using PEP 695
`type` statements, which mypy refuses to parse when targeting < 3.12. The library source
itself stays 3.11-compatible (`StrEnum` is 3.11+).

**`plotting.py` is excluded from coverage.** It produces Matplotlib figures; asserting on
pixels is brittle and asserting that it "did not raise" is coverage theatre. The
data-producing functions it calls are all tested directly.

---

## Testing

**The synthetic chain fixture is the backbone of the test suite.** `tests/conftest.py`
builds an option chain from a *known* smile with a known rate and dividend yield, then the
tests push it through the real cleaning, forward-extraction and inversion pipeline and
check the original volatilities come back to 1e-4. Unit-testing each function separately
would not catch a sign error in how they compose.

**Greeks are tested against central finite differences of the price function**, not against
a second copy of the analytic formula. The two derivations share only the normal CDF, so an
algebra slip cannot hide in both. The first version checked one parameter point per Greek,
checked charm for calls only, and took vanna, volga and charm as differences of the
*analytic* delta and vega (so an error in delta would have been inherited). `greeks_check`
now differences only the price, including the mixed partials, over a grid built for where
Greeks go wrong:

* *Moneyness in standard deviations* (`z = ln(K/F)/(sigma sqrt(tau))` from -3 to +3), not a
  fixed percentage of spot. A one-day option 20% out of the money is dozens of standard
  deviations out and every Greek underflows to zero, so "agreement" there is vacuous.
* *Down to one day to expiry*, both option types, 10% and 40% vol, non-zero `r` and `q`.
* *Steps scaled twice.* Each input is bumped by a multiple of the distance over which the
  price actually bends (`S sigma sqrt(tau)` for spot, `sigma` for vol, `tau` for time).
  And the multiple comes from balancing truncation against the price formula's own
  round-off, which is about `eps * S` (the price is a difference of two terms of size S),
  i.e. `eps / (sigma sqrt(tau))` relative to the natural price scale. The textbook
  `eps^(1/3)`, `eps^(1/4)` under-steps short-dated options and loses about an order of
  magnitude on one-day gamma and volga; `results/greeks_fd.md` shows both.
* *Relative error with a floor* of 1e-3 of the Greek's largest value across the smile, so a
  Greek that crosses zero (volga near the money) is not reported as a huge relative error.
* *The check is checked.* Tests plant the classic unit bugs (vega per vol point, theta per
  day), a sign flip, and a put-only slip in charm, and require each to be caught.

**Heston is validated three independent ways** -- Black-Scholes limit, a second quadrature
(Lewis), and a Monte Carlo scheme. A Fourier pricer checked only against itself is not
checked at all.

**Network paths are tested against a stub `yfinance` injected into `sys.modules`.** The
download, cache-fallback and rate-curve paths are exactly the ones that break silently in
production (a renamed Yahoo column, a 404 expiry, an offline laptop) and exactly the ones a
live-network test cannot pin down, because it passes or fails for reasons outside the repo.

**Three tests encode findings rather than expectations**, because the naive assertion was
wrong and the reason is interesting:

* *Jarrow-Rudd is not exactly risk-neutral.* It fixes `p = 1/2` and solves for `(u, d)`, so
  `p*u + (1-p)*d` misses the growth factor by `sigma^4 dt^2 / 12` per step. The test pins
  that quantity rather than loosening a tolerance. CRR and Leisen-Reimer solve for `p` and
  are exact to machine precision.
* *The CRR early-exercise boundary alternates with step parity.* A lattice at step `k` has
  nodes only at `S0 u^(2j-k)`, so odd and even steps sit on interleaved grids. Within one
  parity the boundary is monotone.
* *The SPY smile is not monotone in strike.* The call wing turns back up past about
  `k = +0.07`, so "ATM vol exceeds everything beyond `k = +0.1`" is false. Skew is tested
  as an interpolated 5%-moneyness risk reversal instead, which is what the claim actually
  means.

**One vol per expiry only beats one global vol in the metric both are fitted in.** Both are
vega-weighted means, so the nesting guarantee holds for the vega-weighted squared error, not
for plain unweighted RMSE -- on a synthetic Heston surface the unweighted ordering inverts
by a hair (2.811% vs 2.799%). The test asserts the guarantee that exists.

**Coverage is 98% of statements with branch coverage on.** `plotting.py` is excluded from
the target (asserting on pixels is brittle) but still has smoke tests, because its figures
are in the README and a silent breakage would ship.
