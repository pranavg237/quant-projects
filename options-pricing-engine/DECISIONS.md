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

**Second-order Greeks included (vanna, volga, charm, dual delta).** Vanna and volga are
what a vol trader actually risk-manages; dual delta is `dV/dK`, whose negative is the
risk-neutral CDF of `S_T` and therefore the quantity a surface must keep monotone for the
implied density to stay non-negative. It is used in the surface sanity checks.

---

## Binomial trees

**Three parameterisations: CRR, Jarrow-Rudd, Leisen-Reimer.** CRR is the reference
everyone knows and shows the classic `O(1/n)` sawtooth convergence. Leisen-Reimer inverts
the Peizer-Pratt normal approximation so the tree is centred on the strike; it converges
`O(1/n^2)` with no oscillation and reaches 1e-5 in ~100 steps where CRR needs ~10^5.
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
alone* (SE 0.0175) to **58x better than plain MC** (SE 0.0043). Numbers are in the README.

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
for the given moneyness, i.e. the best-conditioned starting point. On a synthetic smile
this gives ~95% Newton steps and a worst case of 14 iterations.

**Convergence is on the price residual, so the *vol* accuracy is vega-dependent.** For
deep-OTM short-dated strikes a 1e-10 price tolerance can still leave ~1e-6 of vol error.
That is inherent to the inversion, not a solver defect, and the `ImpliedVolResult`
diagnostics expose it.

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
algebra slip cannot hide in both.

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

**Coverage is 99% of statements with branch coverage on.** `plotting.py` is excluded from
the target (asserting on pixels is brittle) but still has smoke tests, because its figures
are in the README and a silent breakage would ship.
