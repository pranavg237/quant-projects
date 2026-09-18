# Design decisions

Running log of the non-obvious choices, and why.

---

## Two engines, not one

**The idealised Avellaneda-Stoikov world and a full order book are both implemented, and
the same policies run in both.** This is the central design choice of the repo.

The closed-form A-S solution is *optimal* only in its own model: an arithmetic-Brownian
mid, no book, and fills arriving as a Poisson process with intensity `A exp(-kappa*delta)`.
That is the only setting in which the implementation can be validated against something
external -- the paper's Table 1 -- so it exists. But it is also a setting where the maker
always gets filled if a fill arrives, regardless of who else is quoting, which is exactly
the assumption that flatters every market-making backtest ever written.

The book engine removes it. Fills come out of a real matching engine, so they depend on
queue position, and a fraction of the order flow is informed. Running both and reporting
the gap is more honest than picking one.

## Prices are integers

Ticks, not currency, everywhere inside the book. A matching engine sorts and compares
prices constantly; doing that in floating point invites the bug where two orders that
should be at the same level land in different levels because `0.1 + 0.2 != 0.3`.
Conversion happens once, at the boundary.

## Self-trade prevention steps *past* an order and keeps going

The first implementation stopped matching at a price level that contained only the
aggressor's own orders. That silently under-fills any participant whose own quote sits at
the touch -- which is a market maker, essentially always. The fix walks to the next price
level instead, which is also what real venue STP does. There is a test for exactly this.

## `Side` is an `IntEnum` with values +1 / -1

So `side * size` is a signed quantity directly, and the same arithmetic works for
inventory, PnL and impact without a branch. Halves the places a sign can be wrong.

## Comparisons are at a matched average spread

Every policy in every comparison quotes the same *average* width, set from the A-S optimal
spread. Without this control the "better" strategy is simply whichever one was configured
to quote wider, and the comparison measures nothing. Avellaneda and Stoikov impose the same
control in their own paper.

## The naive benchmark has a position limit

Comparing A-S against a maker with *no* inventory control at all is too easy a win: that
maker's position is an unbounded random walk. A hard position limit is what a simple desk
actually runs, so `InventoryLimitPolicy` is the third policy in every comparison. It turns
out to capture most of the Sharpe benefit -- and A-S still beats it, significantly, which
is a much more interesting result than beating the straw man.

## Common random numbers, and paired tests

Run `i` of every policy faces the same price path and the same order flow. The difference
between policies is then almost entirely the strategy rather than the draw, and the paired
t-test on per-run differences has far lower variance than comparing two independent means.
It costs nothing and it is the cheapest way to make a simulation study trustworthy.

## The fill probability is exact, not `lambda * dt`

The paper's discretisation uses `lambda*dt`; the true probability of at least one Poisson
arrival in `dt` is `1 - exp(-lambda*dt)`. They agree to `O(dt^2)`, but at the paper's own
parameters (`lambda ~ 45`, `dt = 0.005`) the linear form overstates the fill rate by 10%,
and it exceeds 1 outright for tight quotes on a coarse grid. The default is exact;
`fill_model="linear"` exists so the published table can be reproduced, and it is what makes
the reproduction land on 65.7 against 65.0 rather than 57.9.

## Model parameters are estimated from the book, not chosen

`kappa` is measured in *inverse price units*. Copying the paper's `kappa = 1.5` (for an
asset trading at 100 with a 1.5-wide spread) onto a book whose spread is 2 ticks produces
quotes 115 ticks wide, and the maker never trades. That was the first version's behaviour.

So `sigma`, `A` and `kappa` are estimated from the simulated market by probing it: post at
a grid of distances, measure fills per unit time, regress `log(lambda)` on distance. Only
`gamma` and the risk horizon stay free, because those are preferences rather than
properties of the market -- and they are what the sensitivity analysis sweeps.

The probe re-posts every step, so it never ages into queue priority. That makes the
estimate of `A` deliberately **conservative**: a maker that leaves an order to rest will do
better than the probe suggests.

## The noise depth profile is humped, not exponential

Real limit order books are thin at the touch, thickest a few ticks out, and thin again
beyond. An exponential distribution puts the most liquidity right at the best price and
leaves the deep book empty, so a probe ten ticks out still fills at a fifth of the touch
rate, the measured fill curve is nearly flat, and a model fitted to it concludes that
quoting very wide is optimal. Switching to a Gamma with shape 2.5 raised the fitted `kappa`
by ~35% and the log-linear fit's R-squared from 0.81 to 0.86. Both numbers are in a test.

## Informed impact is released gradually

The first version applied an informed trade's permanent impact instantaneously. The result
was a **flat markout curve**: the whole move had already happened by the first markout
horizon, so the chart -- the standard desk measure of adverse selection -- carried no shape
information at all. `info_impact_speed` now releases about 5% of the outstanding impact per
step, so price discovery takes ~20 steps and the markout curve builds the way a real one
does. Setting it to 1.0 recovers the old behaviour, and a test pins both.

## The fill's reference mid is taken *before* the step's price move

A fill is recorded against the mid the maker quoted against, not the mid after informed
impact and diffusion have been applied. Recording the post-move mid would fold the
adverse-selection loss into "spread captured" and leave the first markout horizon blind to
informed impact -- i.e. it would make every strategy look better at exactly the thing this
repo is measuring.

## PnL is decomposed, and the decomposition is checked

Terminal wealth splits exactly into spread capture (`sum dq*(S_fill - price)`) and
inventory PnL (`sum dq*(S_T - S_fill)`). The identity is exact, and the code reports the
reconciliation error against simulated wealth so a bookkeeping slip cannot pass silently.
It is 0.0 to machine precision in every run.

## Sharpe is computed across runs, not within a path

A market maker's within-path PnL increments are dominated by mark-to-market swings on
inventory and are strongly autocorrelated, so an intraday Sharpe is close to meaningless.
Across independent runs each observation is one session and the ratio is interpretable.
It is deliberately **not annualised**: doing so would need an assumption about how many
independent sessions a year holds, which is the unearned scaling factor that makes
market-making Sharpe ratios look absurd in print.

## Stationary horizon mode

The published finite-horizon model lets `T - t` shrink to zero, so both the inventory skew
and the risk premium vanish as the session ends. Measured: the standard deviation of
inventory grows from 1.13 mid-session to 3.01 at the bell -- control collapses exactly when
the paper measures it. A real desk has no terminal time, so `HorizonMode.STATIONARY` freezes
the effective horizon at a constant and the strategy becomes time-homogeneous. Inventory
std then stays flat at ~1.2 throughout. Both modes are implemented so the difference can be
measured rather than argued about.

## Tooling

`mypy` runs without a `python_version` pin, for the same numpy-stub reason as the options
project. `plotting.py` is excluded from the coverage target but still has smoke tests,
because its figures are in the README.
