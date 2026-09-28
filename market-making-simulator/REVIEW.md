# Skeptical review

Written as if reviewing someone else's repo. Items marked **FIXED** were found and
addressed during development; the rest are open, with an honest reason.

---

## Defects found and fixed

### 1. Self-trade prevention stopped the match instead of stepping past — **FIXED**
The first matching engine halted when it reached a price level containing only the
aggressor's own orders. That silently under-fills any participant whose own quote sits at
the touch, which is a market maker essentially always. The engine now steps past and
continues to the next level, which is also what real venue STP does.

### 2. The maker threw away its queue position on every step — **FIXED**
`cancel_all` followed by a re-post, every requote step. The maker never accumulated any
time priority at all. Since the entire reason for building a matching engine is that queue
position is what a book-free model cannot represent, this was self-defeating. Quotes are
now only cancelled and replaced when the target price actually changes; fills rose ~60%.

### 3. Informed impact was instantaneous, making the markout curve flat — **FIXED**
Markout is *the* desk measure of adverse selection, and with instantaneous impact the whole
move has already happened by the first markout horizon, so the curve carried no shape
information at all. Impact is now released gradually and the curve builds over the
price-discovery window the way a real one does.

### 4. The fill's reference mid was recorded after the price move — **FIXED**
Which folded the adverse-selection loss into "spread captured" and left the first markout
horizon blind to informed impact. Every strategy looked better at exactly the thing being
measured. The reference is now the mid the maker quoted against.

### 5. Model parameters were hand-picked and produced a maker that never traded — **FIXED**
`kappa = 1.5` copied from the paper, on a book with a 2-tick spread, gives quotes 115 ticks
wide. `sigma`, `A` and `kappa` are now estimated from the simulated book by probing it.

### 6. An exponential noise-depth profile flattened the fill curve — **FIXED**
Exponential depth concentrates liquidity at the touch and leaves the deep book empty, so
probes far from the mid still fill and the fitted `kappa` comes out ~35% too small.
Switched to a humped Gamma profile, which is what real books look like.

### 7. The Monte Carlo fill probability used `lambda*dt` — **FIXED (and kept as an option)**
The exact Poisson probability `1 - exp(-lambda*dt)` is the default. The linear form is
retained because it is what reproduces the paper's published table, and the ~10% gap
between them is the explanation for a discrepancy that would otherwise look like a bug.

---

## Open weaknesses

### A. One market maker, no competition
There is exactly one strategic participant. Every other order is noise. In reality several
makers compete for the same queue, and the equilibrium spread is set by that competition
rather than by one monopolist's optimisation. The monopolistic markup term
`(2/gamma)ln(1+gamma/kappa)` is doing a lot of work here and would shrink sharply with a
second maker. Adding one is the single most valuable extension and is not done.

### B. Informed traders are exogenous and have a fixed impact
`info_impact_ticks` is a constant, and informed traders do not condition on the book. A
real informed trader trades more when the spread is tight and the book is deep, which makes
adverse selection *endogenous* to the maker's own quoting. That feedback loop is absent, so
the maker cannot make itself less adversely selected by quoting wider -- which is one of
the main levers a real desk has.

### C. Order size is fixed at one unit for the maker
Real market making is as much about *how much* to quote as where. Sizing is the natural
second control variable and Avellaneda-Stoikov says nothing about it; Guéant-Lehalle-
Fernandez-Tapia does. Not implemented.

### D. No latency
Quotes update instantly and cancels never race a fill. In practice latency is the dominant
source of adverse selection for a passive maker: you are picked off precisely because your
cancel did not arrive in time. Modelling it would need an event queue with delays, which is
a substantially different engine.

### E. The reported spread is stale during a fill gap
When a quote is filled, `bid_quotes[i]` reports `nan` until the next requote. The
`mean_spread_captured` metric is therefore an average over the steps where a quote existed,
not over all steps. It is consistent across policies so the comparison is fair, but it is
not quite "the average spread this strategy quoted".

### F. Sharpe is per session, and sessions are arbitrary
The Sharpe numbers here (8-11 for A-S) are per simulated session, and a "session" is
defined by `n_steps` and the arrival rate. The sensitivity sweep shows the A-S Sharpe
rising from 2.8 to 22.7 as the arrival rate goes from 20 to 600 (roughly as `A^0.6`). So the *level* is not comparable to a real desk's
Sharpe; only the *ratio between strategies* is meaningful. The README says so, but it is a
real limitation of any simulation study of this kind.

### G. The fill curve is not exponential, and the model assumes it is
Measured R-squared on the log-linear fit is ~0.85. The tail is fatter than exponential
because market-order sizes are Pareto. The A-S solution is derived under the exponential
assumption, so the "optimal" quotes are optimal for a market slightly different from the
one they are used in. Quantified and reported rather than hidden, but not corrected.

### H. `std(final q) = 3.0` against the paper's 2.0
Every other entry in the Table 1 reproduction matches within Monte Carlo error, including
the symmetric strategy's inventory spread (8.2 against 8.4) and both profit volatilities.
That the *symmetric* number matches while the inventory one does not argues the quoting
logic is right and something about the paper's inventory measurement differs, but I could
not identify what. It is reported rather than tuned away.

### I. No transaction costs in the headline results
`MarketConfig` carries `maker_fee` and `taker_fee` and the engine applies them, but the
headline comparison runs at zero. Real maker rebates would *help* the results (rebates are
negative fees) and taker fees do not apply to a pure maker, so this is conservative in the
direction that matters -- but it is untested at non-zero values beyond a unit test.

### J. Everything is one asset
No cross-asset hedging, no correlated inventory. A real desk manages a portfolio of
inventories and hedges the common factor, which changes the optimal skew substantially.

### K. Markout uses information a desk does not have
The adverse-selection split relies on knowing which takers were informed, which only a
simulation can know. A desk would have to classify flow statistically (by client, by
size, by subsequent markout), and misclassification would blur the split. Markouts are
also measured against the *efficient* price, which the simulated maker observes directly;
a real maker sees only the book mid, which is noisier and lags. Both make the measured
adverse selection cleaner than a desk's would be.

### L. Sensitivity is one parameter at a time
Each sweep moves one parameter with the others at their base values, so interactions (for
example, how the best `gamma` shifts with `sigma`) are not explored. The reference-engine
sweeps use 500 sessions per point and the order-book informed-flow sweep only 100 (for run
time), so the latter's per-session Sharpe standard errors are 0.2-0.9, and differences
smaller than about two of them should not be read as findings.
