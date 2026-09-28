# Market Making Simulator

**A limit order book from scratch, the Avellaneda-Stoikov optimal quoting model, and an honest measurement of what inventory control is actually worth.**

The published Table 1 of Avellaneda & Stoikov (2008) is reproduced to two decimal places. Then the same strategies are run in a full matching engine with queue priority and informed traders, to see how much of the theory survives.

---

## The 60-second version

| | Result |
|---|---|
| **Order book** | Price-time priority matching engine, O(1) cancels, self-trade prevention. 23 tests on the invariants that flatter a strategy when broken. |
| **Validation** | Avellaneda & Stoikov Table 1 reproduced with their own discretisation: profit **64.98** vs published 65.0, std **6.62** vs 6.6; symmetric **68.44** vs 68.4, std **13.70** vs 13.4. Analytic average spread **1.4908** vs 1.49. |
| **Idealised world** | A-S per-session Sharpe* **9.29** vs symmetric **4.64** — at an identical average spread. Inventory std 1.24 vs 8.07. |
| **Full order book** | A-S per-session Sharpe* **11.7** vs symmetric **2.8**, at statistically identical PnL (p = 0.53). Max position 10.7 vs 53.3. |
| **Fair benchmark** | Against a naive maker *with a hard position limit*, A-S still wins: per-session Sharpe* 11.7 vs 6.9 and **+1.33 PnL per run, t = 10.7**. |
| **Markout** | The A-S maker's fills against informed takers lose **−1.24 / −1.89 ticks** at +20 / +100 steps, against the model's own −1.28 / −1.99; uninformed fills lose ~0. Adverse selection takes 4-7% of the edge at 15% informed flow. |
| **Adverse selection** | From 0% to 30% informed flow the symmetric maker's per-session Sharpe* falls **8.8 → 1.5**; A-S's stays flat (10.5 → 12.0, ± 0.8-0.9). A-S's smaller total adverse-selection cost comes from *when it unwinds*, not from dodging informed flow. |
| **Sensitivity** | Five parameters, three policies, standard errors on everything. A-S's edge grows with $\kappa$, $\sigma$ and informed flow; at too-low $\gamma$ the naive position limit beats it. |
| **Honest finding** | The published finite-horizon model **throws away its inventory control at the bell**: inventory std grows 1.06 → 2.90 over the session. The time-homogeneous variant stays flat at ~1.2. |

\* **Every Sharpe ratio in this README is per simulated session**: the mean of session
PnL divided by its standard deviation across independent sessions, *not annualised* and
not comparable to a trading strategy's Sharpe. Its level is set by how much order flow the
simulation generates (section 7). Only the ratios *between* policies mean anything. This
is a simulation study, so there is no market data, drawdown history or buy-and-hold
benchmark. The benchmarks are the naive quoting policies below.

```bash
pip install -r requirements.txt
python scripts/run_analysis.py      # ~8 min; writes figures/ and results/
python -m pytest                    # 138 tests, 98% branch coverage
```

---

## 1. The problem

A market maker quotes a bid and an ask and earns the spread when both get hit. The difficulty is everything in between:

- Fills arrive **one side at a time**, so the maker accumulates inventory it did not want.
- That inventory is a directional position, and its risk grows with volatility and holding time.
- Some of the flow hitting you **knows something you do not**, so the mid moves against you right after you trade.

Avellaneda-Stoikov is the canonical answer to the first two. This repo implements it, validates it against the paper, then asks what happens once fills depend on queue position and a fraction of the flow is informed.

---

## 2. The maths

### Setup

The efficient mid is an arithmetic Brownian motion $dS_t = \sigma\,dW_t$. The maker quotes $\delta^b$ below and $\delta^a$ above. Fills arrive as Poisson processes whose intensity falls exponentially with distance:

$$\lambda(\delta) = A e^{-\kappa\delta}.$$

The maker holds inventory $q$ and cash $X$ and maximises exponential utility $\mathbb{E}\big[-e^{-\gamma(X_T + q_T S_T)}\big]$.

### Solution

Solving the HJB equation and expanding to first order in inventory gives the **reservation price**

$$r(s, q, t) = s - q\gamma\sigma^2(T-t)$$

and the **optimal total spread**

$$\delta^a + \delta^b = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right),$$

with the quotes placed symmetrically around $r$, not around $s$:

$$\delta^a = \tfrac12(\delta^a+\delta^b) - q\gamma\sigma^2(T-t), \qquad \delta^b = \tfrac12(\delta^a+\delta^b) + q\gamma\sigma^2(T-t).$$

### The one idea

Everything is in the skew term $q\gamma\sigma^2(T-t)$. A maker who is long shifts **both** quotes down — keener to sell, more reluctant to buy — so inventory mean-reverts without the maker ever crossing the spread. The width never changes; only the centre moves.

![The model in one picture](figures/reservation_price.png)

A quirk worth knowing: the markup term $\frac{2}{\gamma}\ln(1+\gamma/\kappa)$ is *decreasing* in $\gamma$, approaching $2/\kappa$ from below. A more risk-averse maker charges a smaller monopolistic markup — it prefers a likelier small gain — but the linear inventory term dominates, so the total spread still widens.

---

## 3. Validating against the paper

At the paper's parameters ($\gamma=0.1$, $\kappa=1.5$, $A=140$, $\sigma=2$, $T=1$), the analytic average spread is

$$\tfrac12\gamma\sigma^2 T + \tfrac{2}{\gamma}\ln(1+\gamma/\kappa) = 0.2 + 1.2908 = \mathbf{1.4908},$$

against the paper's reported 1.49. That single number pins both terms of the formula and their relative weight against an independent source.

2,000 simulated sessions, using the paper's own first-order fill discretisation:

| Strategy | Spread | Profit | (paper) | Std(profit) | (paper) | Std(final $q$) | (paper) |
|---|---|---|---|---|---|---|---|
| Inventory (A-S) | 1.49 | **64.98** | 65.0 | **6.62** | 6.6 | 3.01 | *2.0* |
| Symmetric | 1.49 | **68.44** | 68.4 | **13.70** | 13.4 | **8.34** | 8.4 |

Six of the seven entries land within Monte Carlo error. The exception is `Std(final q)` for the inventory strategy: 3.01 against 2.0. That the *symmetric* strategy's inventory spread matches (8.34 vs 8.4) argues the quoting logic is right and something about the paper's measurement of that one number differs; I could not identify what, and it is reported rather than tuned away.

**The paper's discretisation overstates fills by ~13%.** It uses $\lambda\Delta t$ where the true probability of at least one arrival is $1-e^{-\lambda\Delta t}$. At $\lambda\approx45$, $\Delta t=0.005$ the gap is 10%, and it accounts for the entire difference between reproducing the paper (65.0) and the exact simulation (57.4). The exact form is the default here; `fill_model="linear"` exists to reproduce the published numbers.

---

## 4. What the model actually buys you

Every strategy quotes the **same average spread**, so the comparison measures inventory management and not how wide someone chose to quote. The benchmark set is:

1. **Avellaneda-Stoikov** — skews around the reservation price.
2. **Symmetric** — fixed half-spread around the mid, no inventory control at all.
3. **Symmetric + position limit** — the realistic naive strategy: stop quoting a side at a hard limit.

The third exists because beating a maker with *no* inventory control is too easy a win.

![Inventory paths](figures/inventory_paths.png)

2,000 sessions in the idealised model, with common random numbers so every policy faces the same price path:

| Policy | Mean PnL | Std PnL | **Sharpe*** | Mean \|q\| | Std final q | Max \|q\| | Trades | Max DD |
|---|---|---|---|---|---|---|---|---|
| Avellaneda-Stoikov | 57.18 | 6.16 | **9.29** | 0.92 | 1.24 | 6.0 | 85.9 | 0.99 |
| Symmetric | 61.17 | 13.19 | 4.64 | 4.29 | 8.07 | 29.0 | 82.0 | 5.72 |
| Symmetric + position limit | 54.22 | 7.48 | 7.25 | 1.57 | 1.94 | 3.0 | 72.7 | 1.84 |

![Risk and return](figures/risk_return.png)

**A-S makes less money than naive symmetric quoting**, by 3.99 per session (paired $t = -15.6$). That is not a defect, it is the trade: it gives up the option value of running a big position in exchange for halving PnL volatility, and doubles the per-session Sharpe doing so. Being explicit about this is more useful than a chart showing only the ratio.

Against the *fair* benchmark it wins on both axes: **+2.96 PnL per session** ($t = 25.9$) **and** a higher per-session Sharpe. The hard limit captures roughly three quarters of the risk benefit; the continuous skew supplies the rest while trading 18% more.

![PnL distributions](figures/pnl_distribution.png)

---

## 5. The same strategies in a real order book

The idealised engine fills the maker whenever a fill "arrives", regardless of who else is quoting. That assumption flatters every market-making backtest ever written, so the second engine removes it: quotes join a real FIFO queue, noise traders add and cancel liquidity, market orders walk the book, and 15% of them are informed.

### Parameters are estimated, not chosen

$\kappa$ is measured in **inverse price units**. Copying the paper's $\kappa=1.5$ onto a book with a 2-tick spread produces quotes 115 ticks wide and a maker that never trades — which is exactly what the first version did. So $\sigma$, $A$ and $\kappa$ are estimated from the simulated market by probing it, the way a desk estimates them from its own fill logs.

![Fill intensity](figures/fill_intensity.png)

The fit gives $A = 458$, $\kappa = 21.96$ per price unit ($1/\kappa = 4.55$ ticks), with $R^2 = 0.865$ on the log-linear regression. **Not 1.0** — the tail is fatter than exponential, because market-order sizes are power-law and the occasional large sweep reaches deep levels far more often than $e^{-\kappa\delta}$ predicts. The model's core assumption holds approximately, and the number says how approximately.

### Volatility must be measured at the right horizon

![Volatility signature](figures/volatility_signature.png)

This chart caught a real bug. Informed impact is released gradually, so the efficient price is **positively autocorrelated** at short horizons and a one-step volatility estimate understates what a maker faces by about 4x ($\sigma = 0.094$ at one step against $0.33$ at the horizon the maker actually holds inventory). With no informed flow the signature is flat — a pure martingale. With instantaneous impact it is flat again at the higher level. Only *slow price discovery* tilts it.

### Results

200 sessions of 3,000 steps each:

| Policy | Mean PnL | Std PnL | **Sharpe*** | Mean \|q\| | Max \|q\| | Trades | Max DD |
|---|---|---|---|---|---|---|---|
| Avellaneda-Stoikov | 12.66 | 1.08 | **11.7** | 1.49 | 10.7 | 262 | 0.18 |
| Symmetric | 12.47 | 4.44 | 2.8 | 8.69 | 53.3 | 241 | 2.26 |
| Symmetric + position limit | 11.34 | 1.65 | 6.9 | 3.40 | 8.6 | 220 | 0.59 |

In the book world A-S's PnL is **statistically indistinguishable** from symmetric quoting ($p = 0.53$) while its per-session Sharpe is 4.2x higher and its worst drawdown 12x smaller. It still beats the position-limit benchmark on PnL by 1.33 per session ($t = 10.7$).

PnL decomposes exactly into spread capture and inventory mark-to-market (reconciliation error 0.00000 in every run):

| Policy | Spread capture | Inventory PnL | Total |
|---|---|---|---|
| Avellaneda-Stoikov | 13.25 | −0.59 | 12.66 |
| Symmetric | 12.89 | −0.42 | 12.47 |
| Symmetric + position limit | 11.78 | −0.44 | 11.34 |

---

## 6. Adverse selection

Informed traders permanently move the efficient price in the direction they trade. With informed fraction $p$ and impact $J$, the expected cost per fill is $pJ$ — a maker quoting inside that loses money no matter how well it manages inventory.

### Markout, split by who you traded with

Every maker fill in the 200-session order-book comparison of section 5 (15% informed, $J = 2$ ticks released at $v = 5\%$ per step) is marked against the efficient mid $h$ steps later. Per fill, **edge** is what the maker earned against the mid it quoted against, **markout** is the signed mid move afterwards, and **realised spread = edge + markout**. Everything is size-weighted per unit filled, and standard errors are clustered by session, since fills in one session are not independent.

A simulation knows which takers were informed, so the informed markout has a number to hit: the impact still to be released after $h$ steps is $J(1-v)^h$, so a fill against an informed taker should mark out at $-J\,(1-(1-v)^h)$. It does:

| Steps after fill | Theory, informed | A-S, informed | Symmetric, informed | Limit, informed | A-S, uninformed | Symmetric, uninformed |
|---|---|---|---|---|---|---|
| +1 | −0.10 | −0.10 | −0.10 | −0.10 | +0.00 | −0.00 |
| +5 | −0.45 | −0.44 | −0.46 | −0.46 | +0.02 | −0.01 |
| +20 | −1.28 | −1.24 ± 0.02 | −1.31 ± 0.03 | −1.29 ± 0.02 | +0.04 ± 0.01 | −0.04 ± 0.01 |
| +100 | −1.99 | −1.89 ± 0.07 | −1.98 ± 0.07 | −1.91 ± 0.07 | +0.08 ± 0.02 | −0.07 ± 0.03 |

(Ticks per unit filled; ± one session-clustered standard error.) Uninformed fills carry essentially no adverse selection; informed fills lose exactly what the model says they should. That agreement is what makes the rest of the table believable.

![Markout](figures/markout.png)

### Where the spread goes

The same fills as PnL per session, marked at +100 steps (by which point 99% of informed impact has been released):

| Policy | Edge earned | Adverse selection: informed | Adverse selection: uninformed | Realised spread |
|---|---|---|---|---|
| Avellaneda-Stoikov | 12.82 | 0.72 | **−0.18** | **12.27** |
| Symmetric | 12.47 | 0.71 | 0.13 | 11.63 |
| Symmetric + position limit | 11.39 | 0.62 | −0.00 | 10.77 |

Edge here is slightly below the spread-capture column of section 5 (12.82 vs 13.25 for A-S) because fills in the last 100 steps have no +100 markout and are left out. Per unit, adverse selection takes 0.22 ticks of A-S's 5.06-tick edge and 0.36 of the symmetric maker's 5.33 — 4% and 7%, around the $pJ = 0.30$ ticks the flow parameters imply. **At 15% informed flow, adverse selection is a small tax on the spread; the large difference between the policies' per-session Sharpe ratios (section 5) comes from inventory risk, not from being picked off.**

### Why A-S is less adversely selected

The A-S maker loses less to adverse selection in total (0.55 vs 0.84 per session). An earlier version of this README said this was because "by skewing away from inventory it avoids being repeatedly filled on the same side by informed flow". The split says otherwise: **its informed losses are the same as the symmetric maker's** (0.72 vs 0.71 per session; −1.89 vs −1.98 ticks per unit, within 1 standard error). The whole difference is in its *uninformed* fills, and splitting those by whether they moved the position towards zero shows where:

| Policy | Unwinding fills: volume / session | markout (+100) | Position-adding fills: volume / session | markout (+100) |
|---|---|---|---|---|
| Avellaneda-Stoikov | 127.6 | **+0.24 ± 0.05** | 87.5 | −0.15 ± 0.08 |
| Symmetric | 97.8 | −0.21 ± 0.08 | 100.3 | +0.07 ± 0.09 |
| Symmetric + position limit | 96.6 | −0.18 ± 0.08 | 84.6 | +0.21 ± 0.09 |

A-S's unwinding fills make money after the fill. The mechanism consistent with this: when an informed buyer lifts A-S's offer, A-S is now short, so it raises both quotes; its bid becomes the aggressive one and buys back from uninformed sellers *while the informed impact is still being released* and the price is still rising. Those buy-backs recover 0.31 per session — about 43% of what the informed flow took. The symmetric maker does not move its quotes, so its unwinding is not timed this way. (Why its own unwinding fills mark out *negatively*, at 2.5 standard errors, I have not pinned down.)

### How much informed flow before it matters

Informed fraction swept in the order book, all three policies, 100 sessions each, with $\sigma$, $A$ and $\kappa$ re-estimated from the book at each fraction:

| Informed fraction | Expected cost (ticks) | A-S Sharpe* | Symmetric Sharpe* | Limit Sharpe* | A-S mean PnL | Symmetric mean PnL |
|---|---|---|---|---|---|---|
| 0% | 0.00 | 10.5 ± 0.8 | 8.8 ± 0.6 | 9.9 ± 0.7 | 12.85 ± 0.12 | 12.81 ± 0.15 |
| 10% | 0.20 | 11.2 ± 0.8 | 4.4 ± 0.4 | 7.6 ± 0.6 | 12.60 ± 0.11 | 12.51 ± 0.29 |
| 20% | 0.40 | 10.8 ± 0.9 | 2.8 ± 0.3 | 6.5 ± 0.5 | 12.68 ± 0.12 | 12.54 ± 0.44 |
| 30% | 0.60 | 12.0 ± 0.9 | 1.5 ± 0.3 | 5.6 ± 0.4 | 12.68 ± 0.11 | 9.85 ± 0.66 |
| 40% | 0.80 | 10.2 ± 0.7 | 1.5 ± 0.2 | 4.5 ± 0.3 | 12.76 ± 0.13 | 11.85 ± 0.79 |

\* per session, not annualised; ± one bootstrap standard error across sessions.

**With no informed flow the policies are close** (A-S 10.5 ± 0.8 vs symmetric 8.8 ± 0.6). Inventory control only pays once inventory is genuinely risky, and informed flow is what makes it so: it drives the measured $\sigma$ up fourfold (section 5), which drives the skew. As the informed fraction rises the symmetric maker's per-session Sharpe collapses, the position-limited maker degrades steadily, and A-S's stays flat within its standard errors. The previous version of this table (66 sessions, a different seed block, no standard errors) showed 12.9 vs 12.2 at 0%; that row does not reproduce on the new seeds, and the gap is about two standard errors — which is why the table now carries them.

---

## 7. Parameter sensitivity

All three policies at every point, with standard errors. Risk aversion $\gamma$, fill decay $\kappa$, volatility $\sigma$ and arrival intensity $A$ are swept in the idealised engine (500 sessions per policy per point, the same seeds at every point); the informed fraction exists only in the order book and is swept there (section 6). At each point the two benchmarks are re-matched to the A-S maker's average spread, so the comparison stays about inventory management. Full table: [`results/sensitivity.csv`](results/sensitivity.csv).

![Sensitivity](figures/sensitivity.png)

Per-session Sharpe*, ± one bootstrap standard error:

| Sweep | A-S | Symmetric | Symmetric + limit |
|---|---|---|---|
| $\gamma = 0.01$ | 7.37 ± 0.27 | 4.82 ± 0.23 | **8.04 ± 0.26** |
| $\gamma = 0.17$ (A-S peak) | **9.91 ± 0.31** | 4.88 ± 0.24 | 7.20 ± 0.24 |
| $\gamma = 3.0$ | 1.32 ± 0.06 | 1.31 ± 0.06 | 1.31 ± 0.06 |
| $\kappa = 0.5$ | **11.82 ± 0.37** | 9.66 ± 0.37 | 9.68 ± 0.30 |
| $\kappa = 4.5$ | **7.41 ± 0.25** | 2.04 ± 0.12 | 3.75 ± 0.13 |
| $\sigma = 0.5$ | **11.43 ± 0.37** | 10.47 ± 0.38 | 9.63 ± 0.30 |
| $\sigma = 5.0$ | **6.37 ± 0.20** | 2.34 ± 0.13 | 3.66 ± 0.13 |
| $A = 20$ | **2.82 ± 0.09** | 1.85 ± 0.10 | 2.18 ± 0.10 |
| $A = 600$ | **22.68 ± 0.73** | 11.90 ± 0.56 | 18.35 ± 0.65 |

\* per simulated session, not annualised, and not comparable to a trading strategy's Sharpe.

**Risk aversion $\gamma$ has an interior optimum — and below it, the naive limit wins.** The A-S per-session Sharpe peaks near $\gamma \approx 0.17$ and falls off on both sides: too much and the quotes are so wide the maker stops trading (at $\gamma = 3$ it makes 3.5 trades a session and all three policies are tied at 1.3); too little and the skew is too weak to control inventory. At $\gamma \le 0.04$ the hard position limit beats A-S (8.04 vs 7.37 at $\gamma = 0.01$; final-inventory std 1.90 vs 3.69). The model is only better than the naive rule when its one free preference parameter is set sensibly.

**The advantage grows with $\kappa$ and $\sigma$** — both make inventory more dangerous relative to what the spread earns. Higher $\kappa$ forces tighter quotes (matched spread 3.85 → 0.64 as $\kappa$ goes 0.5 → 4.5) against unchanged price risk, and the A-S lead over symmetric quoting widens from 1.2x to 3.6x. At low volatility the three policies are close ($\sigma = 0.5$: 11.4 vs 10.5 vs 9.6); at $\sigma = 5$ A-S is 2.7x the symmetric maker. A-S widens as $\sigma$ rises (spread 1.30 → 2.54) and trades 39% less.

**Arrival intensity $A$ raises every policy's per-session Sharpe** (A-S 2.8 → 22.7 as $A$ goes 20 → 600, roughly as $A^{0.6}$) without changing the ranking. More flow means more independent spread captures per session. This is also why the *level* of these ratios is not comparable to a real desk's — it is set by how much flow the simulation happens to generate.

**The same pattern everywhere:** A-S buys a much lower PnL variance at the cost of some mean PnL (in the idealised engine it earns less than symmetric quoting at every swept point, e.g. 57.2 vs 61.0 at the base $\kappa$), and that trade is worth most exactly when inventory is riskiest.

![A session](figures/session.png)

---

## 8. Limitations

- **One market maker, no competition.** Every other order is noise. The monopolistic markup term is doing a lot of work; it would shrink sharply against a second strategic maker. This is the most important missing piece.
- **The markout split uses information a desk does not have.** The simulation knows which takers were informed and marks against the efficient price; a desk has to classify flow statistically and mark against a noisier book mid.
- **Informed traders are exogenous.** They do not condition on the book, so the maker cannot reduce its adverse selection by quoting wider — one of the main levers a real desk has.
- **No latency.** Cancels never race fills. In practice that race *is* adverse selection for a passive maker.
- **Fixed size.** How much to quote is as important as where, and A-S says nothing about it.
- **Sharpe is per simulated session.** Only the ratio between strategies is meaningful; the level is an artefact of the arrival rate.
- **The fill curve is not exponential** ($R^2 = 0.865$), so the "optimal" quotes are optimal for a market slightly different from the one they run in.
- **One asset.** No cross-asset hedging or correlated inventory.

Full list, including six defects found and fixed during development, in [REVIEW.md](REVIEW.md).

---

## 9. Repository layout

```
src/mmsim/
  types.py               Side/Order/Trade, integer tick prices
  book.py                Price-time priority matching engine
  flow.py                Poisson flow, Pareto sizes, informed traders with gradual impact
  avellaneda_stoikov.py  Reservation price, optimal spread, finite and stationary horizon
  strategies.py          The three quoting policies, matched on average spread
  engine.py              The idealised A-S world and the full order-book world
  calibration.py         Estimate sigma, A, kappa from the book; volatility signature
  metrics.py             PnL decomposition, markout by counterparty, bootstrap SEs
  experiments.py         Common random numbers, paired tests, multi-policy sweeps
  style.py / plotting.py One validated palette; every figure in this README
scripts/run_analysis.py  The full study
tests/                   138 tests, 98% statement + branch coverage
```

## 10. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_analysis.py                       # full study, ~8 minutes
python scripts/run_analysis.py --runs-book 40        # a quick version

pytest -q
pytest -q --cov=src/mmsim --cov-report=term-missing
ruff check src tests scripts && mypy src scripts
```

Everything is seeded, and the defaults (2,000 reference sessions, 200 order-book sessions,
500 per reference sweep point, 100 per informed-fraction point) are the settings behind
every number above. Tables are also written as CSV: `results/markout_decomposition.csv`,
`results/markout_uninformed_by_inventory_effect.csv` and `results/sensitivity.csv`.

**Reproducibility check (2026-09-28).** The study was re-run with numpy 2.5 and pandas 3.0.
Every table reproduced exactly except the position-limited policy in the order-book world,
whose mean PnL moved from 11.31 to 11.34 (paired t against A-S from 11.0 to 10.7), and a few
average trade counts, which moved in the fourth significant figure. The tables above show
the new run. The runs are deterministic within one environment, and the cause across
environments was not isolated.
