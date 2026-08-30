# Theory and research grounding

Why this system is built the way it is, and which published results each design
decision rests on. Every claim here maps to code; file references are given.

---

## 1. The two strategies are opposite bets

Momentum and mean reversion are not two independent alphas to be averaged. They
are opposite bets on the sign of serial correlation in returns:

| | Bets that | Profits when | Fails when |
|---|---|---|---|
| **Momentum** | autocorrelation > 0 | trends persist | the trend snaps back |
| **Mean reversion** | autocorrelation < 0 | deviations close | the mean moves |

Run naively side by side, they trade against each other. One book buys what the
other sells, both pay the spread, and what survives is mostly fees. This is the
single most important fact in the design, and it is why allocation is
**conditional on an estimated regime** rather than fixed.

→ `src/godalgo/portfolio/allocator.py`

---

## 2. Momentum

**Jegadeesh & Titman (1993)** established *cross-sectional* momentum: buying past
winners and selling past losers earns abnormal returns over 3-12 month horizons.

**Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum"** established the
*time-series* form: an asset's own past 12-month excess return predicts its own
next-month return, across 58 instruments and 25 years. This is the version
implemented here, because it works on a single instrument and needs no broad
cross-section — which matters when trading a handful of pairs.

Two departures from the equity-literature defaults, both forced by the asset
class:

**Volatility scaling** — *Baltas & Kosowski (2013)*; *Moreira & Muir (2017),
"Volatility-Managed Portfolios"*. Raw momentum takes its largest positions
exactly when volatility is highest, which is when drawdowns are worst. Scaling
signal by inverse trailing volatility materially improves trend-following
Sharpe. Crypto volatility ranges over an order of magnitude, so this is not
optional.

**Multi-horizon blending** — single-lookback trend systems are notoriously
sensitive to the exact window; a 40-day and a 60-day breakout can differ wildly
over a given sample for no economic reason. Averaging geometrically spaced
horizons trades a little peak backtest performance for much less parameter
fragility. That is the right side of the trade when a search loop is scanning
the parameter space and will happily latch onto a lucky window.

→ `src/godalgo/strategies/momentum.py`

---

## 3. Mean reversion

**Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equities Market"**
model a residual spread as an Ornstein-Uhlenbeck process and trade its
normalised deviation (the "s-score"). Discretised OU is:

```
dy_t = a + b * y_{t-1} + e_t          b < 0 implies reversion
half-life = -ln(2) / b
```

Three things separate this from a naive Bollinger-band system, and they account
for most of the difference between a reversion strategy that survives and one
that does not:

**Half-life gating.** A spread with a 300-bar half-life is statistically
mean-reverting and economically untradeable — fees and funding consume the edge
before convergence. The lower bound matters too: a 1-bar half-life is usually
microstructure noise, not a capturable edge at our latency.

**Asymmetric entry/exit.** Entering at |z| = 2 and exiting at |z| = 0.5 creates
hysteresis. Symmetric thresholds produce a stream of round trips as z jitters
across one boundary, and in fee terms that is how a reversion book bleeds out.

**A stop on the z-score.** Reversion's failure mode is a regime break — the mean
moves, and the position scales into a loss *because the further it goes the more
attractive the signal looks*. The stop is the admission that the model is wrong
rather than early.

→ `src/godalgo/strategies/mean_reversion.py`

---

## 4. Regime detection decides who gets capital

Three independent tests, deliberately not variations on one idea.

**Hurst exponent** (*Hurst 1951*; *Mandelbrot & Van Ness 1968*). For a
self-similar process, `Std[x(t+τ) - x(t)] ~ τ^H`. H > 0.5 diffuses faster than a
random walk (trending); H < 0.5 slower (reverting).

Estimated by generalised variance scaling rather than classical rescaled-range
(R/S) analysis: R/S carries a well-documented small-sample bias that pushes H
above 0.5 even on genuine random walks, which would make everything look like a
momentum opportunity.

**Variance ratio test** (*Lo & MacKinlay 1988*). Under a random walk,
`VR(q) = Var(q-period returns) / (q · Var(1-period returns)) = 1`. Same intuition
as Hurst, but with an actual sampling distribution — so it yields a significance
level, not just a number. The **heteroskedasticity-robust** statistic is used
because the homoskedastic version over-rejects badly when volatility clusters,
which in crypto it always does. Using the naive version would manufacture
"regimes" that are really just volatility regimes.

**ADF + OU half-life.** ADF asks whether a unit root can be rejected; half-life
says how fast reversion happens, which is what makes a trade viable.

### The tests are deliberately not weighted equally

The variance ratio is the **primary discriminator** — it is the only one of the
three with a sampling distribution. Hurst is a point estimate with known
small-sample bias and no error bars, so it acts as a **veto**: it can block a
regime call by pointing the other way, but it is not required to clear an
arbitrary band before a properly significant test result counts.

This ordering was corrected during development. Gating on the Hurst band first
made the classifier refuse regimes it had a variance-ratio z-statistic of 10 for,
and left it holding minimal risk 84% of the time.

→ `src/godalgo/features/regime.py`

---

## 5. Self-improvement is where trading systems die

This is the part of the design that needs the most defending, because the
obvious implementation is actively harmful.

**The problem.** A system that re-tunes itself on its own backtests is,
mathematically, an overfitting machine. If you evaluate N configurations and
keep the best, the maximum Sharpe you observe is an *order statistic*, and its
expectation is well above zero even when every configuration is worthless.
Selecting on raw Sharpe therefore promotes noise — reliably, and more
confidently the harder the system searches.

This codebase demonstrates the effect on its own synthetic data. Widening the
search from 8 trials to 80:

| | 8 trials | 80 trials |
|---|---|---|
| OOS Sharpe | 1.88 | **2.24** |
| Deflated Sharpe | 0.887 | **0.534** |

The raw number improved. The evidence got *worse*. Naive selection would have
called that an upgrade.

### The three defences

**Purged walk-forward** (*López de Prado, "Advances in Financial Machine
Learning", 2018*). Parameters are fitted on a training window and judged only on
the window that follows. Two refinements matter because our features use
trailing windows:

- *Purging* — a signal at the first test bar is computed from a window reaching
  back into training. Train and test share information even though their index
  ranges do not overlap. Purging drops the training tail the test set reaches
  into.
- *Embargo* — serial correlation means bars just after the test window still
  carry information about it, so the next fold's training set must not start
  inside that shadow.

Skip either and out-of-sample scores drift up toward in-sample ones, defeating
the entire purpose of the split.

**Deflated Sharpe Ratio** (*Bailey & López de Prado 2014*). Asks whether an
observed Sharpe exceeds what the best of N trials would produce by luck alone,
while also correcting for skew and fat tails — both of which inflate apparent
significance and both endemic to crypto.

```
E[max SR_N] ≈ σ_SR · [(1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))]      γ = Euler-Mascheroni

DSR = Φ( (SR - SR*)·√(T-1) / √(1 - skew·SR + ((kurt-1)/4)·SR²) )
```

Counting N honestly is essential. Undercounting trials is the easiest way to
make a worthless strategy pass.

**Probability of Backtest Overfitting** (*Bailey, Borwein, López de Prado & Zhu
2017*), via combinatorially symmetric cross-validation. A different and
complementary question: across many IS/OOS splits, how often does the
configuration that looked best in-sample land *below median* out-of-sample?

PBO ≈ 0.5 means selection is no better than random — the search is fitting
noise. A candidate can post a fine OOS Sharpe while sitting inside a search whose
winners are systematically noise; PBO catches that and DSR does not.

→ `src/godalgo/backtest/metrics.py`, `src/godalgo/evolve/`

---

## 6. What the loop may and may not do

| Permitted | Forbidden |
|---|---|
| Propose parameters inside declared `ParamSpec` bounds | Generate or modify code |
| Search, and be penalised for searching | Adjust its own acceptance thresholds |
| Promote on out-of-sample evidence | Touch `RiskLimits` |
| Record every decision to an append-only ledger | Erase or rewrite past decisions |

The search space is a closed, validated box. `RiskLimits` appears in no
strategy's `SPACE`, so the optimiser cannot widen its own leash — **a system that
can relax its own stop-loss does not have a stop-loss.**

Random search rather than grid search follows *Bergstra & Bengio (2012)*: at a
fixed budget, random sampling beats a grid because most parameters barely matter
and a grid spends most of its budget resolving the ones that do not.

→ `src/godalgo/evolve/search.py`, `src/godalgo/evolve/promotion.py`,
`src/godalgo/risk/limits.py`

---

## 7. Sizing and risk

**Volatility targeting.** Volatility is far more forecastable than return — it is
strongly autocorrelated, while returns are near-unforecastable at these
horizons. Targeting it turns an unstable risk profile into a roughly stationary
one, which is what makes performance statistics mean the same thing in every
period the promotion gate measures them over.

**Fractional Kelly.** Full Kelly (`edge / variance`) maximises long-run log
wealth *given known parameters*. We do not know the parameters — we estimate the
edge from a finite backtest, and Kelly is brutally sensitive to that estimate.
Overestimating the edge by 2× under full Kelly produces **negative** expected log
growth. Quarter Kelly gives roughly 44% of the growth rate at roughly a quarter
of the variance drag, and stays positive-growth even when the edge estimate is
off by a factor of two.

**No-trade band.** Every rebalance pays the spread, so a continuously varying
target leaks money through changes too small to matter.

→ `src/godalgo/portfolio/sizing.py`, `src/godalgo/risk/limits.py`

---

## 8. Backtest conventions

Three conventions keep results honest. Each exists because its absence is a
standard way to produce an untradeable backtest.

1. **Signals lag one bar.** A signal from bar `t`'s close is traded at `t+1`.
   This is the most common lookahead bug there is, and it is usually worth an
   enormous, entirely fictional Sharpe. Enforced and tested: truncating the data
   must not change past signals (`tests/test_causality.py`).
2. **Costs charged on turnover**, not per trade — fees plus modelled slippage
   scale with how much the position actually moved. Defaults are deliberately
   pessimistic: a backtest that only survives at optimistic costs is telling you
   it does not survive.
3. **Warm-up discarded**, so the first reported bar is one the system could
   genuinely have traded.

→ `src/godalgo/backtest/engine.py`

---

## Selected references

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance.
- Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012). *Time Series Momentum*. Journal of Financial Economics.
- Moreira, A. & Muir, T. (2017). *Volatility-Managed Portfolios*. Journal of Finance.
- Baltas, N. & Kosowski, R. (2013). *Momentum Strategies in Futures Markets and Trend-Following Funds*.
- Avellaneda, M. & Lee, J.H. (2010). *Statistical Arbitrage in the US Equities Market*. Quantitative Finance.
- Lo, A. & MacKinlay, A.C. (1988). *Stock Market Prices Do Not Follow Random Walks*. Review of Financial Studies.
- Mandelbrot, B. & Van Ness, J. (1968). *Fractional Brownian Motions, Fractional Noises and Applications*. SIAM Review.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. Journal of Portfolio Management.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). *The Probability of Backtest Overfitting*. Journal of Computational Finance.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR.
- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*.
