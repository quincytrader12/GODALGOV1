# GODALGOV1

Regime-aware algorithmic crypto trading bot running two quant strategies —
**time-series momentum** and **Ornstein-Uhlenbeck mean reversion** — with a
self-improvement loop gated on out-of-sample statistical evidence.

The design rests on one claim: momentum and mean reversion are *opposite bets on
the sign of serial correlation*, so running them side by side at fixed weights is
close to self-cancelling. Allocation is therefore conditional — a regime
classifier estimates which autocorrelation sign currently holds, and capital
follows that estimate.

See **[docs/THEORY.md](docs/THEORY.md)** for the research grounding behind every
design decision.

## How it fits together

```
ccxt OHLCV / ticks ──> indicators ──> ┌─ momentum (TSMOM, vol-scaled, multi-horizon)
                                      └─ mean reversion (OU z-score, half-life gated)
                                              │
       regime classifier ────────────────────>│   Hurst · variance ratio · ADF
       (which bet is live?)                   │
                                              │
       session overlay ──────────────────────>│   shrunk hour-of-clock drift
       (overnight/clientele drift)            │
                                              v
                                 regime-weighted blend
                                              v
                            vol targeting + no-trade band
                                              v
                   EDGE GATE   expected edge >= 1.5x round-trip cost
                                              v
                   RISK LIMITS  deterministic, outside the loop
                                              v
                   ROUTER  post-only, throttled, one order per symbol
                                              v
                   BROKER   dry-run | paper | live (triple-armed)
```

The self-improvement loop wraps this: propose parameters → purged walk-forward →
promotion gate (deflated Sharpe + PBO) → append-only ledger.

## Usage

```bash
uv sync
python -m godalgo backtest --symbol BTC/USDT --timeframe 1h --limit 8000
python -m godalgo evolve   --symbol BTC/USDT --candidates 16
python -m godalgo live     --symbol BTC/USDT --bar-seconds 60   # dry run by default
python -m godalgo ledger
pytest                     # 112 tests
```

## Execution

Event-driven and autonomous. To be clear about scope: this is **not** HFT in the
co-located, sub-millisecond sense — that is not reachable from Python over ccxt.
Realistic latency is ~10–50ms for WebSocket data and ~50–200ms for order
placement.

At that frequency the binding constraint is not latency but **cost arithmetic**.
A round trip costs the spread plus two fees — at a 2bp spread and 6bp taker
fees, 14bp before slippage. A signal predicting a 5bp move backtests profitably
and loses money live. Speed does not fix that; it only loses faster.

So every order passes an economic gate: expected edge must exceed modelled
round-trip cost by 1.5×, or no order is sent. **The same gate runs in the
backtest**, so the two paths take the same trades — without that symmetry a
backtest predicts nothing about live behaviour.

| Round trip @ default fees | Cost |
|---|---|
| Maker (post-only) | 4bp |
| Taker (crossing) | 16bp |

That 4× gap is why post-only is the default.

### Safety

| Control | Behaviour |
|---|---|
| **Trading mode** | `DRY_RUN` by default — computes orders, sends nothing |
| **Live arming** | Needs `arm=True` **and** `GODALGO_ARM_LIVE=...` **and** credentials |
| **Staleness watchdog** | Flattens if market data stops (a silent feed is worse than a bad signal) |
| **Reconciliation** | Venue is the source of truth; drift beyond tolerance halts |
| **Ambiguous sends** | Reported `UNKNOWN`, never "rejected" — then halt and reconcile |
| **Kill switch** | Cancels resting orders *then* flattens, in that order |
| **In-flight dedupe** | One order per symbol; a signal cannot become a double position |

Credentials are read from the environment only, never accepted as arguments and
never written to disk.

## Session / overnight drift overlay

The equity overnight anomaly (Cooper/Cliff/Gulen 2008; Lou/Polk/Skouras 2019) —
essentially the whole equity risk premium accruing close-to-open — **does not
port directly to crypto**, which has no close and therefore no gap.

What ports is the *mechanism*: Lou/Polk/Skouras attribute the effect to
clientele, predictable variation in *who is trading* by clock. That holds in
crypto via regional sessions, the real CME futures gap (Fri 22:00 → Sun 23:00
UTC), 8-hour funding settlement, and US-session equity spillover.

So the overlay learns a conditional drift **by clock bucket**, with
**empirical-Bayes (James-Stein) shrinkage** — because estimating 24 hourly means
is 24 simultaneous tests, and some bucket always looks significant on noise.

Measured on synthetic data:

| Input | Best raw bucket | After shrinkage |
|---|---|---|
| Pure noise | 3.27bps | **0.000bps** (λ=0) |
| Implanted +6bps US-session drift | 7.38bps | **+2.79bps** (λ=0.54) |

It attaches as a convex blend scaled by the regime allocator's authorised risk,
so the calendar modulates the strategies rather than adding exposure on top —
and cannot restore risk an indeterminate regime withheld. Off by default.

## Self-improvement, and why it is gated

A system that re-tunes itself on its own backtests is mathematically an
overfitting machine: evaluate N configurations, keep the best, and the maximum
Sharpe you observe is an *order statistic* whose expectation is well above zero
even when every configuration is worthless.

Measured on this repo's own synthetic data — widening the search from 8 trials
to 80:

| | 8 trials | 80 trials |
|---|---|---|
| OOS Sharpe | 1.88 | **2.24** |
| Deflated Sharpe | 0.887 | **0.534** |

The raw number improved; the evidence got worse. Naive selection would call that
an upgrade. The gate rejects it.

Promotion requires **all** of: OOS Sharpe floor, deflated Sharpe ≥ 0.95
(corrects for N trials), PBO ≤ 0.35 (does the *selection procedure* generalise?),
a drawdown ceiling, parameter stability across folds, and a minimum improvement
over the incumbent. Every decision — pass or fail — is appended to a JSONL ledger.

### What the loop may and may not do

| Permitted | Forbidden |
|---|---|
| Propose parameters inside declared `ParamSpec` bounds | Generate or modify code |
| Search, and be penalised for searching | Adjust its own acceptance thresholds |
| Promote on out-of-sample evidence | Touch `RiskLimits` |

`RiskLimits` appears in no strategy's search space. **A system that can relax its
own stop-loss does not have a stop-loss.**

## Layout

| Path | Role |
|---|---|
| `core/types.py` | Signals, regimes, targets, fills |
| `data/feed.py` | ccxt OHLCV: pagination, caching, partial-bar handling |
| `features/` | Indicators; regime classification (Hurst, VR, ADF, half-life) |
| `strategies/` | Momentum and mean reversion, over a bounded parameter space |
| `portfolio/` | Regime allocator; vol targeting, fractional Kelly, no-trade band |
| `features/session.py` | Overnight/session drift, James-Stein shrunk |
| `risk/limits.py` | Deterministic caps and kill switch — **not tunable** |
| `execution/` | Brokers (dry-run/paper/live), router, reconciler, live engine |
| `data/stream.py` | Tick → bar aggregation on wall-clock boundaries |
| `backtest/` | Engine with costs; metrics incl. deflated Sharpe and PBO |
| `evolve/` | Purged walk-forward, parameter search, promotion gate + ledger |

## Status

112 tests. Backtest, evolution, and execution paths are implemented and tested,
including order construction, the economic gate, paper fill semantics, arming
refusal, and the session estimator's negative case.

**Not verified against a live venue.** This development sandbox blocks exchange
APIs at the proxy, so everything network-facing — the OHLCV feed's pagination
and retry, and the entire `LiveBroker` path — has been written and reviewed but
never exercised against a real exchange. Treat first live contact as untested
code.

**No market-data stream is wired up.** `LiveEngine` exposes `on_tick()` and
`on_book()` and is driven by whatever feeds them; a `ccxt.pro` WebSocket driver
is not yet written.

Before any real capital: run in `paper` against live data for long enough to
confirm fill rates resemble the paper model's, and verify the kill switch
actually flattens on a venue.

## Setup

Requires Python ≥3.11 and [uv](https://docs.astral.sh/uv/). Prime Agent additionally
requires Node ≥22.8 — see `vendor/prime-agent/` for its build instructions.

```bash
git clone --recurse-submodules https://github.com/quincytrader12/GODALGOV1
cd GODALGOV1
uv sync
uv run python -c "import ccxt; print(len(ccxt.exchanges), 'exchanges')"
```

## Dependencies

### `vendor/prime-agent` (git submodule)

[PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) — MIT
licensed. Pinned to `a903d4b` (tag `beta`, v0.8.1), tracking `main`.

A self-improving RLM (Recursive Language Model) agent harness. It is the intended
**agent runtime** for this bot: its daemon-backed sessions, heartbeats, and schedules
drive the trading loop, and strategy logic is expected to live as Python skills the
harness invokes.

| Capability | Why it matters here |
| --- | --- |
| Persistent Python REPL as the primary tool | Strategy state survives across turns |
| Daemon-backed sessions | The bot keeps running when the terminal detaches |
| `/heartbeat`, `prime-agent schedule` | Re-enter a session on a market-clock cadence |
| Recursive subagents via `rlm(...)` | Parallel research / per-symbol analysis |
| Continual harness state | Durable memories, skills, and subagent specs |
| Bounded autonomous mode | Turn, token, and time budgets with quality gates |

Layout: TypeScript monorepo (`packages/agent`, `packages/ai`, `packages/coding-agent`,
`packages/tui`) plus a Python runtime in `prime-agent-runtime/`.

#### Working with the submodule

```bash
# Already cloned without submodules
git submodule update --init --recursive

# Pull a newer upstream commit (re-pins; commit the change)
git submodule update --remote vendor/prime-agent
```

### `ccxt` (pip dependency)

[ccxt/ccxt](https://github.com/ccxt/ccxt) — MIT licensed. Pinned to `4.5.76` in
`pyproject.toml`, locked in `uv.lock`.

Unified trading API across crypto exchanges — market data, order placement, balances,
and OHLCV history behind one interface, so strategy code isn't written against a single
venue's REST quirks.

Verified on this pin:

- **103** exchanges via `import ccxt`
- **`ccxt.async_support`** — asyncio client (`import ccxt.async_support as ccxt`)
- **`ccxt.pro`** — WebSocket streaming, **76** exchanges with live order book / trade feeds

Both `async_support` and `pro` ship in the base package as of v4; no extras needed.

Not vendored: the full source tree is ~480MB, mostly generated ports to JS/PHP/Go/C#/Java
that this project never uses.

## Security

> [!WARNING]
> Prime Agent executes model-generated Python and project commands with your user
> permissions. Its worker and kernel processes improve lifecycle isolation but are
> **not** a security sandbox.

This matters more than usual for a trading bot. An agent runtime holding exchange API
keys with withdrawal or order authority should be constrained by hard risk limits
enforced in deterministic code *outside* the agent's reasoning — position caps, daily
loss limits, and a kill switch that a bad generation cannot route around.

Exchange credentials belong in the environment, never in the repo. `.env` is gitignored.
