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
ccxt OHLCV ──> indicators ──> ┌─ momentum (TSMOM, vol-scaled, multi-horizon)
                              └─ mean reversion (OU z-score, half-life gated)
                                      │
              regime classifier ──────┤   Hurst · variance ratio · ADF + half-life
              (which bet is live?)    │
                                      v
                         regime-weighted blend
                                      v
                    vol targeting + no-trade band
                                      v
              RISK LIMITS  (deterministic, outside the loop)
                                      v
                                   orders
```

The self-improvement loop wraps this: propose parameters → purged walk-forward →
promotion gate (deflated Sharpe + PBO) → append-only ledger.

## Usage

```bash
uv sync
python -m godalgo backtest --symbol BTC/USDT --timeframe 1h --limit 8000
python -m godalgo evolve   --symbol BTC/USDT --candidates 16
python -m godalgo ledger
pytest                     # 69 tests
```

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
| `risk/limits.py` | Deterministic caps and kill switch — **not tunable** |
| `backtest/` | Engine with costs; metrics incl. deflated Sharpe and PBO |
| `evolve/` | Purged walk-forward, parameter search, promotion gate + ledger |

## Status

Backtest and evolution paths are implemented and tested (69 tests). **No live
order execution yet** — there is no exchange authentication, no order router, and
no position reconciliation. The data feed is read-only and accepts no API keys.

The feed's network path is unverified: this development sandbox blocks exchange
APIs, so pagination and retry logic have been reviewed but not exercised against
a live venue. Its frame-normalisation logic is covered by tests.

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
