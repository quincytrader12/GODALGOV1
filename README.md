# GODALGOV1

Algorithmic crypto trading bot. **Work in progress — architecture is still being decided.**

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
