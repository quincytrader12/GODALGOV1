# GODALGOV1

Algorithmic trading bot. **Work in progress — architecture is still being decided.**

## Dependencies

### `vendor/prime-agent` (git submodule)

[PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) — MIT licensed.

A self-improving RLM (Recursive Language Model) agent harness. It is the intended
**agent runtime** for this bot: its daemon-backed sessions, heartbeats, and schedules
drive the trading loop, and strategy logic is expected to live as Python skills the
harness invokes.

Relevant capabilities:

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

Pinned to `a903d4b` (tag `beta`, v0.8.1), tracking `main`.

#### Working with the submodule

```bash
# First clone
git clone --recurse-submodules https://github.com/quincytrader12/GODALGOV1

# Already cloned without submodules
git submodule update --init --recursive

# Pull a newer upstream commit (re-pins; commit the change)
git submodule update --remote vendor/prime-agent
```

> [!WARNING]
> Prime Agent executes model-generated Python and project commands with your user
> permissions. Its worker and kernel processes are **not** a security sandbox. This
> matters more than usual here: an agent runtime with order-placement authority
> should be scoped with hard, non-LLM risk limits enforced outside the agent.
