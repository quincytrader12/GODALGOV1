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
pytest                     # 298 tests
```

## What frequency can it trade?

There is an arithmetic answer, and it is worth knowing before running anything.

**Volatility scales as √time. Costs do not scale at all.** Expected move over a
holding period is `σ_annual · √(T_bar · hold / T_year)`, while a round trip costs
a fixed number of bps — the spread plus two fees — regardless of bar length.
Shorten the bar and edge shrinks as √T while cost stays put. Below some bar
length, no signal pays for itself.

Minimum viable bar, at 60% annual vol and 0.30 conviction:

| holding period | maker (4bp) | taker (16bp) |
|---|---|---|
| 3 bars | 117s | 31 min |
| 25 bars | **14s** | 3.7 min |
| 50 bars | 7s | 112s |

Only two things move that floor: **hold longer** (edge grows as √hold) or **stop
crossing the spread** (maker vs taker is ~4× in cost, ~16× in bar length).
Lowering the edge gate is not on the list — it does not create edge, it only
stops measuring it.

Run `python -m godalgo feasibility --symbol BTC/USDT --timeframe 1h` to get this
for your own config, computed from a real backtest rather than assumptions —
realised volatility, the conviction the strategies actually produce, and their
blended holding period.

### Why it wasn't trading

Earlier this refused every trade at default settings. Two causes, both fixed:

1. **The edge horizon was hardcoded to 3 bars.** Momentum with a 20–100 bar
   lookback holds ~28 bars; mean reversion holds ~its half-life, ~51. Expected
   move scales with √hold, so a 3-bar assumption understated momentum's edge by
   **3.03×** and refused trades that cleared their costs comfortably. Holding
   period is now derived from each strategy's own parameters.
2. **The no-trade band was absolute.** "2% of equity" means something completely
   different when vol targeting caps positions at 0.15 versus 1.0 — it blocked
   nearly everything on a volatile asset. The band is now a fraction of the
   intended position size, so it is scale-invariant.

At default settings the same run now sends 8 orders / 9 fills instead of zero,
while still refusing 17 of 25 decisions — selective, not permissive.

## Terminal UI

A local, single-page terminal — the bot's decision state rendered as a neural
cluster. Run it with:

```bash
python -m godalgo ui --demo      # fabricated positions, nothing traded
python run-terminal.py --demo    # double-clickable launcher
```

### Download a prebuilt executable

Builds for Windows, macOS and Linux are produced by CI on real runners for each
platform — PyInstaller does not cross-compile, so a Windows `.exe` cannot be
made on a Linux machine.

| Get it from | How |
|---|---|
| **Releases** | Push a tag: `git tag v0.1.0 && git push origin v0.1.0` |
| **Actions artifacts** | Run *build terminal* from the Actions tab, then download from the run |

The download is a **zip containing a folder**, not a bare executable. Unzip it
and run `godalgo-terminal.exe` from inside; the files beside it are the
libraries it loads at startup, so moving the exe out on its own breaks it.

```
godalgo-terminal.exe --demo    # fabricated data, nothing traded
godalgo-terminal.exe           # attached to a real session
```

Both binaries are **unsigned**, so expect a warning on first run: Windows
SmartScreen ("More info" → "Run anyway") and, on macOS, `xattr -d
com.apple.quarantine <file>` plus `chmod +x`. That is normal for an unsigned
PyInstaller build.

Every CI build runs the full test suite first and is then smoke-tested by
starting the binary and requesting the page, its JS, its CSS and the API — the
build succeeding proves nothing on its own, since the classic PyInstaller
failure is a binary that starts, serves the API, and 404s its own page.

### Building the executable yourself

```bash
pip install pyinstaller
python build-exe.py
```

Produces `dist/godalgo-terminal/` — a folder holding the executable and its
libraries, needing no Python on the target machine.

```bash
./dist/godalgo-terminal/godalgo-terminal --demo   # fabricated positions
./dist/godalgo-terminal/godalgo-terminal          # attached to a real session
```

`--onefile` collapses that folder into one file, at a real cost: the whole
~150 MB archive unpacks to a temp directory *on every launch* before any code
runs, which is 11.7s cold against 0.47s for the folder, and worse on Windows
where Defender scans each extracted file as it appears. The folder is the
default for that reason.

Two things break a naive PyInstaller build, both handled by the script:
**static files** (PyInstaller bundles imported modules, not data — without the
mapping the binary serves the API and 404s its own page, which reads as a server
fault) and **hidden imports** (uvicorn and ccxt resolve much of their machinery
by name at runtime, invisible to static analysis).

Verified: the built binary serves the page, JS, CSS and API, with the cluster
live.

### Running it 24/7 on Windows

Leaving the window open is enough — the bot trades for as long as the process
lives. To have Windows start it for you and bring it back if it dies:

```
godalgo-terminal.exe service install     # start it when I sign in
godalgo-terminal.exe service start       # ...and start it now
godalgo-terminal.exe service status
godalgo-terminal.exe service uninstall
```

This registers a scheduled task. It is not an NT service, and the difference
matters less than it sounds: a real service has to implement the Service
Control Manager protocol, and registering a plain executable with `sc create`
produces one Windows kills seconds later for failing to answer a start request.
The scheduled task gives the same practical result — starts on its own,
restarts on failure, survives closing the window — using only what Windows
ships with.

Three settings in the task definition are the point of writing it as XML rather
than `schtasks` flags, none of which are reachable from the command-line form:

| Setting | Why |
|---|---|
| `MultipleInstancesPolicy: IgnoreNew` | A restart must not start a **second bot on the same account**. Two instances would each size off the same buying power while blind to the other's position. |
| `ExecutionTimeLimit: PT0S` | The default is 72 hours, after which Windows terminates the process mid-position. |
| `RestartOnFailure: 1 min × 999` | Comes back from a crash without anyone present. |

It starts **at sign-in, not at boot**, and that is a constraint rather than a
preference: your exchange keys live in `%USERPROFILE%\.godalgo`, restricted by
ACL to your account. A task running at boot has no logged-on user, so it runs as
SYSTEM — which cannot read them, and would come up and trade nothing. Running as
*you* at boot means storing your Windows password in Task Scheduler; `--at-boot`
does that and says so before prompting.

`service stop` kills the process. It does **not** close open positions — an
orderly exit goes through the terminal, which flattens first.

### Switching between paper and live

The header carries a **DRY / PAPER / LIVE** switch.

| Switch | Requires |
|---|---|
| dry ↔ paper | nothing |
| → live | a key with **allow this key to place orders** ticked, **and** typing `GO LIVE` |
| live → paper | nothing — reducing risk is never harder than taking it |

**Every switch flattens first.** A paper position is not a real one: switching
to live while holding one leaves the engine convinced it owns something the
venue has never heard of, and its next decision is sized against that fiction.
Switching *out* of live while holding one abandons real exposure with nothing
managing its stop. A failed flatten refuses the switch outright.

**The interface can request live; it cannot authorise it.** The confirmation is
checked server-side, so a mis-click cannot start trading real money.

**The presence of an API key is not consent to use it.** Each route to a key
carries its own consent record, and there is no route without one:

* a key added in the terminal is **read-only** until you tick *allow this key
  to place orders* on that specific key;
* a key supplied through `GODALGO_API_KEY` / `GODALGO_API_SECRET` still needs
  `GODALGO_ARM_LIVE`, because there is no per-key tick to apply to it.

The environment variable used to gate both routes. It was dropped from the
terminal route only because the tick replaced it: the app ships as a
double-clicked executable, where "export a variable first" is not friction but
a wall, and friction the legitimate operator cannot clear is an outage rather
than a safety feature.

### Proving it works before funding anything

Binance has no paper account, so the honest question — *is this actually
talking to the exchange?* — has to be answerable without depositing. Three
checks answer it, in increasing order of what they prove, and **none of them
can place an order**:

| Check | Needs a key? | Needs funds? | Proves |
|---|---|---|---|
| Reachability | no | no | The venue is up, the region is not blocked, the network path works |
| Live price | no | no | Market data — the entire input to the strategy stack — is arriving |
| Balance read | yes | **no** | The key, secret, signature, clock and IP allow-list all work |

The last one passes on an **empty account**. A zero balance still proves the
credential, which is the whole point.

Press **Test venue connection** in the Connections panel, or just watch the
header: the market-data poller runs from startup in every mode, using the
public API, so live prices appear before anything is configured.

For the order path itself, Binance's **Spot Testnet** is real API, real order
lifecycle, fake money. Get keys from
[testnet.binance.vision](https://testnet.binance.vision), add them with
**testnet / sandbox keys** ticked, and the whole stack runs end to end without
a deposit. A testnet key is preferred over a real one when both are permitted
to trade — someone who set one up is mid-verification, and quietly picking
their real-money key instead is exactly the surprise this system exists to
avoid.

### The watchlist

The panel beside the cluster shows what the bot is currently looking at —
twelve liquid majors by default, live, from startup, in every mode. It costs
no key and no funds, so it is also the fastest answer to "is this thing
running".

| Column | |
|---|---|
| instrument | `POS` if a position is open in it, `ON` if it is the symbol being decided |
| last | Live price |
| 24h | The venue's own 24-hour change, not one derived from our history — which would be wrong for the first hour of a run |
| spread | Basis points of the mid; the first thing that makes a symbol untradeable |

Ordering is what the bot is **in**, then what it is **on**, then by turnover.
Sorting alphabetically would bury the row that matters under whatever starts
with an A.

Two things keep it cheap enough to run beside a 60fps canvas:

* **One request per poll, not one per symbol.** `fetch_tickers` covers the
  whole list in a single call. The per-symbol form is also sequential, so a
  dozen symbols meant a dozen round trips of latency stacked inside one tick —
  the usual reason a watchlist feels slow, and a good way to trip a rate limit.
* **Rows are mutated, never rebuilt.** Regenerating a dozen rows of `innerHTML`
  once a second reallocates every node, drops the scroll position and kills any
  text selection. Only cells whose text actually changed are touched.

Measured in Chromium with every cell changing every frame: `renderWatchlist`
takes 0.5ms median and 0.7ms worst against a 16.7ms budget, no long tasks, and
a node count flat at 594 over 40 seconds.

A symbol the venue stops returning goes **grey**, not missing. A row that
vanishes and reappears makes the list flicker and loses your place.

### The trading loop runs in the terminal

Until recently it did not, and that mattered: the mode switch changed a broker
that nothing was attached to, so pressing LIVE changed a label and traded
nothing. The bot lived behind `python -m godalgo live`, which is not what
someone running a double-clicked executable is going to do.

The terminal now owns a `LiveEngine`. The constraints are unchanged:

* **The UI still cannot place an order.** It starts and stops a session; every
  order decision belongs to the engine exactly as it does headless. There is
  still one path to the exchange.
* **The engine is autonomous, not a remote control.** Nothing in the interface
  can push a target weight, size, or side into it.
* **A display failure cannot break trading.** Fills reach the screen through an
  observer whose exceptions are swallowed inside the engine.

**Dry run is a real session.** The full pipeline runs — data, regime, signals,
sizing, risk, routing — and the broker discards the orders at the last step. So
you can watch what the bot *would* have done, against live prices, before
funding anything.

The header carries a session pill separate from the mode pill, because "which
broker" and "is the loop actually running" are different questions and the
second is the one a seemingly idle bot leaves unanswered:

| Pill | Meaning |
|---|---|
| `VIEWER` | No loop attached (demo mode) |
| `WARMING UP` | Running, but below the strategies' warm-up — it cannot signal yet |
| `TRADING` | Running and able to act |
| `STOPPED` | The loop died; the reason is in Activity |

`WARMING UP` and `TRADING` are distinguished deliberately: warming up resolves
itself, whereas a warmed-up bot placing no orders is a bot deciding not to
trade, and those need very different responses from you.

### Status lamps and the activity log

Four lamps in the header, deliberately not one: an unreachable venue, stale
data, a rejected key and an unconfigured Telegram are four different problems
needing four different actions, and sharing an indicator would hide that.

| Lamp | Green when | Other states |
|---|---|---|
| `LINK` | the browser is streaming from the terminal | cyan pulse while reconnecting |
| `VENUE` | the exchange is reachable | red with the reason |
| `DATA` | prices are arriving | amber if they go stale |
| `KEY` | a stored key passed its balance read | amber = stored but never tested; cyan pulse = testing; grey = no key |
| `TG` | Telegram is connected | grey = not configured |

The `KEY` lamp has four states rather than two on purpose. "Never tested" and
"tested and failed" are different facts, and a single grey lamp meaning both
tells you nothing — which is exactly what it did.

The **Activity** panel is the running record: every venue call, credential
change, mode switch and failure, with the remedy rather than just the symptom.
It exists because a bot that is working and a bot that is silently doing
nothing look identical from the outside, and the second is the expensive case.

Failures are named, not lumped together. Binance returns the same shape of
error for a wrong secret, a drifted clock and an IP that is not on the
allow-list, and the fix differs for each:

| What you see | What it actually is |
|---|---|
| `ip_not_allowed` (`-2015`) | The key is fine; your public IP is not on its allow-list |
| `clock_skew` (`-1021`) | Your PC's clock has drifted — nothing about the key will fix it |
| `bad_signature` (`-1022`) | The secret is wrong, often a stray space |
| `geo_blocked` (HTTP 451) | The venue refused the region; a different venue is needed |

### Telegram

Connect it from the Telegram panel: paste the bot token from **@BotFather** and
your chat id from **@userinfobot**, and it sends a test message immediately —
a settings form that accepts anything and reports nothing teaches you it worked
when it did not. The token is stored in the same owner-only file as the
exchange keys, is never returned over HTTP, and is never logged; the send URL
embeds it, so failures log a status code and never a URL.

### What it shows

| Panel | Content |
|---|---|
| **Neural cluster** | Every position as a floating neuron. **Orange** = open, **green** = closed in profit, **red** = closed at a loss. Size scales with notional (square-root, so one large trade doesn't swallow the panel). Synapses link nearby positions — amber for the same instrument. Click any neuron for its full record: instrument, side, open, close, mark, duration, fees, net P&L, and the strategy, regime and conviction that opened it. |
| **Health laser** | A single 0–1 score driven by connection, data staleness, halt state, reconnects, errors and drawdown. Multiplicative, so one failure shows through. Stale data is weighted heaviest — it's the failure that looks most like normality. |
| **P&L** | Net / realised / unrealised, return, win rate, profit factor, and a live equity curve. |
| **Journal** | Every closed trade, plus completed days. Rolls over at **UTC midnight** and pushes the daily summary to Telegram. |
| **Connections** | Add exchange API keys and check Telegram wiring. |

Open positions stay **orange** rather than being coloured by running P&L — an
open position hasn't made or lost anything yet, and colouring it by an
unrealised number invites treating a paper gain as a result.

### Telegram

```bash
export GODALGO_TELEGRAM_TOKEN="123456:AA..."
export GODALGO_TELEGRAM_CHAT_ID="987654321"
```

The daily digest fires automatically at UTC rollover; "Send digest now" in the
Connections panel pushes the current day on demand.

### Security

This process holds keys that can move money and serves HTTP with **no
authentication**, so:

| Control | Behaviour |
|---|---|
| **Bind address** | `127.0.0.1` only — `_assert_loopback` *rejects* `0.0.0.0` or any LAN address rather than warning about it |
| **Key storage** | `~/.godalgo/credentials.json`, mode `0600`, created owner-only before any secret is written |
| **Over the wire** | The API returns a masked view only (`KEY1******7890`). Secrets are never sent to the browser, in any form |
| **Trading** | A stored key is `trade_enabled=false` unless explicitly ticked — pasting a key into a form cannot by itself authorise orders |
| **Logs** | No key material at any level; Telegram errors log a status code, never the URL, which contains the token |
| **The UI cannot trade** | It is a read-only view. It never places orders or mutates engine state — a UI that can trade is a second, untested path to the exchange |

Secrets never leave your machine.

## Scanning and autonomy

The bot scans a universe, trades what qualifies, sizes off buying power, and
retunes itself — all without manual input.

```bash
python -m godalgo scan --timeframe 1h --top 25          # rank a universe
python -m godalgo live --scan --max-symbols 4           # trade what qualifies
```

`--scan` runs the fleet: the supervisor ranks the universe, starts a driver per
selected symbol, and rotates as conditions change. Without it `live` trades the
single `--symbol` you name.

### What the scanner rejects, and why that is the point

Filters run cheapest-first, and each rejects more than the next:

| Filter | Rejects |
|---|---|
| **Liquidity** | Thin books that fail on execution whatever the signal says |
| **Volatility band** | Too quiet to clear costs; too violent to size |
| **Feasibility** | **Expected edge below round-trip cost** |
| **Regime clarity** | No readable regime |
| **Correlation** | Symbols that are the same trade as one already selected |

The feasibility filter is the one most scanners omit. A symbol can be the
strongest trend in the universe and still be untradeable because its moves are
smaller than its spread — ranking on signal strength alone selects exactly
those instruments.

The correlation pass is not optional in crypto. On a test universe of BTC/ETH/SOL
clones the scanner selected **one** and rejected the other two as *"correlated
with a selection"*. Without it you hold one bet three times while believing you
are diversified.

### The fleet

`PortfolioDriver` runs one engine and one driver **per symbol**, sharing a
single exchange connection. Each engine stays single-symbol and knows nothing
about the others; everything portfolio-wide reaches it through two hooks the
supervisor owns — a target clamp and a buying-power source. Rewriting one engine
to hold N symbols would have meant re-deriving per-symbol state throughout, and
every existing test of the single-symbol path would have stopped covering what
actually runs.

Starting drivers is the easy half. **Stopping one that still holds a position is
where a multi-symbol bot goes wrong**: cancel the task and the position is left
open with nothing managing its stop. So retirement flattens first, and a failed
flatten *keeps the driver running* rather than orphaning the position.

### Portfolio supervision

`PortfolioSupervisor` runs one engine per selected symbol and owns everything
only meaningful across them:

- **Aggregate gross exposure** — a per-symbol cap says nothing about the total;
  three engines each at "20% of equity" is 60% of the account
- **Buying power** divided among engines rather than promised in full to each
- **Universe rotation** with a minimum hold, so a symbol on the selection
  threshold is not churned in and out at full cost
- **Retiring flattens first** — dropping the engine that owns a position leaves
  it open with nothing managing its stop

Reductions are never blocked by an exposure limit. A limit that prevents risk
being *reduced* is not a risk limit.

### Autonomous retuning

`Autopilot` refits parameters on a schedule while trading and swaps in only what
clears the same promotion gate a human run faces.

| Permitted | Forbidden |
|---|---|
| Propose parameters within declared bounds | Lower its own acceptance criteria |
| Swap after passing the gate | Swap under an open position |
| Search on a worker thread | Touch `RiskLimits` |

Criteria are a frozen dataclass and Autopilot holds no reference to
`RiskLimits`. **The bot can learn how to trade; it cannot learn to take more
risk.** Off by default — self-modification is opt-in, like live trading.

### Stops and sizing

| Mechanism | Question it answers |
|---|---|
| **Initial stop** (ATR) | How much am I willing to lose being wrong? |
| **Break-even move** | When does this trade stop being able to hurt me? |
| **Trailing stop** | How much open gain will I give back? |

All in ATR multiples, not percentages, so one configuration behaves sensibly on
a quiet pair and a violent one. **The ratchet is structural** — a stop can only
move toward price. Widening one as price approaches is how a bounded loss
becomes unbounded, so it is impossible rather than discouraged.

Size follows from the stop:

```
quantity = equity × risk_per_trade / |entry − stop|
```

Every trade risks the same fraction regardless of stop width — a 4% stop and a
1% stop both risk 1%. Sizing by notional instead makes risk-per-trade a function
of volatility, which is the thing being controlled. Sizing off *current* equity
is what compounds the account and what makes drawdowns self-limiting; a steeper
drawdown taper sits on top, floored so the system keeps enough size to trade
back. Orders cap at 95% of available buying power.

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

### WebSocket driver

`WebSocketDriver` runs the bot off ccxt.pro streams (76 venues support them).
Four independent tasks, deliberately not one loop:

| Task | Does | Why separate |
|---|---|---|
| trade loop | `watch_trades` → `ingest_tick` | cheap and sync, must never block |
| book loop | `watch_order_book` → `on_book` | same |
| decision loop | runs the expensive decision | regime refit is ~100ms; inline it would stall socket reads and drop ticks exactly when the market moves |
| watchdog | timer-driven staleness check | a watchdog woken by data cannot detect data not arriving |

The bar signal is a **single-slot latch, not a queue**. If bars complete faster
than decisions finish, the right move is to act once on the newest state — a
queued decision computed three bars ago would trade on information the market
has already left behind.

Reconnection uses exponential backoff with jitter (so two streams failing
together don't reconnect in lockstep and hammer the venue). `--max-reconnects 0`
retries forever; a positive value halts once exhausted, which is safer than a
bot reconnecting indefinitely while holding a position nobody is watching.

Verified end-to-end against a simulated exchange (real venues are unreachable
from the dev sandbox): 142 trades → 47 bars → 24 decisions (23 coalesced by the
latch) → 6 orders → 7 fills, and on shutdown the halt path cancelled and
flattened the position.

### Safety

| Control | Behaviour |
|---|---|
| **Trading mode** | `DRY_RUN` by default — computes orders, sends nothing |
| **Live arming** | Needs `arm=True` **and** a credential explicitly permitted to trade (the per-key tick, or `GODALGO_ARM_LIVE` for an environment key) |
| **Staleness watchdog** | Flattens if market data stops (a silent feed is worse than a bad signal) |
| **Reconciliation** | Venue is the source of truth; drift beyond tolerance halts |
| **Ambiguous sends** | Reported `UNKNOWN`, never "rejected" — then halt and reconcile |
| **Kill switch** | Cancels resting orders *then* flattens, in that order |
| **In-flight dedupe** | One order per symbol; a signal cannot become a double position |
| **Stale book** | Refuses to price passively against a book older than 10s |
| **Decision timeout** | A wedged decision halts rather than freezing the bot in position |
| **Venue precision** | Order size and price quantised to the venue's own rules, read from `load_markets()` |
| **Preflight** | `godalgo preflight` validates venue, credentials, symbol, limits and fees before any order |

Credentials come from the environment or the terminal's owner-only store
(`~/.godalgo/credentials.json`, mode 0600 on POSIX and a user-only ACL on
Windows, where permission bits do nothing). They are never accepted as
positional arguments, never returned over HTTP — the API serves a masked view
only — and never logged at any level.

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
| `ui/` | Local terminal: position tracking, journal, Telegram, credentials, server |
| `feasibility.py` | Can this configuration trade at this frequency? |
| `data/stream.py` | Tick → bar aggregation on wall-clock boundaries |
| `data/scanner.py` | Universe ranking: liquidity, feasibility, correlation |
| `portfolio/supervisor.py` | Multi-symbol exposure, buying power, rotation |
| `execution/portfolio_driver.py` | Runs one driver per symbol; rotates the fleet |
| `risk/stops.py` | Initial, break-even and trailing stops — ratcheted |
| `evolve/autopilot.py` | Retunes itself while trading, behind the gate |
| `execution/mode.py` | Guarded dry/paper/live switching |
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

**Timeframe matters more than it looks.** On 1-minute bars a crypto pair
annualises to ~135% volatility, so the default 20% vol target yields a position
scalar of ~0.15 — positions too small to clear the minimum trade size, and the
bot correctly declines to trade at all. That is the system telling you its edge
is not at that frequency. Either raise `target_annual_vol` deliberately, or use
longer bars. Do not "fix" it by lowering the cost gate.

Before any real capital: run in `paper` against live data long enough to confirm
real fill rates resemble the paper model's (post-only fill assumptions are the
most likely thing to be wrong), and verify the kill switch actually flattens on
a venue rather than only in simulation.

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
