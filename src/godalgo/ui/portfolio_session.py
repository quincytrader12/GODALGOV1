"""Many symbols, scanned, decided and executed.

The terminal traded one instrument. That was never the design -- the scanner,
the supervisor and the allocator all exist to run a book -- and leaving them
unconnected meant the visible product was a single-symbol bot with a lot of
unused machinery behind it.

This is the loop that joins them:

    feed ticks -> engines (one per symbol) -> supervisor -> book -> broker
                     ^                                              |
                     +--------- scanner rotates the universe -------+

Four things it is careful about, each of which is a way to lose money that a
single-symbol design cannot even express:

* **Aggregate exposure.** Three engines each within their own 20% cap is 60%
  of the account, not 20%. The supervisor owns the portfolio ceiling and
  divides buying power by the concurrency limit rather than by the current
  count -- dividing by the live count lets the first symbol admitted claim the
  whole book and forces every later one to trade at a fraction of it.

* **Retiring means flattening.** A symbol dropping out of the universe must
  have its position closed, not merely stop being watched. Stopping the engine
  that owns a position leaves it open with nothing managing its stop, which is
  the worst outcome available here and the easiest mistake to make.

* **One data path.** Every engine is fed from the market feed's existing
  batched poll rather than opening its own connection. Twelve symbols with
  twelve sockets is twelve reconnection paths to maintain and twelve ways to
  be quietly stale.

* **Ticks are observations, not repeats.** The feed polls on a timer, so the
  same price arriving twice is one observation seen twice. Forwarding both
  would inflate volume and manufacture bars out of a still market.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from godalgo.execution.types import TradingMode

if TYPE_CHECKING:
    from godalgo.ui.server import UIBridge

__all__ = ["PortfolioSession"]

logger = logging.getLogger(__name__)

_SEED_BARS = 500
_TICK_SECONDS = 1.0
"""How often the session looks at the feed's prices.

Faster than the feed polls, so a new price is dispatched promptly rather than
waiting out a second cadence, and cheap because it reads a dictionary.
"""


@dataclass
class PortfolioSession:
    """A scanner-driven book, running inside the terminal.

    Interchangeable with ``TradingSession``: same ``start``/``stop``, so the
    mode switch drives either without knowing which it holds.
    """

    bridge: UIBridge
    exchange_id: str = "alpaca"
    bar_seconds: int = 60
    book: Any = None
    max_concurrent: int = 4
    rescan_hours: float = 6.0

    supervisor: Any = field(default=None, init=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _broker: Any = field(default=None, init=False, repr=False)
    _history: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_price: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_cap_reason: dict[str, str] = field(
        default_factory=dict, init=False, repr=False,
    )

    # -- status ------------------------------------------------------------
    #
    # The same shape ``TradingSession`` reports, because the mode switch, the
    # session endpoint and the front end all read it and none of them should
    # need to know which kind of session they hold.

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def warmed_up(self) -> bool:
        """Whether any engine has enough history to produce a signal.

        Reported because "warming up" and "seeing no opportunity" look
        identical from outside and mean completely different things: one
        resolves itself, the other is the bot deciding not to trade.

        Any rather than all: a book where one symbol is ready is a book that
        can act, and waiting for the slowest would report a working bot as
        blind.
        """
        engines = self.supervisor.engines.values() if self.supervisor else ()
        return any(e.bars.n_complete > e._warmup for e in engines)

    @property
    def symbol(self) -> str:
        """The headline instrument: the largest live position.

        A single-symbol session had one answer to this. A book's answer is
        whichever position is doing the most, which is what the header should
        describe.
        """
        engines = self.supervisor.engines if self.supervisor else {}
        if not engines:
            return self.bridge.symbol
        return max(engines, key=lambda s: abs(engines[s].state.current_weight))

    def status(self) -> dict[str, Any]:
        engines = self.supervisor.engines if self.supervisor else {}
        return {
            "running": self.running,
            "symbol": self.symbol,
            "bar_seconds": self.bar_seconds,
            "warmed_up": self.warmed_up,
            "bars": max(
                (e.bars.n_complete for e in engines.values()), default=0,
            ),
            "symbols": sorted(engines),
            "gross_exposure": (
                self.supervisor.gross_exposure if self.supervisor else 0.0
            ),
            "max_concurrent": self.max_concurrent,
            "engine": (
                engines[self.symbol].state.snapshot()
                if self.symbol in engines else {}
            ),
            "portfolio": (
                self.supervisor.state.snapshot() if self.supervisor else None
            ),
        }

    # -- lifecycle ---------------------------------------------------------

    async def start(self, broker: Any) -> None:
        """Start trading the scanned universe against ``broker``."""
        await self.stop()

        from godalgo.data.scanner import MarketScanner
        from godalgo.portfolio.supervisor import PortfolioSupervisor, SupervisorConfig

        self._broker = broker
        mode = _mode_of(broker)

        self.supervisor = PortfolioSupervisor(
            config=SupervisorConfig(
                max_concurrent=self.max_concurrent,
                rescan_hours=self.rescan_hours,
            ),
            scanner=MarketScanner(),
            history=lambda: self._history,
            tickers=self._tickers,
            make_engine=self._make_engine,
            equity=lambda: self.bridge.equity,
        )

        self.bridge.events.record(
            "warn" if mode is TradingMode.LIVE else "info", "engine",
            f"portfolio session starting in {mode.value}",
            f"scanning {len(self.bridge.universe)} symbols, trading at most "
            f"{self.max_concurrent} at once",
        )
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop trading, flattening every engine on the way out."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        supervisor = self.supervisor
        if supervisor is not None:
            for symbol in list(supervisor.engines):
                # Through the supervisor so retirement flattens rather than
                # merely detaches. A stopped engine still owning a position is
                # a position with nothing managing its stop.
                with contextlib.suppress(Exception):
                    await supervisor._retire(symbol)
        self.supervisor = None
        self._last_price.clear()

    # -- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        """Rotate, dispatch, publish. Never raises out of the task."""
        try:
            await self._rotate()
            while True:
                await self._dispatch()
                await self._publish()
                if self.supervisor is not None and self.supervisor.due_for_scan():
                    await self._rotate()
                await asyncio.sleep(_TICK_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            self.bridge.events.error(
                "engine", "the portfolio loop stopped",
                f"{type(exc).__name__}: {exc}"[:300],
            )
            logger.exception("portfolio session failed")

    async def _rotate(self) -> None:
        """Re-scan and admit or retire symbols."""
        if self.supervisor is None:
            return
        await self._refresh_history()
        try:
            admitted, retired = await self.supervisor.rotate()
        except Exception as exc:  # noqa: BLE001 - a bad scan is not fatal
            self.bridge.events.warn(
                "engine", "universe scan failed",
                f"{type(exc).__name__}: {exc}"[:200],
            )
            return

        scan = self.supervisor.state.last_scan
        if admitted or retired:
            self.bridge.events.info(
                "engine", "universe rotated",
                f"admitted {', '.join(admitted) or 'nothing'}; "
                f"retired {', '.join(retired) or 'nothing'}",
            )
        elif scan is not None and not scan.selected:
            # "Nothing matched" is only true when something was scanned. This
            # distinguishes it from "could not run" and from "not finished".
            self.bridge.events.info(
                "engine", "nothing passed the scan",
                scan.summary if hasattr(scan, "summary") else
                f"{len(self.bridge.universe)} symbols examined, none tradeable "
                f"— refusing is the expected outcome, not a fault",
            )

        for symbol in admitted:
            await self._seed(symbol)

    async def _dispatch(self) -> None:
        """Push new prices into the engine that owns each symbol."""
        if self.supervisor is None:
            return
        for symbol, engine in list(self.supervisor.engines.items()):
            price = self.bridge.prices.get(symbol)
            if not price or price <= 0:
                continue
            if self._last_price.get(symbol) == price:
                # The feed polls on a timer; the same price twice is one
                # observation seen twice, and forwarding both manufactures
                # volume and bars out of a still market.
                continue
            self._last_price[symbol] = price
            try:
                await engine.on_tick(price, moment=datetime.now(UTC))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one symbol, not the book
                self.bridge.events.warn(
                    "engine", f"{symbol} decision failed",
                    f"{type(exc).__name__}: {exc}"[:200],
                )

    # -- engines -----------------------------------------------------------

    def _make_engine(self, symbol: str) -> Any:
        """Build the engine for one symbol.

        Every engine shares the broker, so the venue sees one account, and
        carries the same strategies -- the book decides how much each gets,
        not which ideas it may have.
        """
        from godalgo.execution.engine import LiveEngine, LiveEngineConfig
        from godalgo.strategies.mean_reversion import MeanReversionStrategy
        from godalgo.strategies.momentum import MomentumStrategy

        engine = LiveEngine(
            self._broker,
            MomentumStrategy(),
            MeanReversionStrategy(),
            LiveEngineConfig(
                symbol=symbol, bar_seconds=self.bar_seconds,
                mode=_mode_of(self._broker),
            ),
        )
        engine.on_fill = self._on_fill
        return engine

    def _on_fill(
        self, symbol: str, signed_qty: float, price: float, fee: float
    ) -> None:
        self.bridge.record_fill(symbol, signed_qty, price, fee)
        self.bridge.events.good(
            "engine",
            f"{'bought' if signed_qty > 0 else 'sold'} {abs(signed_qty):.6g} {symbol}",
            f"at {price:,.4f}",
        )

    # -- data --------------------------------------------------------------

    def _tickers(self) -> dict[str, dict[str, float]]:
        """Liquidity per symbol, from the watchlist the feed already fills."""
        return {
            symbol: {
                "quoteVolume": row.quote_volume,
                "bid": row.price * (1 - row.spread_bps / 20_000.0),
                "ask": row.price * (1 + row.spread_bps / 20_000.0),
            }
            for symbol, row in self.bridge.watchlist.items()
            if row.price > 0
        }

    async def _refresh_history(self) -> None:
        """Bars for the whole universe, for the scanner to rank on.

        Fetched on the scan cadence rather than per tick: this is a request per
        symbol, and the scanner runs every few hours.
        """
        if self.exchange_id != "alpaca":
            return
        from godalgo.ui.alpaca_probes import client_for
        from godalgo.ui.session import _alpaca_timeframe

        credential = self.bridge.credentials.first_for("alpaca")
        if credential is None:
            self.bridge.events.warn(
                "engine", "cannot scan without an Alpaca key",
                "Alpaca serves market data to authenticated requests only. Add "
                "a key in Connections and the scanner will start ranking.",
            )
            return

        import pandas as pd

        client = client_for(credential)
        timeframe = _alpaca_timeframe(self.bar_seconds)
        failed: list[str] = []
        try:
            for symbol in self.bridge.universe:
                try:
                    rows = await client.bars(symbol, timeframe, limit=_SEED_BARS)
                except Exception:  # noqa: BLE001 - one symbol, not the scan
                    failed.append(symbol)
                    continue
                if not rows:
                    # No data is not the same as a failed fetch, and recording
                    # it as a permanent verdict is how a scanner kills itself:
                    # one credential-less launch must not blacklist a universe.
                    continue
                frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
                self._history[symbol] = frame[
                    ["open", "high", "low", "close", "volume"]
                ]
        finally:
            await client.close()

        if failed:
            self.bridge.events.warn(
                "engine", f"history unavailable for {len(failed)} symbol(s)",
                f"{', '.join(failed[:6])} — these are skipped this scan and "
                f"retried on the next, not marked dead",
            )

    async def _seed(self, symbol: str) -> None:
        """Prime a newly admitted engine so it can signal now."""
        engine = self.supervisor.engines.get(symbol) if self.supervisor else None
        frame = self._history.get(symbol)
        if engine is None or frame is None or frame.empty:
            return
        with contextlib.suppress(Exception):
            engine.seed_history(frame)

    # -- publishing --------------------------------------------------------

    async def _publish(self) -> None:
        """Copy portfolio state onto the bridge, and apply the book."""
        supervisor = self.supervisor
        if supervisor is None:
            return

        engines = supervisor.engines
        self.bridge.decisions_run = sum(e.state.bars_seen for e in engines.values())
        self.bridge.halted = any(e.state.halted for e in engines.values())
        self.bridge.halt_reason = next(
            (e.state.halt_reason for e in engines.values() if e.state.halt_reason),
            None,
        )
        self.bridge.current_weight = supervisor.gross_exposure
        self.bridge.target_weight = sum(
            e.state.target_weight for e in engines.values()
        )

        # The headline symbol follows the largest live position rather than a
        # fixed one, so the header describes what the book is actually doing.
        if engines:
            # Regime and conviction are deliberately not copied here.
            # ``EngineState`` does not carry them -- they live inside the
            # decision, not its result -- and inventing them from a field that
            # does not exist is how the first version of this loop crashed on
            # every publish. A number the display cannot source is one it must
            # not show.
            self.bridge.symbol = max(
                engines, key=lambda s: abs(engines[s].state.current_weight)
            )

        for symbol, price in self.bridge.prices.items():
            self.bridge.tracker.mark(symbol, price)

        self._apply_book(engines)
        self.bridge.portfolio = supervisor.state.snapshot()

    def _apply_book(self, engines: dict[str, Any]) -> None:
        """Cap each engine at what the allocator permits. Only ever reduces."""
        if self.book is None or not engines:
            return
        try:
            self.book.observe(self.bridge.prices)
            self.book.rebuild(
                list(engines),
                equity=self.bridge.equity,
                peak_equity=self.bridge.peak_equity,
                current={s: e.state.current_weight for s, e in engines.items()},
            )
        except Exception:  # noqa: BLE001 - sizing must not kill the loop
            logger.exception("could not rebuild the book")
            return

        for symbol, engine in engines.items():
            permitted, why = self.book.permitted_weight(
                symbol, engine.state.target_weight
            )
            if abs(permitted) >= abs(engine.state.target_weight) - 1e-9:
                continue
            if hasattr(engine, "set_weight_cap"):
                engine.set_weight_cap(abs(permitted))
            if self._last_cap_reason.get(symbol) != why:
                self._last_cap_reason[symbol] = why
                self.bridge.events.info("book", f"{symbol} capped by the book", why)


def _mode_of(broker: Any) -> TradingMode:
    """Read the mode from the broker's type rather than being told.

    A session told it is in paper while holding a live broker would be a
    display that disagrees with reality on the one fact that matters most.
    """
    name = type(broker).__name__.lower()
    if "dryrun" in name:
        return TradingMode.DRY_RUN
    if "paper" in name:
        return TradingMode.PAPER
    return TradingMode.LIVE
