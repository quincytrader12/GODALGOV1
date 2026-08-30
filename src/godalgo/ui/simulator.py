"""Synthetic feed for the terminal.

Runs the UI without a venue or a live engine, so the display can be developed,
demonstrated, and inspected on a machine with no exchange access.

It fabricates *positions*, not market structure. Nothing here is a backtest and
nothing it produces is evidence about strategy performance -- it exists so the
cluster has neurons to draw and the panels have numbers to lay out.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from godalgo.ui.server import UIBridge

__all__ = ["Simulator"]

_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
_REGIMES = ["trending", "mean_reverting", "indeterminate"]
_STRATEGIES = ["momentum", "mean_reversion", "blend"]


@dataclass
class Simulator:
    """Drives a ``UIBridge`` with fabricated fills."""

    bridge: UIBridge
    seed: int = 7
    tick_seconds: float = 0.9
    _rng: random.Random = field(init=False)
    _prices: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._prices = {"BTC/USDT": 64_000.0, "ETH/USDT": 3_200.0,
                        "SOL/USDT": 148.0, "AVAX/USDT": 36.0}

    def seed_history(self, count: int = 26) -> None:
        """Backfill closed positions so the terminal is populated on first paint."""
        now = datetime.now(UTC)
        for i in range(count):
            symbol = self._rng.choice(_SYMBOLS)
            price = self._prices[symbol] * (1 + self._rng.uniform(-0.05, 0.05))
            qty = round(self._rng.uniform(0.02, 0.6) * (60_000 / price), 6)
            side = 1 if self._rng.random() > 0.42 else -1
            opened = now - timedelta(minutes=(count - i) * 27 + self._rng.randint(0, 20))

            self.bridge.tracker.on_fill(
                symbol, side * qty, price, fee=qty * price * 0.0002, moment=opened,
                strategy=self._rng.choice(_STRATEGIES),
                regime=self._rng.choice(_REGIMES),
                conviction=round(self._rng.uniform(0.05, 0.55), 3),
            )
            # A slight positive drift on exits, so the demo shows a mix of
            # outcomes rather than a uniformly red or green cluster.
            move = self._rng.gauss(0.0045, 0.011) * side
            exit_price = price * (1 + move)
            closed = self.bridge.tracker.on_fill(
                symbol, -side * qty, exit_price, fee=qty * exit_price * 0.0002,
                moment=opened + timedelta(minutes=self._rng.randint(6, 90)),
            )
            if closed is not None:
                self.bridge.journal.record(closed, equity=self.bridge.equity)
                self.bridge.update_equity(self.bridge.equity + closed.net_pnl)

    async def run(self) -> None:
        """Tick forever: walk prices, open and close positions, update state."""
        self.bridge.connected = True
        while True:
            now = datetime.now(UTC)
            self.bridge.last_data_at = now
            self.bridge.decisions_run += 1

            for symbol in _SYMBOLS:
                self._prices[symbol] *= 1 + self._rng.gauss(0, 0.0016)
                self.bridge.tracker.mark(symbol, self._prices[symbol])

            symbol = self._rng.choice(_SYMBOLS)
            price = self._prices[symbol]

            if symbol in self.bridge.tracker.open_positions and self._rng.random() < 0.3:
                position = self.bridge.tracker.open_positions[symbol]
                signed = position.quantity if position.side == "long" else -position.quantity
                self.bridge.record_fill(
                    symbol, -signed, price, fee=abs(signed) * price * 0.0002, moment=now
                )
                self.bridge.update_equity(
                    self.bridge.starting_equity
                    + self.bridge.tracker.realised_total
                )
            elif symbol not in self.bridge.tracker.open_positions and self._rng.random() < 0.35:
                side = 1 if self._rng.random() > 0.45 else -1
                qty = round(self._rng.uniform(0.03, 0.5) * (60_000 / price), 6)
                self.bridge.record_fill(
                    symbol, side * qty, price, fee=qty * price * 0.0002, moment=now,
                    strategy=self._rng.choice(_STRATEGIES),
                    regime=self._rng.choice(_REGIMES),
                    conviction=round(self._rng.uniform(0.05, 0.55), 3),
                )

            self.bridge.symbol = symbol
            self.bridge.regime = self._rng.choice(_REGIMES)
            self.bridge.conviction = round(self._rng.uniform(-0.5, 0.5), 3)
            self.bridge.target_weight = round(self.bridge.conviction * 0.3, 4)
            self.bridge.current_weight = round(self.bridge.target_weight * 0.85, 4)
            self.bridge.update_equity(
                self.bridge.starting_equity
                + self.bridge.tracker.realised_total
                + self.bridge.tracker.unrealised_total
            )

            await asyncio.sleep(self.tick_seconds)
