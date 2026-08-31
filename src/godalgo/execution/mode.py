"""Switching between dry run, paper, and live.

Changing mode swaps the broker underneath a running engine, which makes it the
most dangerous operation in the system -- more dangerous than placing an order,
because an order is bounded and a bad mode switch silently invalidates every
position the engine believes it holds.

Three rules, each closing a specific way this goes wrong:

**Every switch flattens first.** A paper position is not a real one. Switching
to live while holding one leaves the engine convinced it owns something the
venue has never heard of, and its next decision is sized against that fiction.
Switching *out* of live while holding one abandons real exposure with nothing
managing its stop. Flattening is the only state from which both brokers agree.

**Live needs more than a click.** Two independent things must hold: a stored
credential the operator explicitly marked as permitted to place orders, and a
typed confirmation phrase. A UI button that arms real money on one click is a
UI button that arms real money by accident. Dropping back to paper needs
neither -- reducing risk should never be harder than taking it.

There are two ways to supply a key, and **each carries its own consent
record**:

* From the terminal, consent is the ``trade_enabled`` flag ticked on that
  specific key. A key added to read market data stays read-only.
* From the environment, there is no per-key flag to tick, so the arming
  variable ``GODALGO_ARM_LIVE`` remains that path's consent record.

The environment gate used to apply to *both* paths, and that was wrong for the
UI one: the terminal ships as a double-clicked executable, where "set an
environment variable first" is not friction but a wall. Friction the legitimate
operator cannot clear is not a safety feature -- it is an outage, and it pushes
people towards running the bot in ways nobody designed. Removing it there was
only defensible because the tick replaced it; the env path kept the variable
because nothing replaced it.

**Nothing is inferred.** Credentials being present does not mean live is
wanted. The presence of an API key is not consent -- a key added to read market
data stays read-only until someone ticks the box on that key.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from godalgo.execution.broker import Broker, DryRunBroker, PaperBroker
from godalgo.execution.types import TradingMode

__all__ = ["ModeChange", "ModeController", "ModeSwitchError"]

logger = logging.getLogger(__name__)

_ARM_ENV_VAR = "GODALGO_ARM_LIVE"
_ARM_TOKEN = "I_UNDERSTAND_THIS_TRADES_REAL_MONEY"
_CONFIRM_PHRASE = "GO LIVE"


class ModeSwitchError(RuntimeError):
    """Raised when a mode change is refused."""


@dataclass(frozen=True, slots=True)
class ModeChange:
    """A completed switch, for the audit trail."""

    at: datetime
    from_mode: TradingMode
    to_mode: TradingMode
    flattened: bool
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "at": self.at.isoformat(),
            "from": self.from_mode.value,
            "to": self.to_mode.value,
            "flattened": self.flattened,
            "note": self.note,
        }


@dataclass
class ModeController:
    """Owns the active broker and performs guarded switches.

    Args:
        mode: Starting mode. Always DRY_RUN unless deliberately set.
        equity: Starting equity for the simulated brokers.
        on_broker_change: Called with the new broker after a successful switch,
            so engines can rebind. Supplied by the caller because the
            controller must not need to know how many engines exist.
        flatten: Async callable that closes every open position. Required for
            any switch that could leave exposure behind.
    """

    mode: TradingMode = TradingMode.DRY_RUN
    equity: float = 10_000.0
    on_broker_change: Callable[[Broker], None] | None = None
    flatten: Callable[[], object] | None = None
    tradeable_credential: Callable[[], object | None] | None = None
    """Returns the stored credential permitted to place orders, or None.

    Supplied by the caller rather than read here, because the controller must
    not depend on the UI's storage layer -- a headless run has no credential
    store and must still be able to go live from the environment.
    """

    history: list[ModeChange] = field(default_factory=list)
    _broker: Broker | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self._broker is None:
            self._broker = self._build(self.mode)

    @property
    def broker(self) -> Broker:
        assert self._broker is not None
        return self._broker

    # -- capability reporting ---------------------------------------------

    def _credential(self) -> object | None:
        """The credential live would trade with, if any."""
        if self.tradeable_credential is None:
            return None
        try:
            return self.tradeable_credential()
        except Exception:
            logger.exception("could not read the credential store")
            return None

    @property
    def live_armed(self) -> bool:
        """Whether a consented credential exists by either route."""
        return self._credential() is not None or self._env_armed()

    def status(self) -> dict[str, object]:
        """Mode state for the UI. Contains no secret material.

        ``live_blockers`` is the substance. An unavailable live switch has to
        say *which* requirement is missing: "unavailable" on its own sends
        people to re-paste keys that were never the problem.
        """
        credential = self._credential()
        env_armed = self._env_armed()

        blockers: list[str] = []
        source: str | None = None
        testnet = False

        if credential is not None:
            source = getattr(credential, "exchange_id", None)
            testnet = bool(getattr(credential, "testnet", False))
        elif env_armed:
            source = "environment"
        elif self._env_credentials_present():
            # A key is there but nobody said it may trade. Naming the exact
            # remedy matters: this is the state an operator lands in after
            # exporting credentials and expecting that to be enough.
            blockers.append(
                f"credentials are in the environment but {_ARM_ENV_VAR} is not "
                "set; or add the key in the terminal and tick 'allow this key "
                "to place orders'"
            )
        else:
            blockers.append(
                "no exchange key is marked 'allow this key to place orders'"
            )

        return {
            "mode": self.mode.value,
            "live_armed": not blockers,
            "live_available": not blockers,
            "live_blockers": blockers,
            "live_source": source,
            "live_testnet": testnet,
            "confirm_phrase": _CONFIRM_PHRASE,
            "history": [c.to_dict() for c in self.history[-10:]],
        }

    @staticmethod
    def _env_credentials_present() -> bool:
        return bool(
            os.environ.get("GODALGO_API_KEY") and os.environ.get("GODALGO_API_SECRET")
        )

    @classmethod
    def _env_armed(cls) -> bool:
        """The environment path's consent record: credentials *and* the token.

        Both, because a key sitting in a shell profile is not a decision to
        trade with it.
        """
        return (
            cls._env_credentials_present()
            and os.environ.get(_ARM_ENV_VAR) == _ARM_TOKEN
        )

    # -- switching ---------------------------------------------------------

    async def switch(
        self, target: TradingMode, *, confirm: str | None = None, note: str = ""
    ) -> ModeChange:
        """Change mode, flattening first.

        Args:
            target: Mode to switch to.
            confirm: Confirmation phrase. Required only when entering LIVE.
            note: Free text recorded with the change.

        Raises:
            ModeSwitchError: If live is requested without arming, credentials,
                or confirmation; or if positions could not be flattened.
        """
        if target is self.mode:
            return ModeChange(datetime.now(UTC), self.mode, target, False, "no change")

        if target is TradingMode.LIVE:
            self._assert_live_permitted(confirm)

        # Flatten before the swap, never after. After the swap the old
        # positions belong to a broker the engine no longer holds, so nothing
        # can close them.
        flattened = False
        if self.flatten is not None:
            try:
                result = self.flatten()
                if hasattr(result, "__await__"):
                    await result
                flattened = True
            except Exception as exc:
                # Refusing the switch is the safe failure: staying in the
                # current mode keeps whatever is open under the broker that
                # actually owns it.
                logger.exception("could not flatten before mode switch")
                raise ModeSwitchError(
                    f"refusing to switch mode: positions could not be closed ({exc})"
                ) from exc

        previous = self.mode
        self._broker = self._build(target)
        self.mode = target

        if self.on_broker_change is not None:
            self.on_broker_change(self._broker)

        change = ModeChange(datetime.now(UTC), previous, target, flattened, note)
        self.history.append(change)
        level = logger.warning if target is TradingMode.LIVE else logger.info
        level("mode changed %s -> %s (flattened=%s)", previous.value, target.value, flattened)
        return change

    def _assert_live_permitted(self, confirm: str | None) -> None:
        if self._credential() is None and not self._env_armed():
            if self._env_credentials_present():
                raise ModeSwitchError(
                    f"credentials are present but not armed: set {_ARM_ENV_VAR}="
                    f"{_ARM_TOKEN}, or add the key in the terminal and tick "
                    "'allow this key to place orders'"
                )
            raise ModeSwitchError(
                "no exchange key is permitted to place orders. Add a key in the "
                "terminal and tick 'allow this key to place orders' on it."
            )
        if (confirm or "").strip().upper() != _CONFIRM_PHRASE:
            raise ModeSwitchError(
                f"live trading requires the confirmation phrase {_CONFIRM_PHRASE!r}"
            )

    def _build(self, mode: TradingMode) -> Broker:
        if mode is TradingMode.LIVE:
            from godalgo.execution.live import LiveBroker

            # LiveBroker re-checks the credential's trade permission itself.
            # Two independent checks on the one irreversible path is deliberate.
            return LiveBroker(arm=True, credential=self._credential())
        if mode is TradingMode.PAPER:
            return PaperBroker(starting_equity=self.equity)
        return DryRunBroker(starting_equity=self.equity)
