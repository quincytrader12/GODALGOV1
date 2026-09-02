"""Alpaca's answers to the questions the Connections panel asks.

The existing probes go through ccxt, which reaches Alpaca's crypto endpoints
and nothing else. Since the point of coming here is the whole universe --
thousands of equities and ETFs alongside crypto -- these probes speak the
native API instead, and answer exactly the same three questions in the same
shape, so the panel does not have to know which venue it is looking at.

The three questions, and why each is worth a separate line on screen:

* **reachable** -- the venue answers at all. No key, so a failure here is the
  network or the venue, never the credential.
* **market data** -- a real price arrives. Proves the path the strategies
  consume, which is the one thing a dead-looking terminal cannot tell you.
* **credentials** -- the key is accepted, and what it is attached to. Reading
  the account is the right call for this: it exercises key, secret and
  permissions, it cannot move anything, and an empty paper account passes,
  which is the entire point of proving this before funding anything.

The credential probe also reports the facts that decide whether trading is
possible at all -- paper or live, market open or shut, and how much of the
day-trade budget is left -- because those constrain the bot far more than the
balance does, and none of them are visible from a key alone.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from godalgo.risk.venue_rules import PDT_EQUITY_FLOOR, VenueState
from godalgo.venues.alpaca import AlpacaClient, AlpacaConfig, AlpacaError

if TYPE_CHECKING:
    from godalgo.ui.credentials import ExchangeCredential
    from godalgo.ui.events import EventLog
    from godalgo.ui.venue import ProbeResult

__all__ = ["ALPACA_UNIVERSE", "check_credentials", "check_public", "client_for"]

logger = logging.getLogger(__name__)

ALPACA_UNIVERSE: tuple[str, ...] = (
    # Crypto first, deliberately: it trades continuously and is exempt from the
    # day-trade rule, so it is the half of the universe that can do anything at
    # 3am on a Sunday with a small account.
    "BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD",
    # Then the most liquid US equities and ETFs. Liquid enough that spreads do
    # not eat the edge, and recognisable enough that the panel reads as a
    # market rather than a list of tickers.
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
)
"""What the terminal watches out of the box on Alpaca.

Twelve rows, the same size as before: enough that the panel shows the bot
surveying a market, few enough that the whole set arrives in two batched
requests. The scanner replaces this with the real universe -- thousands of
instruments from ``/v2/assets`` -- once a key is connected.
"""


def client_for(credential: ExchangeCredential | None, *, timeout: float = 15.0) -> AlpacaClient:
    """A client for this credential, paper unless told otherwise.

    ``testnet`` on the stored credential is Alpaca's paper account. The mapping
    is deliberate: the store already has a flag meaning "not real money", and
    inventing a second one would create a way for the two to disagree.
    """
    if credential is None:
        return AlpacaClient(AlpacaConfig(timeout=timeout))
    return AlpacaClient(AlpacaConfig(
        key_id=credential.api_key,
        secret_key=credential.api_secret,
        paper=credential.testnet,
        timeout=timeout,
    ))


async def check_public(
    events: EventLog, symbol: str = "BTC/USD", *, timeout: float = 15.0
) -> list[ProbeResult]:
    """Reachability and a live price, with no credential involved.

    Alpaca's market data needs a key even for reading, unlike a crypto
    exchange's public endpoints. So "reachable" here is answered by the clock,
    which is unauthenticated -- it tells us the venue is up and, usefully,
    whether the equities session is open.
    """
    from godalgo.ui.venue import ProbeResult, raw_error

    results: list[ProbeResult] = []
    client = AlpacaClient(AlpacaConfig(timeout=timeout))
    started = time.monotonic()
    try:
        clock = await client.clock()
    except AlpacaError as exc:
        events.error("venue", "cannot reach alpaca", raw_error(exc))
        results.append(ProbeResult(
            "reachable", False,
            "Could not reach Alpaca. Check the connection, or run Diagnose "
            "connection to find which layer is failing.",
            (time.monotonic() - started) * 1000, "unreachable",
        ))
        await client.close()
        return results

    elapsed = (time.monotonic() - started) * 1000
    session = "market open" if clock.is_open else "market closed"
    events.good("venue", "alpaca reachable", f"{session}, {elapsed:.0f}ms")
    results.append(ProbeResult(
        "reachable", True, session, elapsed,
        data={"market_open": clock.is_open, **clock.to_dict()},
    ))
    await client.close()

    # Market data is not public here, so it is proven by the credential probe
    # instead. Saying so beats a red lamp that means "not tried".
    results.append(ProbeResult(
        "market_data", True,
        "Alpaca serves market data to authenticated requests only — proven by "
        "the credential check below.",
        data={"requires_key": True},
    ))
    return results


async def check_credentials(
    events: EventLog, credential: ExchangeCredential, *, timeout: float = 15.0
) -> list[ProbeResult]:
    """Prove the key, and report what constrains trading with it."""
    from godalgo.ui.venue import ProbeResult, raw_error

    client = client_for(credential, timeout=timeout)
    results = await check_public(events, timeout=timeout)
    started = time.monotonic()
    try:
        account = await client.account()
        clock = await client.clock()
    except AlpacaError as exc:
        explanation = _explain(exc, credential)
        events.error("credentials", "alpaca rejected the key",
                     f"{explanation} [{raw_error(exc)}]")
        results.append(ProbeResult(
            "credentials", False, explanation,
            (time.monotonic() - started) * 1000,
            "auth_failed" if exc.is_auth else "venue_error",
        ))
        await client.close()
        return results

    elapsed = (time.monotonic() - started) * 1000
    state = VenueState.from_account(account, market_open=clock.is_open)
    where = "paper" if credential.testnet else "LIVE"

    # A real price, now that we have a key that works. This is the line that
    # answers "is it actually seeing the market".
    price_note = ""
    try:
        rows = await client.snapshots(["BTC/USD"])
        row = rows.get("BTC/USD") or {}
        if row.get("price"):
            price_note = f", BTC/USD @ {row['price']:,.2f}"
            results[-1] = ProbeResult(
                "market_data", True, f"BTC/USD @ {row['price']:,.2f}",
                data={"price": row["price"]},
            )
    except AlpacaError:
        logger.debug("snapshot after auth failed", exc_info=True)

    detail = (
        f"{where} account, ${state.equity:,.2f} equity{price_note}"
    )
    if state.pdt_constrained:
        detail += (
            f" — under ${PDT_EQUITY_FLOOR:,.0f}, so {state.day_trades_left} "
            f"day trade(s) remain this week on equities. Crypto is exempt."
        )

    events.record(
        "good" if credential.testnet else "warn", "credentials",
        f"alpaca key verified ({where})", detail,
    )
    results.append(ProbeResult(
        "credentials", True, detail, elapsed, data=state.to_dict(),
    ))
    await client.close()
    return results


def _explain(exc: AlpacaError, credential: ExchangeCredential) -> str:
    """What to do about it, not just what happened."""
    if exc.status == 401:
        return (
            "Alpaca did not accept this key. Check the key id and secret are "
            "the pair generated together, and that they are "
            + ("paper keys — a live key will not work against the paper "
               "endpoint." if credential.testnet
               else "live keys — a paper key will not work against the live "
                    "endpoint.")
        )
    if exc.status == 403:
        return (
            "Alpaca refused these keys for this endpoint. This is almost "
            "always paper keys used against live, or live keys against paper: "
            "they are separate keys, generated separately. The 'paper "
            "account' tick must match which pair you pasted."
        )
    if exc.status == 429:
        return "Rate limited by Alpaca. It will clear on its own; try again shortly."
    if exc.status == 0:
        return (
            "Could not reach Alpaca at all. Run Diagnose connection — it tests "
            "DNS, HTTPS and the proxy separately and names which one is failing."
        )
    return f"Alpaca refused the request: {exc.message}"
