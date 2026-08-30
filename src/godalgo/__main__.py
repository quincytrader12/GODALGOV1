"""Command line entry point.

    python -m godalgo backtest --symbol BTC/USDT --timeframe 1h --limit 8000
    python -m godalgo evolve   --symbol BTC/USDT --candidates 16
    python -m godalgo ledger
    python -m godalgo live      --symbol BTC/USDT --bar-seconds 60 --mode paper
    python -m godalgo feasibility --symbol BTC/USDT --timeframe 1h
    python -m godalgo preflight   --symbol BTC/USDT
    python -m godalgo ui          --demo
    python -m godalgo scan        --timeframe 1h --top 20

Deliberately thin. It wires existing components together and prints results; it
holds no strategy logic of its own, so anything that works here works identically
when the agent runtime drives the same functions directly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from godalgo.backtest.engine import BacktestConfig, run_backtest
from godalgo.data.feed import OHLCVFeed, bars_per_day
from godalgo.evolve.promotion import PromotionLedger, evaluate_candidate
from godalgo.evolve.search import SearchConfig, propose_candidates
from godalgo.evolve.walkforward import walk_forward_evaluate
from godalgo.strategies.mean_reversion import MeanReversionParams, MeanReversionStrategy
from godalgo.strategies.momentum import MomentumParams, MomentumStrategy

DEFAULT_CACHE = Path("data/cache")
DEFAULT_LEDGER = Path("data/promotion_ledger.jsonl")


class DataUnavailable(RuntimeError):
    """Raised when market data could not be fetched, with an actionable message."""


def _load_bars(args: argparse.Namespace):
    """Fetch history, converting network failures into a clear message.

    Reachability is the single most common failure here -- a blocked egress, a
    geo-restricted venue, a wrong symbol -- and a raw ccxt traceback tells the
    user nothing about which of those it was.
    """
    import ccxt

    feed = OHLCVFeed(exchange_id=args.exchange, cache_dir=DEFAULT_CACHE)
    print(f"fetching {args.symbol} {args.timeframe} from {args.exchange} ...", file=sys.stderr)
    try:
        bars = feed.fetch(args.symbol, args.timeframe, limit=args.limit)
    except ccxt.BadSymbol as exc:
        raise DataUnavailable(
            f"{args.exchange} does not list {args.symbol!r}: {exc}"
        ) from exc
    except ccxt.NetworkError as exc:
        raise DataUnavailable(
            f"could not reach {args.exchange}: {exc}\n"
            f"  - check outbound network access to the exchange API\n"
            f"  - some venues are geo-restricted; try --exchange kraken\n"
            f"  - cached data is used when available (see {DEFAULT_CACHE})"
        ) from exc
    except ccxt.BaseError as exc:
        raise DataUnavailable(f"{args.exchange} rejected the request: {exc}") from exc

    if bars.empty:
        raise DataUnavailable(
            f"{args.exchange} returned no bars for {args.symbol} {args.timeframe}"
        )
    print(f"  {len(bars)} bars  {bars.index[0]} -> {bars.index[-1]}", file=sys.stderr)
    return bars


def cmd_backtest(args: argparse.Namespace) -> int:
    bars = _load_bars(args)
    config = BacktestConfig(bars_per_day=bars_per_day(args.timeframe))
    result = run_backtest(
        bars, MomentumStrategy(), MeanReversionStrategy(), config, args.symbol
    )
    print(result.summary())

    shares = result.frame["regime"].apply(lambda r: r.value).value_counts(normalize=True)
    print("\nregime share:")
    for name, share in shares.items():
        print(f"  {name:<16s} {share:6.1%}")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    """One search round: propose, walk forward, and put the winner to the gate."""
    bars = _load_bars(args)
    config = BacktestConfig(bars_per_day=bars_per_day(args.timeframe))

    candidates = propose_candidates(
        MomentumParams(),
        MeanReversionParams(),
        SearchConfig(n_candidates=args.candidates, seed=args.seed),
    )
    print(f"evaluating {len(candidates)} candidates ...", file=sys.stderr)

    result = walk_forward_evaluate(
        bars,
        candidates,
        lambda m, r: (MomentumStrategy(m), MeanReversionStrategy(r)),
        config,
        train_size=args.train_size,
        test_size=args.test_size,
        symbol=args.symbol,
    )
    if result.n_skipped:
        print(
            f"warning: {result.n_skipped} candidate/fold pairs were skipped "
            f"(warm-up longer than the fold?)",
            file=sys.stderr,
        )

    gate = evaluate_candidate(result, incumbent_sharpe=args.incumbent)
    print(gate.summary())

    ledger = PromotionLedger(DEFAULT_LEDGER)
    ledger.record(gate, symbol=args.symbol, note=args.note)
    print(f"\nrecorded to {DEFAULT_LEDGER}")
    return 0 if gate.promoted else 1


def cmd_ledger(args: argparse.Namespace) -> int:
    ledger = PromotionLedger(DEFAULT_LEDGER)
    entries = ledger.entries()
    if not entries:
        print("ledger is empty")
        return 0

    print(f"{len(entries)} decisions, {sum(e['promoted'] for e in entries)} promoted\n")
    for entry in entries[-args.tail :]:
        verdict = "PROMOTED" if entry["promoted"] else "rejected"
        m = entry["metrics"]
        print(
            f"{entry['timestamp'][:19]}  {entry['symbol']:<12s} {verdict:<9s} "
            f"sharpe={m['oos_sharpe']:>6.2f} dsr={m['deflated_sharpe']:.3f} "
            f"pbo={m['pbo']:.3f}"
        )
        if entry["failures"]:
            print(f"    {'; '.join(entry['failures'])}")

    summary = ledger.rejection_summary()
    if summary:
        print("\nblocked by:")
        for name, count in summary.items():
            print(f"  {name:<22s} {count}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Rank a universe and report what is worth trading, and what is not.

    Rejections are printed alongside selections on purpose: knowing that
    everything qualified but was the same trade is a different situation from
    nothing qualifying, and the counts distinguish them.
    """
    import ccxt

    from godalgo.data.feed import OHLCVFeed, bars_per_day
    from godalgo.data.scanner import MarketScanner, ScanCriteria
    from godalgo.execution.broker import FeeSchedule
    from godalgo.strategies.momentum import MomentumStrategy

    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    print(f"loading markets from {args.exchange} ...", file=sys.stderr)
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers() if exchange.has.get("fetchTickers") else {}
    except ccxt.BaseError as exc:
        raise DataUnavailable(f"could not reach {args.exchange}: {exc}") from exc

    symbols = [
        s for s, m in markets.items()
        if m.get("active") and m.get("quote") == args.quote and m.get("spot", True)
    ]
    # Rank by turnover before fetching bars: pulling history for every listing
    # would be thousands of calls to score names that cannot pass the liquidity
    # filter anyway.
    symbols.sort(key=lambda s: -(tickers.get(s, {}).get("quoteVolume") or 0))
    symbols = symbols[: args.top]
    print(f"fetching {len(symbols)} histories ...", file=sys.stderr)

    feed = OHLCVFeed(exchange_id=args.exchange, cache_dir=DEFAULT_CACHE)
    history = {}
    for symbol in symbols:
        try:
            bars = feed.fetch(symbol, args.timeframe, limit=args.limit)
            if len(bars) >= 400:
                history[symbol] = bars
        except (ccxt.BaseError, ValueError) as exc:
            print(f"  skipped {symbol}: {exc}", file=sys.stderr)

    scanner = MarketScanner(
        criteria=ScanCriteria(quote_currency=args.quote, max_candidates=args.max_candidates),
        fees=FeeSchedule(maker=args.maker_fee, taker=args.taker_fee),
        bars_per_day=bars_per_day(args.timeframe),
        holding_bars=MomentumStrategy().expected_holding_bars,
    )
    result = scanner.scan(history, tickers)
    print(result.summary())
    return 0 if result.selected else 1


def cmd_feasibility(args: argparse.Namespace) -> int:
    """Can this configuration clear its costs at this frequency?

    Answered from a real backtest rather than assumptions: realised volatility,
    the conviction the strategies actually produce, and their blended holding
    period. Worth running before anything else -- a bot that will not trade is
    usually being told, correctly, that its frequency is not viable.
    """
    from godalgo.backtest.engine import BacktestConfig, run_backtest
    from godalgo.data.feed import bars_per_day, timeframe_to_minutes
    from godalgo.execution.broker import FeeSchedule
    from godalgo.feasibility import assess_from_backtest

    bars = _load_bars(args)
    bar_seconds = timeframe_to_minutes(args.timeframe) * 60

    momentum, reversion = MomentumStrategy(), MeanReversionStrategy()
    config = BacktestConfig(bars_per_day=bars_per_day(args.timeframe))
    result = run_backtest(bars, momentum, reversion, config, args.symbol)

    report = assess_from_backtest(
        result, bar_seconds, momentum, reversion,
        fees=FeeSchedule(maker=args.maker_fee, taker=args.taker_fee),
        spread_bps=args.spread_bps,
        min_edge_multiple=config.min_edge_multiple,
    )
    print(report.describe())
    print()
    print(f"  momentum holds  {momentum.expected_holding_bars:.0f} bars")
    print(f"  reversion holds {reversion.expected_holding_bars:.0f} bars")
    return 0 if report.tradeable_as_maker else 1


def cmd_preflight(args: argparse.Namespace) -> int:
    """Validate everything an order depends on, without sending one."""
    from godalgo.execution.live import ArmingError, LiveBroker

    async def run() -> int:
        try:
            broker = LiveBroker(arm=True)
        except ArmingError as exc:
            print(f"not armed: {exc}", file=sys.stderr)
            return 2
        try:
            report = await broker.preflight(args.symbol)
        finally:
            await broker.close()

        print(f"exchange        {report['exchange']}")
        print(f"symbol          {report['symbol']}")
        market = report["market"]
        print(f"amount precision {market['amount_precision']}  "
              f"min amount {market['min_amount']}")
        print(f"price precision  {market['price_precision']}  "
              f"min notional {market['min_notional']}")
        print(f"fees            maker={market['maker_fee']} taker={market['taker_fee']}")
        if "round_trip_bps" in report:
            rt = report["round_trip_bps"]
            print(f"round trip      maker {rt['maker']:.1f}bps / taker {rt['taker']:.1f}bps")
        print(f"equity          {report.get('equity')}")
        print(f"position        {report.get('existing_position')}")
        print(f"open orders     {report.get('open_orders')}")
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")
        return 0 if report["ok"] else 1

    return asyncio.run(run())


def cmd_live(args: argparse.Namespace) -> int:
    """Run the autonomous loop.

    Defaults to dry run. Paper and live are opt-in, and live additionally
    requires the arming environment variable that ``LiveBroker`` checks.
    """
    from godalgo.execution.broker import DryRunBroker, PaperBroker
    from godalgo.execution.driver import DriverConfig, WebSocketDriver
    from godalgo.execution.engine import LiveEngine, LiveEngineConfig
    from godalgo.execution.types import TradingMode
    from godalgo.features.session import SessionConfig

    mode = TradingMode(args.mode)
    if mode is TradingMode.LIVE:
        from godalgo.execution.live import ArmingError, LiveBroker

        try:
            broker = LiveBroker(arm=True)
        except ArmingError as exc:
            print(f"refusing to start live: {exc}", file=sys.stderr)
            return 2
    elif mode is TradingMode.PAPER:
        broker = PaperBroker(starting_equity=args.equity)
    else:
        broker = DryRunBroker(starting_equity=args.equity)

    bars = _load_bars(args)
    config = LiveEngineConfig(
        symbol=args.symbol,
        bar_seconds=args.bar_seconds,
        mode=mode,
        session=SessionConfig(tilt_weight=args.session_tilt) if args.session_tilt > 0 else None,
    )
    engine = LiveEngine(broker, MomentumStrategy(), MeanReversionStrategy(), config)
    engine.seed_history(bars)

    driver = WebSocketDriver(
        engine,
        DriverConfig(
            exchange_id=args.exchange,
            max_reconnect_attempts=args.max_reconnects,
        ),
    )

    print(
        f"engine ready: {args.symbol} {args.bar_seconds}s bars, mode={mode.value}, "
        f"{engine.bars.n_complete} bars seeded (warm-up {engine._warmup})",
        file=sys.stderr,
    )
    if engine.bars.n_complete <= engine._warmup:
        print(
            f"warning: seeded {engine.bars.n_complete} bars but warm-up needs "
            f"{engine._warmup}; the bot will not signal until it has caught up. "
            f"Raise --limit.",
            file=sys.stderr,
        )

    try:
        asyncio.run(_run_driver(driver))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    print(f"stats:  {driver.stats.snapshot()}")
    print(f"engine: {engine.state.snapshot()}")
    return 0


async def _run_driver(driver) -> None:
    """Run the driver, converting Ctrl-C into a graceful stop.

    A hard kill would leave the position open; ``stop`` routes through the
    engine's halt, which cancels resting orders and flattens first.
    """
    task = asyncio.create_task(driver.run())
    try:
        await task
    except asyncio.CancelledError:
        await driver.stop()
        raise


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the local terminal.

    Binds to loopback only. The process holds exchange credentials and has no
    authentication, so it must never be reachable off this machine.
    """
    import threading

    from godalgo.ui.server import UIBridge, run_server
    from godalgo.ui.simulator import Simulator

    bridge = UIBridge(starting_equity=args.equity, equity=args.equity, mode="dry_run")
    bridge.symbol = args.symbol

    if args.demo:
        simulator = Simulator(bridge)
        simulator.seed_history()

        def pump() -> None:
            asyncio.run(simulator.run())

        threading.Thread(target=pump, daemon=True).start()
        print("demo mode: positions are fabricated, not traded", file=sys.stderr)

    print(f"terminal: http://{args.host}:{args.port}", file=sys.stderr)
    try:
        run_server(bridge, host=args.host, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="godalgo", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p):
        p.add_argument("--symbol", default="BTC/USDT")
        p.add_argument("--timeframe", default="1h")
        p.add_argument("--exchange", default="binance")
        p.add_argument("--limit", type=int, default=8000)

    p_bt = sub.add_parser("backtest", help="run one backtest on default parameters")
    add_data_args(p_bt)
    p_bt.set_defaults(func=cmd_backtest)

    p_ev = sub.add_parser("evolve", help="run one gated search round")
    add_data_args(p_ev)
    p_ev.add_argument("--candidates", type=int, default=16)
    p_ev.add_argument("--train-size", type=int, default=2500)
    p_ev.add_argument("--test-size", type=int, default=700)
    p_ev.add_argument("--seed", type=int, default=None)
    p_ev.add_argument(
        "--incumbent",
        type=float,
        default=None,
        help="OOS Sharpe of the live configuration, to enforce the improvement bar",
    )
    p_ev.add_argument("--note", default=None)
    p_ev.set_defaults(func=cmd_evolve)

    p_live = sub.add_parser("live", help="run the autonomous execution loop")
    add_data_args(p_live)
    p_live.add_argument("--bar-seconds", type=int, default=60)
    p_live.add_argument(
        "--mode", choices=["dry_run", "paper", "live"], default="dry_run",
        help="dry_run computes orders and sends nothing (default)",
    )
    p_live.add_argument("--equity", type=float, default=10_000.0)
    p_live.add_argument(
        "--session-tilt", type=float, default=0.0,
        help="session/overnight drift overlay weight in [0,1]; 0 disables",
    )
    p_live.add_argument(
        "--max-reconnects", type=int, default=0,
        help="0 retries forever; a positive value halts once exhausted",
    )
    p_live.set_defaults(func=cmd_live)

    p_feas = sub.add_parser(
        "feasibility", help="can this configuration clear its costs at this frequency?"
    )
    add_data_args(p_feas)
    p_feas.add_argument("--maker-fee", type=float, default=0.0002)
    p_feas.add_argument("--taker-fee", type=float, default=0.0006)
    p_feas.add_argument("--spread-bps", type=float, default=2.0)
    p_feas.set_defaults(func=cmd_feasibility)

    p_scan = sub.add_parser("scan", help="rank a universe and report what is tradeable")
    p_scan.add_argument("--exchange", default="binance")
    p_scan.add_argument("--quote", default="USDT")
    p_scan.add_argument("--timeframe", default="1h")
    p_scan.add_argument("--top", type=int, default=25,
                        help="most liquid N symbols to fetch history for")
    p_scan.add_argument("--limit", type=int, default=1500)
    p_scan.add_argument("--max-candidates", type=int, default=6)
    p_scan.add_argument("--maker-fee", type=float, default=0.0002)
    p_scan.add_argument("--taker-fee", type=float, default=0.0006)
    p_scan.set_defaults(func=cmd_scan)

    p_pre = sub.add_parser(
        "preflight", help="validate venue, credentials, and symbol without trading"
    )
    p_pre.add_argument("--symbol", default="BTC/USDT")
    p_pre.set_defaults(func=cmd_preflight)

    p_ui = sub.add_parser("ui", help="launch the local terminal UI")
    p_ui.add_argument("--symbol", default="BTC/USDT")
    p_ui.add_argument("--host", default="127.0.0.1", help="loopback addresses only")
    p_ui.add_argument("--port", type=int, default=8787)
    p_ui.add_argument("--equity", type=float, default=10_000.0)
    p_ui.add_argument("--demo", action="store_true",
                      help="populate with fabricated positions (nothing is traded)")
    p_ui.add_argument("--no-browser", action="store_true")
    p_ui.set_defaults(func=cmd_ui)

    p_led = sub.add_parser("ledger", help="show promotion history")
    p_led.add_argument("--tail", type=int, default=20)
    p_led.set_defaults(func=cmd_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except DataUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
