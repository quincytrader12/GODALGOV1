"""Command line entry point.

    python -m godalgo backtest --symbol BTC/USDT --timeframe 1h --limit 8000
    python -m godalgo evolve   --symbol BTC/USDT --candidates 16
    python -m godalgo ledger
    python -m godalgo live      --symbol BTC/USDT --bar-seconds 60 --mode paper

Deliberately thin. It wires existing components together and prints results; it
holds no strategy logic of its own, so anything that works here works identically
when the agent runtime drives the same functions directly.
"""

from __future__ import annotations

import argparse
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


def _load_bars(args: argparse.Namespace):
    feed = OHLCVFeed(exchange_id=args.exchange, cache_dir=DEFAULT_CACHE)
    print(f"fetching {args.symbol} {args.timeframe} from {args.exchange} ...", file=sys.stderr)
    bars = feed.fetch(args.symbol, args.timeframe, limit=args.limit)
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


def cmd_live(args: argparse.Namespace) -> int:
    """Run the autonomous loop.

    Defaults to dry run. Paper and live are opt-in, and live additionally
    requires the arming environment variable that ``LiveBroker`` checks.
    """
    import asyncio

    from godalgo.execution.broker import DryRunBroker, PaperBroker
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

    print(
        f"engine ready: {args.symbol} {args.bar_seconds}s bars, mode={mode.value}, "
        f"{engine.bars.n_complete} bars seeded (warm-up {engine._warmup})",
        file=sys.stderr,
    )
    print(
        "no live market-data stream is wired up yet -- feed ticks via "
        "LiveEngine.on_tick()/on_book(). See README.",
        file=sys.stderr,
    )
    asyncio.run(_report(engine))
    return 0


async def _report(engine) -> None:
    print(f"state: {engine.state.snapshot()}")


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
    p_live.set_defaults(func=cmd_live)

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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
