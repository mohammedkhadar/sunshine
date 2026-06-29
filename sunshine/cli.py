from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from sunshine.bot import SunshineBot
from sunshine.config import load_config
from sunshine.fetcher import create_fetcher
from sunshine.storage import Storage

console = Console()


def cmd_monitor(_: argparse.Namespace) -> int:
    SunshineBot().run()
    return 0


def cmd_poll(_: argparse.Namespace) -> int:
    bot = SunshineBot()
    count = bot.poll_once()
    console.print(f"Processed {count} new post(s)")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    storage = Storage()
    config = load_config()

    table = Table(title="Sunshine Bot Status")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Trading mode", config.trading.mode)
    table.add_row("Fetcher", config.fetcher.primary)
    table.add_row("Last seen post", storage.get_last_seen_post_id() or "(not set)")
    table.add_row("Trades today", str(storage.trades_today_count()))
    console.print(table)

    signals = storage.recent_signals(5)
    if signals:
        sig_table = Table(title="Recent Signals")
        sig_table.add_column("ID")
        sig_table.add_column("Category")
        sig_table.add_column("Confidence")
        sig_table.add_column("Preview")
        for s in signals:
            sig_table.add_row(
                str(s["id"]),
                s["category"] or "",
                f"{s['confidence']:.0%}" if s["confidence"] else "",
                (s.get("post_content") or "")[:60] + "...",
            )
        console.print(sig_table)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    config = load_config()
    fetcher = create_fetcher(config.fetcher)
    bot = SunshineBot(config)

    if args.text:
        from sunshine.models import TruthPost
        from datetime import datetime, timezone

        post = TruthPost(
            id="manual",
            content=args.text,
            created_at=datetime.now(timezone.utc),
            source="cli",
        )
        signal = bot.analyzer.analyze(post)
        if signal:
            console.print_json(json.dumps(signal.__dict__, default=str))
        else:
            console.print("[yellow]No signal generated[/yellow]")
        return 0

    posts = fetcher.fetch_latest(limit=args.limit)
    for post in posts[: args.limit]:
        signal = bot.analyzer.analyze(post)
        if signal or args.all:
            console.print(f"\n[bold]Post {post.id}[/bold] ({post.created_at})")
            console.print(post.content[:200])
            if signal:
                console.print(
                    f"  → {signal.category} ({signal.confidence:.0%}): "
                    + ", ".join(f"{a.side.value} {a.symbol}" for a in signal.actions)
                )
            else:
                console.print("  → no signal")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from sunshine.backtester import run_backtest, print_backtest_result
    from sunshine.config import load_config

    config = load_config()
    result = run_backtest(
        config=config,
        start_date=args.start,
        end_date=args.end,
        max_posts=args.max_posts,
        use_llm=args.llm,
        use_hybrid=args.hybrid,
        use_technical=args.technical,
        use_regime=args.regime,
    )
    print_backtest_result(result)
    return 0


def cmd_fast_backtest(args: argparse.Namespace) -> int:
    from sunshine.fasttrader import FastBacktester, print_fast_result
    from sunshine.config import load_config

    config = load_config()
    bt = FastBacktester(
        config,
        score_threshold=args.threshold,
        sl_pct=args.sl,
        stop_atr_multiple=args.atr_multiple,
        position_usd=args.position,
        daily_loss_limit=args.daily_loss_limit,
    )
    result = bt.run(
        start_date=args.start,
        end_date=args.end,
        max_posts=args.max_posts,
    )
    print_fast_result(result)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from sunshine.backtester import optimize_parameters, print_optimize_results
    from sunshine.config import load_config

    config = load_config()
    results = optimize_parameters(
        config=config,
        start_date=args.start,
        end_date=args.end,
        max_posts=args.max_posts,
        mode=args.mode,
    )
    print_optimize_results(results, top_n=args.top)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sunshine",
        description="Trading bot driven by Trump's Truth Social posts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("monitor", help="Run continuous polling loop").set_defaults(func=cmd_monitor)
    sub.add_parser("poll", help="Poll once for new posts").set_defaults(func=cmd_poll)
    sub.add_parser("status", help="Show bot status and recent signals").set_defaults(func=cmd_status)

    analyze = sub.add_parser("analyze", help="Analyze posts for signals")
    analyze.add_argument("--limit", type=int, default=10, help="Posts to scan")
    analyze.add_argument("--all", action="store_true", help="Show posts without signals too")
    analyze.add_argument("--text", type=str, help="Analyze arbitrary text")
    analyze.set_defaults(func=cmd_analyze)

    backtest = sub.add_parser("backtest", help="Run backtest on historical posts")
    backtest.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    backtest.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    backtest.add_argument("--max-posts", type=int, default=None, help="Limit number of posts to process")
    backtest.add_argument("--llm", action="store_true", help="Use LLM analyzer (requires OPENROUTER_API_KEY)")
    backtest.add_argument("--hybrid", action="store_true", help="Rule-based gatekeeper then LLM refinement")
    backtest.add_argument("--technical", action="store_true", help="Add SMA50/RSI14 technical filter")
    backtest.add_argument("--regime", action="store_true", help="Add SPY 200-day MA market regime filter")
    backtest.set_defaults(func=cmd_backtest)

    fast_backtest = sub.add_parser("fast-backtest", help="Event-driven backtest: LLM impact scoring, tight stops, same-day exit")
    fast_backtest.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    fast_backtest.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    fast_backtest.add_argument("--max-posts", type=int, default=None, help="Limit posts to process")
    fast_backtest.add_argument("--threshold", type=float, default=7.0, help="Minimum impact score (1-10)")
    fast_backtest.add_argument("--sl", type=float, default=0.03, help="Stop-loss fraction fallback (e.g. 0.03 = 3%%)")
    fast_backtest.add_argument("--atr-multiple", type=float, default=1.5, help="ATR(14) multiple for stop distance")
    fast_backtest.add_argument("--position", type=float, default=1000.0, help="Notional per trade ($)")
    fast_backtest.add_argument("--daily-loss-limit", type=float, default=150.0, help="Max daily loss before stopping ($)")
    fast_backtest.set_defaults(func=cmd_fast_backtest)

    optimize = sub.add_parser("optimize", help="Sweep TP/SL/confidence to find optimal params")
    optimize.add_argument("--start", type=str, default="2025-03-01", help="Start date (YYYY-MM-DD)")
    optimize.add_argument("--end", type=str, default="2025-06-01", help="End date (YYYY-MM-DD)")
    optimize.add_argument("--max-posts", type=int, default=500, help="Posts to process per run")
    optimize.add_argument("--mode", type=str, default="rule", choices=["rule", "hybrid", "llm"],
                          help="Analyzer mode for optimization")
    optimize.add_argument("--top", type=int, default=10, help="Show top N results")
    optimize.set_defaults(func=cmd_optimize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
