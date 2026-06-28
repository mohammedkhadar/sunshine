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
    bot.bootstrap()
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
