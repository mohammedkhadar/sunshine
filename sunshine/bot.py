from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sunshine.analyzer import create_analyzer, SignalAnalyzer
from sunshine.config import AppConfig, load_config
from sunshine.fetcher import create_fetcher
from sunshine.storage import Storage
from sunshine.trader import create_trader

logger = logging.getLogger(__name__)
console = Console()


class SunshineBot:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.storage = Storage()
        self.fetcher = create_fetcher(self.config.fetcher)
        self.analyzer = create_analyzer(
            self.config.playbook,
            self.config.trading,
            llm_config=self.config.llm,
        )
        self.trader = create_trader(self.config.trading, self.storage)

    def process_post(self, post) -> bool:
        is_new = self.storage.save_post(post)
        if not is_new:
            return False

        signal = self.analyzer.analyze(post)
        if signal is None:
            console.print(f"[dim]Post {post.id}: no actionable signal[/dim]")
            return True

        self._print_signal(post, signal)
        self.storage.save_signal(signal)
        self.trader.execute(signal)
        return True

    def _print_signal(self, post, signal) -> None:
        table = Table(title="Trading Signal", show_header=True)
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Post ID", post.id)
        table.add_row("Category", signal.category)
        table.add_row("Sentiment", signal.sentiment)
        table.add_row("Confidence", f"{signal.confidence:.0%}")
        kw = ", ".join(signal.matched_keywords) if signal.matched_keywords else "(LLM)"
        table.add_row("Keywords", kw)
        if signal.llm_summary:
            table.add_row("LLM", signal.llm_summary)
        for action in signal.actions:
            table.add_row(
                "Action",
                f"{action.side.value.upper()} {action.symbol} — {action.reason}",
            )
        console.print(
            Panel(
                post.content[:400] + ("..." if len(post.content) > 400 else ""),
                title=f"New Truth Social post ({post.created_at.strftime('%Y-%m-%d %H:%M UTC')})",
                subtitle=post.url or post.source,
            )
        )
        console.print(table)

    def poll_once(self, since_time: datetime | None = None) -> int:
        last_id = self.storage.get_last_seen_post_id()

        kwargs: dict[str, Any] = {"limit": self.config.fetcher.poll_limit}
        if last_id:
            kwargs["since_id"] = last_id
        elif since_time:
            kwargs["since_time"] = since_time
        else:
            kwargs["since_time"] = datetime.now(timezone.utc) - timedelta(minutes=15)

        posts = self.fetcher.fetch_latest(**kwargs)

        if not posts:
            return 0

        posts.sort(key=lambda p: int(p.id))
        max_id = posts[-1].id
        for post in posts:
            self.process_post(post)
        self.storage.set_last_seen_post_id(max_id)
        return len(posts)

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        console.print(
            Panel(
                f"Mode: [bold]{self.config.trading.mode}[/bold]\n"
                f"Fetcher: {self.config.fetcher.primary}\n"
                f"Poll every {self.config.fetcher.poll_interval}s\n"
                f"Min confidence: {self.config.trading.min_confidence:.0%}",
                title="Sunshine Bot",
                subtitle="Truth Social → signals → trades",
            )
        )

        while True:
            try:
                if not self.trader.market_open():
                    console.print("[yellow]Market closed — sleeping 15 min[/yellow]")
                    time.sleep(900)
                    continue
                count = self.poll_once()
                if count:
                    console.print(f"[cyan]{datetime.now(timezone.utc):%H:%M:%S}[/cyan] polled — {count} post(s)")
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break
            except Exception as exc:
                logger.exception("Poll error: %s", exc)
            time.sleep(self.config.fetcher.poll_interval)
