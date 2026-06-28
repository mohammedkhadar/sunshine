from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sunshine.analyzer import SignalAnalyzer
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
        self.analyzer = SignalAnalyzer(
            self.config.playbook,
            min_confidence=self.config.trading.min_confidence,
        )
        self.trader = create_trader(self.config.trading, self.storage)
        self._since_id: str | None = None

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
        table.add_row("Keywords", ", ".join(signal.matched_keywords))
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

    def poll_once(self) -> int:
        posts = self.fetcher.fetch_latest(
            since_id=self._since_id,
            limit=self.config.fetcher.poll_limit,
        )

        if not posts:
            return 0

        posts.sort(key=lambda p: int(p.id))
        processed = 0
        for post in posts:
            self.process_post(post)
            processed += 1

        self._since_id = max(posts, key=lambda p: int(p.id)).id
        return processed

    def bootstrap(self) -> None:
        """Seed _since_id from the latest post so we only process new posts going forward."""
        posts = self.fetcher.fetch_latest(limit=1)
        if posts:
            self._since_id = posts[0].id
            console.print(
                f"[green]Bootstrapped — watching for posts newer than {self._since_id}[/green]"
            )

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
        self.bootstrap()

        while True:
            try:
                count = self.poll_once()
                console.print(f"[cyan]{datetime.now(timezone.utc):%H:%M:%S}[/cyan] polled — {count} post(s)")
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break
            except Exception as exc:
                logger.exception("Poll error: %s", exc)
            time.sleep(self.config.fetcher.poll_interval)
