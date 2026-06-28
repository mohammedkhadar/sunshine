from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from sunshine.config import AppConfig
from sunshine.fetcher import CnnArchiveFetcher
from sunshine.models import TruthPost
from sunshine.scorer import ImpactScorer

logger = logging.getLogger(__name__)

BACKTEST_TICKERS = [
    "NUE", "CLF",
    "GLD", "XLE", "XOM", "CVX", "COP", "OIH", "XOP",
    "LMT", "RTX", "NOC",
]


@dataclass
class FastTrade:
    post_id: str
    post_text: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    notional: float
    pnl: float
    pnl_pct: float
    score: int
    sector: str
    direction: str
    reasoning: str
    exit_reason: str

    @property
    def is_winner(self) -> bool:
        if self.side == "buy":
            return self.exit_price > self.entry_price
        return self.exit_price < self.entry_price


@dataclass
class FastBacktestResult:
    total_return: float = 0.0
    total_pnl: float = 0.0
    total_capital: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: list[FastTrade] = field(default_factory=list)
    sector_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbol_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    score_buckets: dict[str, int] = field(default_factory=dict)

    def compute(self, initial_equity: float = 10000.0) -> None:
        if not self.trades:
            return
        self.total_trades = len(self.trades)
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        self.wins = len(winners)
        self.losses = len(losers)
        self.win_rate = self.wins / self.total_trades if self.total_trades else 0.0
        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_capital = sum(t.notional for t in self.trades)
        self.total_return = self.total_pnl / max(self.total_capital, 1.0)
        self.avg_win = float(np.mean([t.pnl for t in winners])) if winners else 0.0
        self.avg_loss = float(np.mean([t.pnl for t in losers])) if losers else 0.0
        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1.0
        self.profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

        returns = pd.Series([t.pnl / t.notional for t in self.trades])
        if len(returns) > 1 and returns.std() > 0:
            self.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

        equity = initial_equity + np.cumsum([0.0] + [t.pnl for t in self.trades])
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        self.max_drawdown = float(abs(min(dd)))

        for t in self.trades:
            self.sector_stats.setdefault(t.sector, {"trades": 0, "wins": 0, "pnl": 0.0})
            self.sector_stats[t.sector]["trades"] += 1
            self.sector_stats[t.sector]["wins"] += 1 if t.is_winner else 0
            self.sector_stats[t.sector]["pnl"] += t.pnl

        for t in self.trades:
            self.symbol_stats.setdefault(t.symbol, {"trades": 0, "wins": 0, "pnl": 0.0})
            self.symbol_stats[t.symbol]["trades"] += 1
            self.symbol_stats[t.symbol]["wins"] += 1 if t.is_winner else 0
            self.symbol_stats[t.symbol]["pnl"] += t.pnl

        for t in self.trades:
            bucket = f"{t.score}/10"
            self.score_buckets[bucket] = self.score_buckets.get(bucket, 0) + 1


class FastBacktester:
    """Backtester for the event-driven strategy:
    - Topic-filtered posts → LLM impact score
    - Trade only when score >= threshold (default 7/10)
    - Entry at next trading day's open
    - Tight stop-loss (0.5-1%)
    - Same-day exit (proxy for 30-90 min hold)
    """

    def __init__(
        self,
        config: AppConfig,
        score_threshold: float = 7.0,
        sl_pct: float = 0.0075,
        position_usd: float = 1000.0,
        max_daily: int = 8,
    ) -> None:
        self.config = config
        from pathlib import Path
        cache_path = str(Path.home() / ".sunshine" / "score_cache.json")
        self.scorer = ImpactScorer(config.llm, threshold=score_threshold, cache_path=cache_path)
        self.sl_pct = sl_pct
        self.position_usd = position_usd
        self.max_daily = max_daily
        self._price_data: dict[str, pd.DataFrame] = {}

    def download_prices(self, start: str, end: str) -> None:
        import time as _time
        tickers = [t for t in BACKTEST_TICKERS if t not in self._price_data]
        if not tickers:
            return
        for ticker in tickers:
            for attempt in range(2):
                try:
                    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                    if df.empty:
                        break
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    self._price_data[ticker] = df
                    break
                except Exception as exc:
                    logger.warning("yfinance %s attempt %d/2 failed: %s", ticker, attempt + 1, exc)
                    if attempt == 0:
                        _time.sleep(2)

    def _ts_to_dt(self, ts) -> datetime:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return datetime.combine(ts, datetime.min.time())

    def _normalize_dt(self, dt: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        if ts.tz is not None:
            ts = ts.tz_convert("America/New_York")
        return ts

    def _get_entry_day(self, dt: datetime) -> datetime | None:
        target = self._normalize_dt(dt).normalize()
        for sym in BACKTEST_TICKERS:
            df = self._price_data.get(sym)
            if df is not None and not df.empty:
                df_index = pd.DatetimeIndex(df.index)
                if df_index.tz is None:
                    target_local = target.tz_localize(None)
                else:
                    target_local = target.tz_localize("America/New_York")
                idx = df_index.searchsorted(target_local, side="right")
                if idx < len(df):
                    entry_dt = self._ts_to_dt(df.index[idx])
                    entry_dt = entry_dt.replace(hour=9, minute=30, second=0, tzinfo=timezone.utc)
                    return entry_dt
                return None
        return None

    def _simulate_trade(
        self, symbol: str, side: str, entry_price: float, dt: datetime,
    ) -> tuple[float, datetime, str]:
        df = self._price_data.get(symbol)
        if df is None or df.empty:
            return entry_price, dt, "close"

        target = self._normalize_dt(dt).normalize()
        df_index = pd.DatetimeIndex(df.index)
        if df_index.tz is None:
            target = target.tz_localize(None)
        idx = df_index.searchsorted(target, side="right")
        if idx >= len(df):
            return entry_price, dt, "close"

        row = df.iloc[idx]
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        exit_dt = self._ts_to_dt(df.index[idx])
        exit_dt = exit_dt.replace(hour=16, minute=0, second=0, tzinfo=timezone.utc)

        if side == "buy":
            stop = entry_price * (1 - self.sl_pct)
            if low <= stop:
                return stop, exit_dt, "stop"
            return close, exit_dt, "close"
        else:
            stop = entry_price * (1 + self.sl_pct)
            if high >= stop:
                return stop, exit_dt, "stop"
            return close, exit_dt, "close"

    def _position_for_score(self, score: int) -> float:
        if score >= 9:
            return 3000.0
        if score >= 7:
            return 2000.0
        if score >= 5:
            return 1000.0
        return 500.0

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        max_posts: int | None = None,
    ) -> FastBacktestResult:
        fetcher = CnnArchiveFetcher(self.config.fetcher)
        all_posts = fetcher.fetch_latest(limit=99999)
        all_posts.sort(key=lambda p: int(p.id))

        posts: list[TruthPost] = all_posts
        if start_date:
            sd = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            posts = [p for p in posts if p.created_at >= sd]
        if end_date:
            ed = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            posts = [p for p in posts if p.created_at <= ed]
        if max_posts:
            posts = posts[:max_posts]

        if not posts:
            return FastBacktestResult()

        data_start = posts[0].created_at.strftime("%Y-%m-%d")
        data_end = (posts[-1].created_at + timedelta(days=5)).strftime("%Y-%m-%d")
        self.download_prices(data_start, data_end)

        scored = self.scorer.score_many(posts)
        logger.info("Scored %d/%d posts above threshold", len(scored), len(posts))

        result = FastBacktestResult()
        daily_trade_count = 0
        last_trade_day = None

        for s in scored:
            trade_day = s["created_at"].date()
            if last_trade_day != trade_day:
                daily_trade_count = 0
                last_trade_day = trade_day

            if daily_trade_count >= self.max_daily:
                continue

            entry_dt = self._get_entry_day(s["created_at"])
            if entry_dt is None:
                continue

            position_usd = self._position_for_score(s["score"])

            for symbol in s["symbols"]:
                if daily_trade_count >= self.max_daily:
                    break

                entry_price = self._get_entry_price(symbol, entry_dt)
                if entry_price is None or entry_price <= 0:
                    continue

                side = "buy" if s["direction"] == "bullish" else "sell"

                exit_price, exit_time, exit_reason = self._simulate_trade(
                    symbol, side, entry_price, entry_dt,
                )

                if side == "buy":
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = position_usd * pnl_pct

                trade = FastTrade(
                    post_id=s["post_id"],
                    post_text=s["post_text"],
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_dt,
                    exit_time=exit_time,
                    notional=position_usd,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    score=s["score"],
                    sector=s["sector"],
                    direction=s["direction"],
                    reasoning=s["reasoning"],
                    exit_reason=exit_reason,
                )
                result.trades.append(trade)
                daily_trade_count += 1

        result.compute()
        return result

    def _get_entry_price(self, symbol: str, dt: datetime) -> float | None:
        df = self._price_data.get(symbol)
        if df is None or df.empty:
            return None
        target = self._normalize_dt(dt).normalize()
        df_index = pd.DatetimeIndex(df.index)
        if df_index.tz is None:
            target = target.tz_localize(None)
        idx = df_index.searchsorted(target, side="right")
        if idx >= len(df):
            return None
        row = df.iloc[idx]
        return float(row["Open"])


def print_fast_result(result: FastBacktestResult) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    summary = Table(title="Fast Backtest Summary", show_header=False)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total Trades", str(result.total_trades))
    summary.add_row("Wins", str(result.wins))
    summary.add_row("Losses", str(result.losses))
    summary.add_row("Win Rate", f"{result.win_rate:.1%}")
    summary.add_row("Total Return", f"{result.total_return:+.2%}")
    summary.add_row("Total P&L", f"${result.total_pnl:+.2f}")
    summary.add_row("Avg Win", f"${result.avg_win:+.2f}")
    summary.add_row("Avg Loss", f"${result.avg_loss:+.2f}")
    summary.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    summary.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    summary.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
    console.print(summary)

    if result.sector_stats:
        table = Table(title="Performance by Sector")
        table.add_column("Sector")
        table.add_column("Trades")
        table.add_column("Wins")
        table.add_column("Win Rate")
        table.add_column("P&L")
        for sec, stats in sorted(result.sector_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] else 0
            table.add_row(sec, str(stats["trades"]), str(stats["wins"]), f"{wr:.0%}", f"${stats['pnl']:+.2f}")
        console.print(table)

    if result.symbol_stats:
        table = Table(title="Performance by Symbol")
        table.add_column("Symbol")
        table.add_column("Trades")
        table.add_column("Wins")
        table.add_column("Win Rate")
        table.add_column("P&L")
        for sym, stats in sorted(result.symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] else 0
            table.add_row(sym, str(stats["trades"]), str(stats["wins"]), f"{wr:.0%}", f"${stats['pnl']:+.2f}")
        console.print(table)

    if result.score_buckets:
        table = Table(title="Trades by Score Bucket")
        table.add_column("Score")
        table.add_column("Trades")
        for bucket in sorted(result.score_buckets):
            table.add_row(bucket, str(result.score_buckets[bucket]))
        console.print(table)
