from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import json as json_module
from pathlib import Path

from sunshine.analyzer import SignalAnalyzer, HybridAnalyzer, create_analyzer
from sunshine.config import AppConfig
from sunshine.fetcher import CnnArchiveFetcher
from sunshine.models import Side, TruthPost

logger = logging.getLogger(__name__)

BACKTEST_TICKERS = [
    "NUE", "CLF", "STLD", "BABA", "JD",
    "GLD", "COIN", "MSTR", "IBIT", "LMT", "RTX", "NOC",
    "XLE", "XOM", "CVX", "CXW", "GEO", "UNH", "LLY",
    "META", "GOOGL", "MSFT", "CAT", "PWR", "DE",
    "SH",
]


@dataclass
class SimulatedTrade:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    notional: float
    pnl: float
    pnl_pct: float
    reason: str
    category: str
    exit_reason: str

    @property
    def is_winner(self) -> bool:
        if self.side == Side.BUY:
            return self.exit_price > self.entry_price
        return self.exit_price < self.entry_price


@dataclass
class BacktestResult:
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
    trades: list[SimulatedTrade] = field(default_factory=list)
    category_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbol_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            self.category_stats.setdefault(t.category, {"trades": 0, "wins": 0, "pnl": 0.0})
            self.category_stats[t.category]["trades"] += 1
            self.category_stats[t.category]["wins"] += 1 if t.is_winner else 0
            self.category_stats[t.category]["pnl"] += t.pnl

        for t in self.trades:
            self.symbol_stats.setdefault(t.symbol, {"trades": 0, "wins": 0, "pnl": 0.0})
            self.symbol_stats[t.symbol]["trades"] += 1
            self.symbol_stats[t.symbol]["wins"] += 1 if t.is_winner else 0
            self.symbol_stats[t.symbol]["pnl"] += t.pnl


class Backtester:
    def __init__(
        self,
        config: AppConfig,
        use_llm: bool = False,
        use_hybrid: bool = False,
        use_technical: bool = False,
        use_regime: bool = False,
        cache_path: str | None = None,
    ) -> None:
        self.config = config
        self._enable_technical = use_technical
        if use_hybrid:
            self.analyzer = HybridAnalyzer(config.playbook, config.trading, config.llm)
            if use_technical:
                self.analyzer.enable_technical_filter(use_regime=use_regime)
        elif use_llm:
            self.analyzer = create_analyzer(config.playbook, config.trading, config.llm)
        else:
            self.analyzer = SignalAnalyzer(
                config.playbook,
                min_confidence=config.trading.min_confidence,
            )
        self._price_data: dict[str, pd.DataFrame] = {}
        self._cached_posts: list[TruthPost] | None = None
        self._cached_start: str | None = None
        self._cached_end: str | None = None
        self._cached_max: int | None = None
        self.tp_pct = config.trading.take_profit_pct
        self.sl_pct = config.trading.stop_loss_pct
        self._llm_cache: dict[str, dict] = {}
        self._cache_path = cache_path
        if cache_path:
            self._load_cache()

    def _load_cache(self) -> None:
        if not self._cache_path:
            return
        p = Path(self._cache_path)
        if p.exists():
            try:
                with p.open() as f:
                    self._llm_cache = json_module.load(f)
                logger.info("Loaded %d cached LLM responses from %s", len(self._llm_cache), self._cache_path)
            except Exception as exc:
                logger.warning("Failed to load LLM cache: %s", exc)

    def _save_cache(self) -> None:
        if not self._cache_path or not self._llm_cache:
            return
        p = Path(self._cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with p.open("w") as f:
                json_module.dump(self._llm_cache, f)
            logger.info("Saved %d LLM responses to cache", len(self._llm_cache))
        except Exception as exc:
            logger.warning("Failed to save LLM cache: %s", exc)

    def _analyze_with_cache(self, post: TruthPost) -> Signal | None:
        if post.id in self._llm_cache:
            cached = self._llm_cache[post.id]
            if cached is None:
                return None
            from sunshine.analyzer import LLMRefiner, HybridAnalyzer
            if isinstance(self.analyzer, HybridAnalyzer) and isinstance(self.analyzer.llm, LLMRefiner):
                self.analyzer.llm._cache[post.id] = cached
        signal = self.analyzer.analyze(post)
        if isinstance(self.analyzer, SignalAnalyzer):
            return signal
        if signal is not None:
            from sunshine.analyzer import LLMRefiner, HybridAnalyzer
            if isinstance(self.analyzer, HybridAnalyzer) and isinstance(self.analyzer.llm, LLMRefiner):
                cached_refinement = self.analyzer.llm._cache.get(post.id)
                if isinstance(cached_refinement, dict):
                    self._llm_cache[post.id] = cached_refinement
                else:
                    self._llm_cache[post.id] = {"confidence": signal.confidence, "reasoning": signal.llm_summary or ""}
            else:
                self._llm_cache[post.id] = True
        else:
            self._llm_cache[post.id] = None
        return signal

    def download_prices(self, start: str, end: str) -> None:
        tickers = [t for t in set(BACKTEST_TICKERS) if t not in self._price_data]
        if not tickers:
            return
        logger.info("Downloading price data for %d tickers from %s to %s", len(tickers), start, end)
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if df.empty:
                    logger.warning("No price data for %s", ticker)
                    continue
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self._price_data[ticker] = df
            except Exception as exc:
                logger.warning("Failed to download %s: %s", ticker, exc)

    def _ts_to_dt(self, ts) -> datetime:
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return datetime.combine(ts, datetime.min.time())

    def _normalize_dt(self, dt: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        if ts.tz is not None:
            ts = ts.tz_convert("America/New_York")
        return ts

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

    def _simulate_exit(
        self, symbol: str, side: Side, entry_price: float, dt: datetime,
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

        if side == Side.BUY:
            tp_level = entry_price * (1 + self.tp_pct)
            sl_level = entry_price * (1 - self.sl_pct)
            hit_tp = high >= tp_level
            hit_sl = low <= sl_level
            if hit_tp and hit_sl:
                dist_tp = tp_level - entry_price
                dist_sl = entry_price - sl_level
                if dist_tp <= dist_sl:
                    return tp_level, exit_dt, "tp"
                return sl_level, exit_dt, "sl"
            if hit_tp:
                return tp_level, exit_dt, "tp"
            if hit_sl:
                return min(sl_level, low), exit_dt, "sl"
        else:
            tp_level = entry_price * (1 - self.tp_pct)
            sl_level = entry_price * (1 + self.sl_pct)
            hit_tp = low <= tp_level
            hit_sl = high >= sl_level
            if hit_tp and hit_sl:
                dist_tp = entry_price - tp_level
                dist_sl = sl_level - entry_price
                if dist_tp <= dist_sl:
                    return tp_level, exit_dt, "tp"
                return sl_level, exit_dt, "sl"
            if hit_tp:
                return tp_level, exit_dt, "tp"
            if hit_sl:
                return max(sl_level, high), exit_dt, "sl"

        return close, exit_dt, "close"

    def set_params(self, tp_pct: float | None = None, sl_pct: float | None = None) -> None:
        if tp_pct is not None:
            self.tp_pct = tp_pct
        if sl_pct is not None:
            self.sl_pct = sl_pct

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        max_posts: int | None = None,
    ) -> BacktestResult:
        posts: list[TruthPost]
        sd_str = start_date or ""
        ed_str = end_date or ""
        mp_str = str(max_posts or "")
        if (
            self._cached_posts is not None
            and self._cached_start == sd_str
            and self._cached_end == ed_str
            and self._cached_max == mp_str
        ):
            posts = self._cached_posts
            logger.info("Using %d cached posts", len(posts))
        else:
            fetcher = CnnArchiveFetcher(self.config.fetcher)
            all_posts = fetcher.fetch_latest(limit=99999)
            all_posts.sort(key=lambda p: int(p.id))
            logger.info("Loaded %d historical posts from archive", len(all_posts))

            if start_date:
                sd = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                all_posts = [p for p in all_posts if p.created_at >= sd]
            if end_date:
                ed = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                all_posts = [p for p in all_posts if p.created_at <= ed]
            if max_posts:
                all_posts = all_posts[:max_posts]

            posts = all_posts
            self._cached_posts = posts
            self._cached_start = sd_str
            self._cached_end = ed_str
            self._cached_max = mp_str
        if not posts:
            logger.warning("No posts in the specified range")
            return BacktestResult()

        data_start = posts[0].created_at.strftime("%Y-%m-%d")
        data_end = (posts[-1].created_at + timedelta(days=5)).strftime("%Y-%m-%d")
        self.download_prices(data_start, data_end)

        result = BacktestResult()
        if hasattr(self.analyzer, "_category_hits"):
            self.analyzer._category_hits = {}

        symbol_cooldowns: dict[str, datetime] = {}
        daily_trade_count = 0
        last_trade_day = None
        max_daily = self.config.trading.max_daily_trades
        cooldown_secs = 3600

        for idx, post in enumerate(posts):
            if self._cache_path and idx > 0 and idx % 100 == 0:
                self._save_cache()
            signal = self._analyze_with_cache(post)
            if signal is None:
                continue

            trade_day = post.created_at.date()
            if last_trade_day != trade_day:
                daily_trade_count = 0
                last_trade_day = trade_day

            if daily_trade_count >= max_daily:
                continue

            scale = (
                (signal.confidence - self.config.trading.min_confidence)
                / (1.0 - self.config.trading.min_confidence)
            )
            per_trade = (self.config.trading.max_position_usd / max(len(signal.actions), 1)) * scale

            if per_trade < 1:
                continue

            for action in signal.actions:
                if daily_trade_count >= max_daily:
                    break

                if action.symbol in symbol_cooldowns:
                    if post.created_at < symbol_cooldowns[action.symbol]:
                        continue

                entry_price = self._get_entry_price(action.symbol, post.created_at)
                if entry_price is None or entry_price <= 0:
                    continue

                exit_price, exit_time, exit_reason = self._simulate_exit(
                    action.symbol, action.side, entry_price, post.created_at,
                )

                if action.side == Side.BUY:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = per_trade * pnl_pct

                trade = SimulatedTrade(
                    symbol=action.symbol,
                    side=action.side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=post.created_at,
                    exit_time=exit_time,
                    notional=per_trade,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=action.reason,
                    category=signal.category,
                    exit_reason=exit_reason,
                )
                result.trades.append(trade)
                symbol_cooldowns[action.symbol] = post.created_at + timedelta(seconds=cooldown_secs)
                daily_trade_count += 1

        result.compute()
        return result


def run_backtest(
    config: AppConfig | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_posts: int | None = None,
    use_llm: bool = False,
    use_hybrid: bool = False,
    use_technical: bool = False,
    use_regime: bool = False,
    cache_path: str | None = None,
) -> BacktestResult:
    if config is None:
        from sunshine.config import load_config
        config = load_config()
    if start_date is None:
        start_date = "2025-01-01"
    needs_cache = use_llm or use_hybrid
    if needs_cache and cache_path is None:
        cache_path = str(Path.home() / ".sunshine" / "llm_cache.json")
    bt = Backtester(
        config,
        use_llm=use_llm,
        use_hybrid=use_hybrid,
        use_technical=use_technical,
        use_regime=use_regime,
        cache_path=cache_path,
    )
    result = bt.run(start_date=start_date, end_date=end_date, max_posts=max_posts)
    if cache_path:
        bt._save_cache()
    return result


def optimize_parameters(
    config: AppConfig | None = None,
    start_date: str | None = "2025-03-01",
    end_date: str | None = "2025-06-01",
    max_posts: int | None = 500,
    mode: str = "rule",
) -> list[dict]:
    """Sweep TP/SL/confidence to find optimal parameters.
    Uses a single Backtester instance with pre-downloaded prices.
    """
    if config is None:
        from sunshine.config import load_config
        config = load_config()

    tp_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    sl_values = [0.02, 0.03, 0.05, 0.07, 0.10]
    conf_values = [0.40, 0.50, 0.55, 0.60]

    use_llm = mode == "llm"
    use_hybrid = mode == "hybrid"

    bt = Backtester(
        config,
        use_llm=use_llm,
        use_hybrid=use_hybrid,
        cache_path=None,
    )
    bt.run(start_date=start_date, end_date=end_date, max_posts=max_posts)

    results: list[dict] = []
    total = len(tp_values) * len(sl_values) * len(conf_values)
    count = 0

    for tp in tp_values:
        for sl in sl_values:
            for conf in conf_values:
                bt.set_params(tp_pct=tp, sl_pct=sl)
                cfg = config
                cfg.trading.take_profit_pct = tp
                cfg.trading.stop_loss_pct = sl
                cfg.trading.min_confidence = conf

                result = bt.run(
                    start_date=start_date,
                    end_date=end_date,
                    max_posts=max_posts,
                )
                count += 1
                score = result.sharpe_ratio if result.total_trades > 10 else -99
                results.append({
                    "tp": tp,
                    "sl": sl,
                    "min_conf": conf,
                    "trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "total_return": result.total_return,
                    "profit_factor": result.profit_factor,
                    "sharpe": result.sharpe_ratio,
                    "max_dd": result.max_drawdown,
                    "score": score,
                })
                logger.info(
                    "[%d/%d] TP=%.0f%% SL=%.0f%% conf=%.0f%% → %d trades, SR=%.2f, ret=%.2f%%",
                    count, total, tp * 100, sl * 100, conf * 100,
                    result.total_trades, result.sharpe_ratio, result.total_return * 100,
                )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def print_optimize_results(results: list[dict], top_n: int = 10) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Top {min(top_n, len(results))} Parameter Sets (sorted by Sharpe)")
    table.add_column("Rank")
    table.add_column("TP")
    table.add_column("SL")
    table.add_column("Min Conf")
    table.add_column("Trades")
    table.add_column("Win Rate")
    table.add_column("Return")
    table.add_column("Profit F.")
    table.add_column("Sharpe")
    table.add_column("Max DD")

    for i, r in enumerate(results[:top_n]):
        table.add_row(
            str(i + 1),
            f"{r['tp']:.0%}",
            f"{r['sl']:.0%}",
            f"{r['min_conf']:.0%}",
            str(r["trades"]),
            f"{r['win_rate']:.1%}",
            f"{r['total_return']:+.2%}",
            f"{r['profit_factor']:.2f}",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd']:.2%}",
        )
    console.print(table)


def print_backtest_result(result: BacktestResult) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    summary = Table(title="Backtest Summary", show_header=False)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total Trades", str(result.total_trades))
    summary.add_row("Wins", str(result.wins))
    summary.add_row("Losses", str(result.losses))
    summary.add_row("Win Rate", f"{result.win_rate:.1%}")
    summary.add_row("Total Return", f"{result.total_return:+.2%}")
    summary.add_row("Total P&L", f"${result.total_pnl:+.2f}")
    summary.add_row("Total Capital Deployed", f"${result.total_capital:.2f}")
    summary.add_row("Avg Win", f"${result.avg_win:+.2f}")
    summary.add_row("Avg Loss", f"${result.avg_loss:+.2f}")
    summary.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    summary.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    summary.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
    console.print(summary)

    if result.category_stats:
        cat_table = Table(title="Performance by Category")
        cat_table.add_column("Category")
        cat_table.add_column("Trades")
        cat_table.add_column("Wins")
        cat_table.add_column("Win Rate")
        cat_table.add_column("P&L")
        for cat, stats in sorted(result.category_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] else 0
            cat_table.add_row(
                cat,
                str(stats["trades"]),
                str(stats["wins"]),
                f"{wr:.0%}",
                f"${stats['pnl']:+.2f}",
            )
        console.print(cat_table)

    if result.symbol_stats:
        sym_table = Table(title="Performance by Symbol")
        sym_table.add_column("Symbol")
        sym_table.add_column("Trades")
        sym_table.add_column("Wins")
        sym_table.add_column("Win Rate")
        sym_table.add_column("P&L")
        for sym, stats in sorted(result.symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] if stats["trades"] else 0
            sym_table.add_row(
                sym,
                str(stats["trades"]),
                str(stats["wins"]),
                f"{wr:.0%}",
                f"${stats['pnl']:+.2f}",
            )
        console.print(sym_table)
