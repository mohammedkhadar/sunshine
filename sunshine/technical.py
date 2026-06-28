from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from sunshine.models import Side

logger = logging.getLogger(__name__)


class RegimeFilter:
    """Market regime filter using SPY 200-day MA.
    Blocks longs in bearish regimes and shorts in bullish regimes.
    """

    def __init__(self) -> None:
        self._cache: pd.DataFrame | None = None

    def _get_spy_data(self, dt: datetime) -> pd.DataFrame | None:
        if self._cache is None:
            end = (dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            start = (dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            try:
                df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
                if df.empty:
                    return None
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self._cache = df
            except Exception as exc:
                logger.debug("Failed to download SPY: %s", exc)
                return None
        df = self._cache
        cutoff = dt.replace(tzinfo=None) if dt.tzinfo else dt
        filtered = df[df.index <= cutoff]
        return filtered if not filtered.empty else None

    def check(self, side: Side, dt: datetime) -> tuple[bool, str]:
        df = self._get_spy_data(dt)
        if df is None or len(df) < 200:
            return True, "pass (insufficient data)"

        price = float(df["Close"].iloc[-1])
        sma200 = float(df["Close"].rolling(200).mean().iloc[-1])

        if side == Side.BUY and price < sma200:
            return False, f"Bearish regime: SPY ${price:.2f} below 200-MA ${sma200:.2f}"
        if side == Side.SELL and price > sma200:
            return False, f"Bullish regime: SPY ${price:.2f} above 200-MA ${sma200:.2f}"

        regime = "bullish" if price >= sma200 else "bearish"
        return True, f"pass ({regime})"


class TechnicalFilter:
    """Price-based filters using SMA50 and RSI14 to confirm trade signals."""

    def __init__(self, regime_filter: RegimeFilter | None = None) -> None:
        self._cache: dict[str, pd.DataFrame] = {}
        self._regime = regime_filter

    def _get_data(self, symbol: str, up_to: datetime) -> pd.DataFrame | None:
        key = symbol
        if key not in self._cache:
            end = (up_to + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            start = (up_to - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            try:
                df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
                if df.empty:
                    return None
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                self._cache[key] = df
            except Exception as exc:
                logger.debug("Failed to download %s: %s", symbol, exc)
                return None

        df = self._cache[key]
        cutoff = up_to.replace(tzinfo=None) if up_to.tzinfo else up_to
        filtered = df[df.index <= cutoff]
        if filtered.empty:
            return None
        return filtered

    def _sma50(self, symbol: str, dt: datetime) -> float | None:
        df = self._get_data(symbol, dt)
        if df is None or len(df) < 50:
            return None
        return float(df["Close"].rolling(50).mean().iloc[-1])

    def _rsi14(self, symbol: str, dt: datetime) -> float | None:
        df = self._get_data(symbol, dt)
        if df is None or len(df) < 15:
            return None
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _latest_close(self, symbol: str, dt: datetime) -> float | None:
        df = self._get_data(symbol, dt)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])

    def check(self, symbol: str, side: Side, dt: datetime) -> tuple[bool, str]:
        if self._regime:
            ok, reason = self._regime.check(side, dt)
            if not ok:
                return False, reason

        price = self._latest_close(symbol, dt)
        sma = self._sma50(symbol, dt)
        rsi = self._rsi14(symbol, dt)

        if side == Side.BUY:
            if price is not None and sma is not None and price < sma:
                return False, f"Price ${price:.2f} below SMA50 ${sma:.2f}"
            if rsi is not None and rsi > 70:
                return False, f"RSI {rsi:.0f} overbought (>70)"
        else:
            if price is not None and sma is not None and price > sma:
                return False, f"Price ${price:.2f} above SMA50 ${sma:.2f}"
            if rsi is not None and rsi < 30:
                return False, f"RSI {rsi:.0f} oversold (<30)"

        return True, "pass"
