from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod

from sunshine.config import TradingConfig
from sunshine.models import Side, Signal, TradeAction
from sunshine.storage import Storage

logger = logging.getLogger(__name__)

CLOSE_DELAY = 30 * 60
COOLDOWN_SECONDS = 3600


class Trader(ABC):
    @abstractmethod
    def execute(self, signal: Signal) -> list[dict]:
        ...


class DryRunTrader(Trader):
    def __init__(self, config: TradingConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self._daily_count = 0
        self._cooldowns: dict[str, float] = {}

    def execute(self, signal: Signal) -> list[dict]:
        results: list[dict] = []
        if self._daily_count >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached (%s)", self.config.max_daily_trades)
            return results

        scale = (signal.confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)
        per_trade = (self.config.max_position_usd / max(len(signal.actions), 1)) * scale

        if per_trade < 1:
            logger.info("Confidence %.0f%% too low for meaningful position, skipping", signal.confidence * 100)
            return results

        now = time.time()
        for action in signal.actions:
            if action.symbol in self._cooldowns and now < self._cooldowns[action.symbol]:
                logger.info("Symbol %s on cooldown, skipping", action.symbol)
                continue

            action.notional_usd = per_trade
            self._daily_count += 1
            self._cooldowns[action.symbol] = now + COOLDOWN_SECONDS
            results.append(
                {
                    "symbol": action.symbol,
                    "side": action.side.value,
                    "notional_usd": per_trade,
                    "status": "dry_run",
                }
            )
            logger.info(
                "[DRY RUN] %s %s $%.2f — %s",
                action.side.value.upper(),
                action.symbol,
                per_trade,
                action.reason,
            )
            self._schedule_close(action.symbol, action.side, per_trade)
            self._log_trailing_stop(action.symbol, action.side, per_trade)
            self._log_take_profit(action.symbol, action.side, per_trade)
        return results

    def _schedule_close(self, symbol: str, side: Side, notional: float) -> None:
        close_side = Side.BUY if side == Side.SELL else Side.SELL
        threading.Timer(CLOSE_DELAY, self._close_position, args=[symbol, close_side, notional]).start()
        logger.info("Scheduled close: %s %s $%.2f in 30m", close_side.value.upper(), symbol, notional)

    def _close_position(self, symbol: str, side: Side, notional: float) -> None:
        logger.info("[DRY RUN] CLOSE %s %s $%.2f (scheduled)", side.value.upper(), symbol, notional)

    def _log_trailing_stop(self, symbol: str, side: Side, notional: float) -> None:
        trail_pct = self.config.stop_loss_pct * 100
        logger.info(
            "[DRY RUN] Trailing stop set: %s %s trails %.0f%% (entry $0.00)",
            side.value.upper(), symbol, trail_pct,
        )

    def _log_take_profit(self, symbol: str, side: Side, notional: float) -> None:
        tp_pct = self.config.take_profit_pct * 100
        logger.info(
            "[DRY RUN] Take-profit set: %s %s at +%.0f%% entry",
            side.value.upper(), symbol, tp_pct,
        )


class AlpacaTrader(Trader):
    def __init__(self, config: TradingConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._client = None
        self._cooldowns: dict[str, float] = {}

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key or not self.secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required for live/paper trading")

        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
        from alpaca.trading.requests import MarketOrderRequest

        self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._OrderType = OrderType
        self._MarketOrderRequest = MarketOrderRequest
        return self._client

    def _market_open(self) -> bool:
        try:
            clock = self._get_client().get_clock()
            return clock.is_open
        except Exception:
            return True

    def _get_current_price(self, symbol: str) -> float | None:
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            dc = StockHistoricalDataClient(self.api_key, self.secret_key)
            resp = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            return float(resp[symbol].price)
        except Exception:
            return None

    def _build_order(self, symbol: str, side, notional: float):
        if side == self._OrderSide.SELL:
            price = self._get_current_price(symbol)
            if not price:
                raise RuntimeError(f"Could not fetch price for {symbol}")
            qty = int(notional / price)
            if qty < 1:
                raise RuntimeError(f"Notional ${notional:.2f} too low for 1 share of {symbol} (${price:.2f})")
            return self._MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=self._TimeInForce.DAY,
            )
        return self._MarketOrderRequest(
            symbol=symbol, notional=round(notional, 2), side=side, time_in_force=self._TimeInForce.DAY,
        )

    def execute(self, signal: Signal) -> list[dict]:
        results: list[dict] = []

        if not self._market_open():
            logger.warning("Market closed — skipping trade for %s", signal.category)
            return results

        client = self._get_client()

        scale = (signal.confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)
        per_trade = (self.config.max_position_usd / max(len(signal.actions), 1)) * scale

        if per_trade < 1:
            logger.info("Confidence %.0f%% too low for meaningful position, skipping", signal.confidence * 100)
            return results

        now = time.time()
        for action in signal.actions:
            if action.symbol in self._cooldowns and now < self._cooldowns[action.symbol]:
                logger.info("Symbol %s on cooldown, skipping", action.symbol)
                continue

            side = (
                self._OrderSide.BUY
                if action.side == Side.BUY
                else self._OrderSide.SELL
            )
            try:
                order = client.submit_order(
                    self._build_order(action.symbol, side, per_trade)
                )
                self._cooldowns[action.symbol] = now + COOLDOWN_SECONDS
                results.append(
                    {
                        "symbol": action.symbol,
                        "side": action.side.value,
                        "notional_usd": per_trade,
                        "status": str(order.status),
                        "order_id": str(order.id),
                    }
                )
                logger.info(
                    "Alpaca order %s: %s %s $%.2f",
                    order.id,
                    action.side.value.upper(),
                    action.symbol,
                    per_trade,
                )
                self._schedule_close(action.symbol, action.side, per_trade)
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
                if fill_price:
                    self._submit_trailing_stop(action.symbol, action.side, per_trade, fill_price)
                    self._submit_take_profit(action.symbol, action.side, per_trade, fill_price)
            except Exception as exc:
                logger.error("Order failed for %s: %s", action.symbol, exc)
        return results

    def _submit_trailing_stop(self, symbol: str, side: Side, notional: float, entry_price: float) -> None:
        trail_pct = self.config.stop_loss_pct * 100
        qty = round(notional / entry_price, 4)

        if side == Side.BUY:
            stop_side = self._OrderSide.SELL
        else:
            stop_side = self._OrderSide.BUY

        try:
            order = self._get_client().submit_order(
                self._MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=stop_side,
                    type=self._OrderType.TRAILING_STOP,
                    trail_percent=trail_pct,
                    time_in_force=self._TimeInForce.DAY,
                )
            )
            logger.info(
                "Trailing stop set: %s %.4f %s at %.0f%% (entry $%.2f, order %s)",
                stop_side.value.upper(), qty, symbol, trail_pct, entry_price, order.id,
            )
        except Exception as exc:
            logger.error("Trailing stop failed for %s: %s", symbol, exc)

    def _submit_take_profit(self, symbol: str, side: Side, notional: float, entry_price: float) -> None:
        tp_pct = self.config.take_profit_pct
        qty = round(notional / entry_price, 4)

        if side == Side.BUY:
            limit_price = round(entry_price * (1 + tp_pct), 2)
            tp_side = self._OrderSide.SELL
        else:
            limit_price = round(entry_price * (1 - tp_pct), 2)
            tp_side = self._OrderSide.BUY

        try:
            order = self._get_client().submit_order(
                self._MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    limit_price=limit_price,
                    side=tp_side,
                    type=self._OrderType.LIMIT,
                    time_in_force=self._TimeInForce.DAY,
                )
            )
            logger.info(
                "Take-profit set: %s %.4f %s at $%.2f (entry $%.2f, order %s)",
                tp_side.value.upper(), qty, symbol, limit_price, entry_price, order.id,
            )
        except Exception as exc:
            logger.error("Take-profit failed for %s: %s", symbol, exc)

    def _schedule_close(self, symbol: str, side: Side, notional: float) -> None:
        close_side = Side.BUY if side == Side.SELL else Side.SELL
        threading.Timer(CLOSE_DELAY, self._close_position, args=[symbol, close_side, notional]).start()
        logger.info("Scheduled close: %s %s $%.2f in 30m", close_side.value.upper(), symbol, notional)

    def _close_position(self, symbol: str, side: Side, notional: float) -> None:
        try:
            client = self._get_client()
            order_side = self._OrderSide.BUY if side == Side.BUY else self._OrderSide.SELL
            order = client.submit_order(
                self._MarketOrderRequest(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=order_side,
                    time_in_force=self._TimeInForce.DAY,
                )
            )
            logger.info(
                "Closed position: %s %s $%.2f (order %s)",
                side.value.upper(), symbol, notional, order.id,
            )
        except Exception as exc:
            logger.error("Close failed for %s: %s", symbol, exc)


def create_trader(config: TradingConfig, storage: Storage) -> Trader:
    mode = config.mode.lower()
    if mode in ("paper", "live"):
        return AlpacaTrader(config, storage)
    return DryRunTrader(config, storage)
