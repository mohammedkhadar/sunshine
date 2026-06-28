from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from sunshine.config import TradingConfig
from sunshine.models import Side, Signal, TradeAction
from sunshine.storage import Storage

logger = logging.getLogger(__name__)


class Trader(ABC):
    @abstractmethod
    def execute(self, signal: Signal, signal_id: int) -> list[dict]:
        ...


class DryRunTrader(Trader):
    def __init__(self, config: TradingConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage

    def execute(self, signal: Signal, signal_id: int) -> list[dict]:
        results: list[dict] = []
        if self.storage.trades_today_count() >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached (%s)", self.config.max_daily_trades)
            return results

        per_trade = self.config.max_position_usd / max(len(signal.actions), 1)

        for action in signal.actions:
            action.notional_usd = per_trade
            self.storage.save_trade(
                signal_id=signal_id,
                symbol=action.symbol,
                side=action.side.value,
                notional_usd=per_trade,
                status="dry_run",
                reason=action.reason,
            )
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
        return results


class AlpacaTrader(Trader):
    def __init__(self, config: TradingConfig, storage: Storage) -> None:
        self.config = config
        self.storage = storage
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key or not self.secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required for live/paper trading")

        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest
        return self._client

    def execute(self, signal: Signal, signal_id: int) -> list[dict]:
        results: list[dict] = []
        if self.storage.trades_today_count() >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return results

        client = self._get_client()
        per_trade = self.config.max_position_usd / max(len(signal.actions), 1)

        for action in signal.actions:
            side = (
                self._OrderSide.BUY
                if action.side == Side.BUY
                else self._OrderSide.SELL
            )
            try:
                order = client.submit_order(
                    self._MarketOrderRequest(
                        symbol=action.symbol,
                        notional=round(per_trade, 2),
                        side=side,
                        time_in_force=self._TimeInForce.DAY,
                    )
                )
                self.storage.save_trade(
                    signal_id=signal_id,
                    symbol=action.symbol,
                    side=action.side.value,
                    notional_usd=per_trade,
                    status=str(order.status),
                    broker_order_id=str(order.id),
                    reason=action.reason,
                )
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
            except Exception as exc:
                logger.error("Order failed for %s: %s", action.symbol, exc)
                self.storage.save_trade(
                    signal_id=signal_id,
                    symbol=action.symbol,
                    side=action.side.value,
                    notional_usd=per_trade,
                    status="failed",
                    reason=str(exc),
                )
        return results


def create_trader(config: TradingConfig, storage: Storage) -> Trader:
    mode = config.mode.lower()
    if mode in ("paper", "live"):
        return AlpacaTrader(config, storage)
    return DryRunTrader(config, storage)
