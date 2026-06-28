from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class TruthPost:
    id: str
    content: str
    created_at: datetime
    url: str = ""
    source: str = "unknown"

    @property
    def text_lower(self) -> str:
        return self.content.lower()


@dataclass
class TradeAction:
    symbol: str
    side: Side
    notional_usd: float
    reason: str


@dataclass
class Signal:
    post_id: str
    post_text: str
    category: str
    sentiment: str
    confidence: float
    actions: list[TradeAction] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    llm_summary: str = ""
