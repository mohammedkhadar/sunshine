from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass
class TradingConfig:
    mode: str = "dry_run"
    max_position_usd: float = 1000.0
    max_daily_trades: int = 10
    min_confidence: float = 0.55
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.03


@dataclass
class FetcherConfig:
    primary: str = "cnn_archive"
    cnn_archive_url: str = "https://ix.cnn.io/data/truth-social/truth_archive.json"
    truth_social_base: str = "https://truthsocial.com"
    poll_limit: int = 20
    poll_interval: float = 5.0
    username: str = "realDonaldTrump"
    account_id: str = "107780257626128497"


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass
class AppConfig:
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    playbook: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}

    account = raw.get("account", {})
    fetcher_raw = raw.get("fetcher", {})
    trading_raw = raw.get("trading", {})
    llm_raw = raw.get("llm", {})

    fetcher = FetcherConfig(
        primary=fetcher_raw.get("primary", "cnn_archive"),
        cnn_archive_url=fetcher_raw.get(
            "cnn_archive_url",
            "https://ix.cnn.io/data/truth-social/truth_archive.json",
        ),
        truth_social_base=fetcher_raw.get("truth_social_base", "https://truthsocial.com"),
        poll_limit=int(fetcher_raw.get("poll_limit", 20)),
        poll_interval=float(os.getenv("POLL_INTERVAL", "5")),
        username=account.get("username", "realDonaldTrump"),
        account_id=str(account.get("account_id", "107780257626128497")),
    )

    trading = TradingConfig(
        mode=os.getenv("TRADING_MODE", trading_raw.get("mode", "dry_run")),
        max_position_usd=float(trading_raw.get("max_position_usd", 1000)),
        max_daily_trades=int(trading_raw.get("max_daily_trades", 10)),
        min_confidence=float(trading_raw.get("min_confidence", 0.55)),
        stop_loss_pct=float(trading_raw.get("stop_loss_pct", 0.05)),
        take_profit_pct=float(trading_raw.get("take_profit_pct", 0.03)),
    )

    llm = LLMConfig(
        provider=llm_raw.get("provider", "openrouter"),
        model=llm_raw.get("model", "openai/gpt-4o-mini"),
        base_url=llm_raw.get("base_url", "https://openrouter.ai/api/v1"),
        api_key_env=llm_raw.get("api_key_env", "OPENROUTER_API_KEY"),
    )

    return AppConfig(
        fetcher=fetcher,
        trading=trading,
        llm=llm,
        playbook=raw.get("playbook", {}),
    )
