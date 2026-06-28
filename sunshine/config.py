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
class AppConfig:
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
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
    )

    return AppConfig(
        fetcher=fetcher,
        trading=trading,
        playbook=raw.get("playbook", {}),
    )
