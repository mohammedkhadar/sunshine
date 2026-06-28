from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sunshine.models import Signal, TruthPost

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "sunshine.db"


class Storage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    url TEXT,
                    source TEXT,
                    fetched_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT NOT NULL,
                    category TEXT,
                    sentiment TEXT,
                    confidence REAL,
                    matched_keywords TEXT,
                    llm_summary TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (post_id) REFERENCES posts(id)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    notional_usd REAL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                );

                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def get_last_seen_post_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key = 'last_seen_post_id'"
            ).fetchone()
            return row["value"] if row else None

    def set_last_seen_post_id(self, post_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (key, value) VALUES ('last_seen_post_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (post_id,),
            )

    def save_post(self, post: TruthPost) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO posts (id, content, created_at, url, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    post.id,
                    post.content,
                    post.created_at.isoformat(),
                    post.url,
                    post.source,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return cur.rowcount > 0

    def save_signal(self, signal: Signal) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals
                (post_id, category, sentiment, confidence, matched_keywords, llm_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.post_id,
                    signal.category,
                    signal.sentiment,
                    signal.confidence,
                    json.dumps(signal.matched_keywords),
                    signal.llm_summary,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def save_trade(
        self,
        signal_id: int | None,
        symbol: str,
        side: str,
        notional_usd: float,
        status: str,
        reason: str,
        broker_order_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades
                (signal_id, symbol, side, notional_usd, status, broker_order_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    symbol,
                    side,
                    notional_usd,
                    status,
                    broker_order_id,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def trades_today_count(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM trades
                WHERE created_at LIKE ?
                """,
                (f"{today}%",),
            ).fetchone()
            return int(row["cnt"])

    def recent_signals(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, p.content AS post_content
                FROM signals s
                JOIN posts p ON p.id = s.post_id
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
