from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from sunshine.config import FetcherConfig
from sunshine.models import TruthPost

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "SunshineBot/1.0 (research; +https://github.com)"}


def strip_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class PostFetcher(ABC):
    @abstractmethod
    def fetch_latest(self, since_id: str | None = None, limit: int = 20) -> list[TruthPost]:
        ...


class CnnArchiveFetcher(PostFetcher):
    """Uses CNN's public Truth Social archive (updated ~every 5 minutes)."""

    def __init__(self, config: FetcherConfig) -> None:
        self.url = config.cnn_archive_url
        self._cache: list[TruthPost] | None = None

    def _load_all(self) -> list[TruthPost]:
        if self._cache is not None:
            return self._cache

        resp = requests.get(self.url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        posts: list[TruthPost] = []
        for item in resp.json():
            posts.append(
                TruthPost(
                    id=str(item["id"]),
                    content=item.get("content", ""),
                    created_at=parse_timestamp(item["created_at"]),
                    url=item.get("url", ""),
                    source="cnn_archive",
                )
            )
        posts.sort(key=lambda p: int(p.id), reverse=True)
        self._cache = posts
        return posts

    def fetch_latest(self, since_id: str | None = None, limit: int = 20) -> list[TruthPost]:
        self._cache = None
        all_posts = self._load_all()
        if since_id is None:
            return all_posts[:limit]

        since_int = int(since_id)
        new_posts = [p for p in all_posts if int(p.id) > since_int]
        return new_posts[:limit]


class TruthSocialFetcher(PostFetcher):
    """Direct Mastodon-compatible API. May require curl_cffi to bypass Cloudflare."""

    def __init__(self, config: FetcherConfig) -> None:
        self.base = config.truth_social_base.rstrip("/")
        self.account_id = config.account_id
        self._session = self._build_session()

    def _build_session(self):
        try:
            from curl_cffi import requests as cffi_requests

            session = cffi_requests.Session(impersonate="chrome")
            session.headers.update(HEADERS)
            return session
        except ImportError:
            session = requests.Session()
            session.headers.update(HEADERS)
            return session

    def fetch_latest(self, since_id: str | None = None, limit: int = 20) -> list[TruthPost]:
        params: dict[str, str | int] = {"limit": limit, "exclude_replies": "true"}
        if since_id:
            params["since_id"] = since_id

        url = f"{self.base}/api/v1/accounts/{self.account_id}/statuses"
        resp = self._session.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            logger.warning("Truth Social rate limited (429)")
            return []
        if resp.status_code == 403:
            logger.warning("Truth Social blocked by Cloudflare — use cnn_archive fetcher")
            return []

        resp.raise_for_status()
        posts: list[TruthPost] = []
        for item in resp.json():
            if item.get("reblog"):
                continue
            posts.append(
                TruthPost(
                    id=str(item["id"]),
                    content=strip_html(item.get("content", "")),
                    created_at=parse_timestamp(item["created_at"]),
                    url=item.get("url", ""),
                    source="truth_social",
                )
            )
        return posts


def create_fetcher(config: FetcherConfig) -> PostFetcher:
    if config.primary == "truth_social":
        return TruthSocialFetcher(config)
    return CnnArchiveFetcher(config)
