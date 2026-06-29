from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from sunshine.models import TruthPost
from sunshine.config import LLMConfig

logger = logging.getLogger(__name__)

TOPIC_KEYWORDS = [
    "tariff", "tariffs", "trade war",
    "federal reserve", "fed chair", "powell", "interest rate",
    "oil", "drill", "energy dominance", "lng", "pipeline", "opec",
    "sanctions", "sanction",
    "china", "russia", "north korea", "iran",
    "military", "defense", "pentagon",
    "farmer", "farmers", "agriculture", "crops", "dairy",
]

SECTOR_SYMBOLS: dict[str, dict[str, list[str]]] = {
    "tariffs": {
        "bullish": ["NUE", "AA"],
        "bearish": [],
    },
    "agriculture": {
        "bullish": ["DE", "CAT"],
        "bearish": [],
    },
    "fed": {
        "bullish": ["GLD"],
        "bearish": [],
    },
    "oil": {
        "bullish": ["BZ=F"],
        "bearish": [],
    },
    "sanctions": {
        "bullish": ["LMT", "RTX", "NOC"],
        "bearish": [],
    },
}


class ImpactScorer:
    """Scores posts for market impact on a 1-10 scale.
    1-3 minimal, 4-6 moderate, 7-8 significant, 9-10 major.
    Only scores posts matching predefined topics (tariffs, agriculture, Fed, oil, sanctions).
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        threshold: float = 7.0,
        cache_path: str | None = None,
    ) -> None:
        self.threshold = threshold
        self._cache: dict[str, dict | None] = {}
        self._cache_path = cache_path
        self._client: OpenRouterClient | None = None
        api_key = os.getenv(llm_config.api_key_env)
        if api_key:
            self._client = OpenRouterClient(api_key, llm_config.base_url)
            self.model = llm_config.model
        if cache_path:
            self._load_cache()

    def _load_cache(self) -> None:
        from pathlib import Path
        p = Path(self._cache_path)
        if p.exists():
            try:
                with p.open() as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached scores from %s", len(self._cache), self._cache_path)
            except Exception as exc:
                logger.warning("Failed to load score cache: %s", exc)

    def _save_cache(self) -> None:
        if not self._cache_path or not self._cache:
            return
        from pathlib import Path
        p = Path(self._cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            serializable = {}
            for k, v in self._cache.items():
                if v is None:
                    serializable[k] = None
                else:
                    entry = dict(v)
                    if hasattr(entry.get("created_at"), "isoformat"):
                        entry["created_at"] = entry["created_at"].isoformat()
                    serializable[k] = entry
            with p.open("w") as f:
                json.dump(serializable, f)
        except Exception as exc:
            logger.warning("Failed to save score cache: %s", exc)

    def matches_topics(self, post: TruthPost) -> bool:
        text = post.text_lower
        return any(t in text for t in TOPIC_KEYWORDS)

    def _build_prompt(self, post: TruthPost) -> str:
        return (
            "You are a quantitative event-driven analyst. "
            "Rate the market impact of this Trump Truth Social post on a scale of 1-10.\n\n"
            "1-3 = Minimal (vague, rhetorical, no policy substance)\n"
            "4-6 = Moderate (relevant topic but no specifics)\n"
            "7-8 = Significant (clear policy direction, will move sector)\n"
            "9-10 = Major (concrete announcement, market-moving)\n\n"
            f"Post: {post.content[:1500]}\n\n"
            "Respond with valid JSON only (no markdown, no extra text):\n"
            '{"score":<1-10>,"direction":"bullish","sector":"tariffs","reasoning":"one sentence"}\n\n'
            "direction must be one of: bullish, bearish, neutral\n"
            "sector must be one of: tariffs, agriculture, fed, oil, sanctions, other"
        )

    def _parse(self, raw: dict) -> dict | None:
        score = raw.get("score")
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            logger.warning("Invalid score from LLM: %s", score)
            return None
        direction = raw.get("direction", "neutral")
        if not isinstance(direction, str) or direction not in ("bullish", "bearish", "neutral"):
            direction = "neutral"
        sector = raw.get("sector", "other")
        if not isinstance(sector, str) or sector not in SECTOR_SYMBOLS:
            sector = "other"
        reasoning = raw.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning) if reasoning else ""
        return {
            "score": int(score),
            "direction": direction,
            "sector": sector,
            "reasoning": reasoning,
        }

    def _symbols_for(self, sector: str, direction: str) -> list[str]:
        mapping = SECTOR_SYMBOLS.get(sector, {})
        if direction == "bullish":
            return mapping.get("bullish", [])
        elif direction == "bearish":
            return mapping.get("bearish", [])
        return []

    def score(self, post: TruthPost) -> dict | None:
        if post.id in self._cache:
            cached = self._cache[post.id]
            if cached is None:
                return None
            if isinstance(cached, dict):
                if isinstance(cached.get("created_at"), str):
                    from datetime import datetime, timezone
                    cached["created_at"] = datetime.fromisoformat(cached["created_at"])
                cached["symbols"] = self._symbols_for(cached.get("sector", "other"), cached.get("direction", "neutral"))
            return cached

        if not self.matches_topics(post):
            self._cache[post.id] = None
            return None

        if not self._client:
            return None

        result = self._client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": self._build_prompt(post)}],
        )
        if result is None:
            self._cache[post.id] = None
            self._save_cache()
            return None

        parsed = self._parse(result)
        if parsed is None:
            self._cache[post.id] = None
            self._save_cache()
            return None

        parsed["symbols"] = self._symbols_for(parsed["sector"], parsed["direction"])
        parsed["post_id"] = post.id
        parsed["created_at"] = post.created_at
        parsed["post_text"] = post.content[:500]

        self._cache[post.id] = parsed
        self._save_cache()

        if parsed["score"] < self.threshold:
            return None

        return parsed

    def score_many(self, posts: list[TruthPost]) -> list[dict]:
        import time as _time
        results = []
        api_calls = 0
        for i, post in enumerate(posts):
            if i > 0 and i % 50 == 0:
                self._save_cache()
            cached = post.id in self._cache
            scored = self.score(post)
            if not cached and self._client:
                api_calls += 1
                if api_calls % 10 == 0:
                    _time.sleep(1)
            if scored:
                results.append(scored)
        self._save_cache()
        return results


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> dict | None:
        import requests as req
        import time as _time

        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = req.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                parsed = self._safe_json_load(content)
                if parsed is not None:
                    return parsed
                return {"text": content}
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("OpenRouter attempt %d failed, retrying in %ds: %s", attempt + 1, wait, exc)
                    _time.sleep(wait)
        logger.warning("OpenRouter API call failed after %d retries: %s", max_retries, last_exc)
        return None

    def _safe_json_load(self, raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse LLM response as JSON: %.200s", raw)
        return None
