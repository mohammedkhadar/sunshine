from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from sunshine.models import Side, Signal, TradeAction, TruthPost

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
FREQ_WINDOW = 30 * 60
FREQ_BOOST_PER_HIT = 0.05
FREQ_BOOST_MAX = 0.15


def extract_tickers(text: str) -> list[str]:
    return list(dict.fromkeys(TICKER_PATTERN.findall(text.upper())))


class SignalAnalyzer:
    def __init__(self, playbook: dict[str, Any], min_confidence: float = 0.55) -> None:
        self.playbook = playbook
        self.min_confidence = min_confidence
        self._anthropic = self._init_llm()
        self._category_hits: dict[str, list[float]] = {}

    def _init_llm(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic

            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            return None

    def _frequency_boost(self, category: str) -> float:
        now = time.time()
        times = self._category_hits.setdefault(category, [])
        times.append(now)
        times[:] = [t for t in times if now - t < FREQ_WINDOW]
        count = len(times)
        if count >= 2:
            return min(FREQ_BOOST_PER_HIT * (count - 1), FREQ_BOOST_MAX)
        return 0.0

    def analyze(self, post: TruthPost) -> Signal | None:
        rule_signal = self._rule_based(post)
        if self._anthropic and rule_signal:
            return self._enrich_with_llm(post, rule_signal)
        return rule_signal

    def _rule_based(self, post: TruthPost) -> Signal | None:
        text = post.text_lower
        best: Signal | None = None

        for category, rules in self.playbook.items():
            keywords = [k.lower() for k in rules.get("keywords", [])]
            matched = [k for k in keywords if k in text]
            if not matched:
                continue

            confidence = min(0.45 + 0.08 * len(matched), 0.85)
            sentiment = rules.get("sentiment", "neutral")

            negative = [p.lower() for p in rules.get("negative_phrases", [])]
            positive = [p.lower() for p in rules.get("positive_phrases", [])]

            if any(p in text for p in negative):
                confidence += 0.12
            if any(p in text for p in positive):
                confidence = max(confidence - 0.2, 0.3)
                sentiment = "bullish_market"

            confidence += self._frequency_boost(category)
            confidence = min(confidence, 0.95)
            if confidence < self.min_confidence:
                continue

            long_symbols = list(rules.get("long", []))
            short_symbols = list(rules.get("short", []))
            bearish_symbols = list(rules.get("bearish", []))
            actions: list[TradeAction] = []

            for symbol in long_symbols:
                actions.append(
                    TradeAction(
                        symbol=symbol,
                        side=Side.BUY,
                        notional_usd=0,
                        reason=f"{category}: keyword match ({', '.join(matched)})",
                    )
                )
            for symbol in bearish_symbols:
                actions.append(
                    TradeAction(
                        symbol=symbol,
                        side=Side.BUY,
                        notional_usd=0,
                        reason=f"{category}: bearish hedge ({', '.join(matched)})",
                    )
                )
            for symbol in short_symbols:
                actions.append(
                    TradeAction(
                        symbol=symbol,
                        side=Side.SELL,
                        notional_usd=0,
                        reason=f"{category}: keyword match ({', '.join(matched)})",
                    )
                )

            for ticker in extract_tickers(post.content):
                if ticker not in {a.symbol for a in actions}:
                    actions.append(
                        TradeAction(
                            symbol=ticker,
                            side=Side.BUY,
                            notional_usd=0,
                            reason=f"Explicit ticker mention in post",
                        )
                    )
                    confidence = min(confidence + 0.1, 0.98)

            candidate = Signal(
                post_id=post.id,
                post_text=post.content[:500],
                category=category,
                sentiment=sentiment,
                confidence=confidence,
                actions=actions,
                matched_keywords=matched,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best

    def _enrich_with_llm(self, post: TruthPost, signal: Signal) -> Signal:
        try:
            prompt = (
                "You are a market analyst. Given this Trump Truth Social post and a draft signal, "
                "reply with ONE line: CONFIDENCE=<0-1> ACTION=hold|trade SUMMARY=<10 words max>\n\n"
                f"Post: {post.content[:800]}\n"
                f"Draft category: {signal.category}, confidence: {signal.confidence:.2f}"
            )
            msg = self._anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            signal.llm_summary = text

            conf_match = re.search(r"CONFIDENCE=([\d.]+)", text)
            if conf_match:
                signal.confidence = float(conf_match.group(1))

            if "ACTION=hold" in text and signal.confidence < self.min_confidence:
                return None
        except Exception as exc:
            logger.warning("LLM enrichment failed: %s", exc)

        return signal
