from __future__ import annotations

import json
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


class OpenRouterClient:
    """Minimal OpenRouter API client using requests."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict | None:
        import requests as req

        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            body["response_format"] = response_format

        try:
            resp = req.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            parsed = self._safe_json_load(content)
            if parsed is not None:
                return parsed
            return {"text": content}
        except Exception as exc:
            logger.warning("OpenRouter API call failed: %s", exc)
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


class LLMAnalyzer:
    """Pure LLM signal analyzer using OpenRouter API.
    No rule-based matching — the LLM decides everything from the post text.
    """

    def __init__(
        self,
        playbook: dict[str, Any],
        min_confidence: float = 0.55,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
    ) -> None:
        self.playbook = playbook
        self.min_confidence = min_confidence
        self.model = model
        self._cache: dict[str, Signal | None] = {}
        self._client: OpenRouterClient | None = None
        if api_key:
            self._client = OpenRouterClient(api_key, base_url)

    @classmethod
    def from_config(cls, playbook: dict, trading_config, llm_config) -> LLMAnalyzer:
        api_key = os.getenv(llm_config.api_key_env)
        if not api_key:
            logger.warning(
                "%s not set — LLM analyzer will fall back to rule-based",
                llm_config.api_key_env,
            )
            return cls(playbook, trading_config.min_confidence)
        return cls(
            playbook=playbook,
            min_confidence=trading_config.min_confidence,
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_key=api_key,
        )

    def _build_ticker_context(self) -> str:
        lines: list[str] = []
        for cat, rules in self.playbook.items():
            syms = set()
            for k in ("long", "short", "bearish"):
                for s in rules.get(k, []):
                    syms.add(s)
            sent = rules.get("sentiment", "neutral")
            lines.append(f"- {cat} ({sent}): {', '.join(sorted(syms))}")
        return "\n".join(lines)

    def _build_messages(self, post: TruthPost) -> list[dict]:
        ticker_ctx = self._build_ticker_context()
        system = (
            "You are a conservative quantitative trading analyst. "
            "Only generate a trade signal when the post has a clear, direct connection to a specific ticker or sector. "
            "If the post is vague, rhetorical, or not clearly market-moving, respond with {\"action\":\"hold\"}. "
            "Use ONLY tickers from the provided list. "
            "Respond with valid JSON only — no markdown, no commentary."
        )
        user = (
            f"Analyze this Truth Social post for trading signals.\n\n"
            f"Available tickers by category:\n{ticker_ctx}\n\n"
            f"Post: {post.content[:1500]}\n\n"
            f"Rules:\n"
            f"1. If no clear trade signal, respond: {{\"action\":\"hold\"}}\n"
            f"2. Only use tickers from the list above\n"
            f"3. category must be one of the listed categories\n"
            f"4. Use exact symbol names (e.g. XOM not Exxon)\n\n"
            f"Respond with JSON:\n"
            f'{{"action":"hold"|"trade","confidence":0.0-1.0,'
            f'"sentiment":"bearish_market"|"bullish_market"|"bullish_sector"|"bullish_crypto"|"neutral",'
            f'"category":"best_matching_category","reasoning":"one sentence",'
            f'"trades":[{{"symbol":"TICKER","side":"buy"|"sell","reason":"brief reason"}}]}}'
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_response(self, post: TruthPost, data: dict) -> Signal | None:
        action = data.get("action", "hold")
        if action != "trade":
            return None

        confidence = float(data.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None

        sentiment = data.get("sentiment", "neutral")
        category = data.get("category", "unknown")
        reasoning = data.get("reasoning", "")
        trades_raw = data.get("trades", [])

        if not trades_raw:
            return None

        actions: list[TradeAction] = []
        for t in trades_raw:
            symbol = str(t.get("symbol", "")).upper().strip()
            side_str = str(t.get("side", "")).lower().strip()
            reason = str(t.get("reason", ""))
            if not symbol or side_str not in ("buy", "sell"):
                continue
            side = Side.BUY if side_str == "buy" else Side.SELL
            actions.append(TradeAction(symbol=symbol, side=side, notional_usd=0, reason=reason))

        if not actions:
            return None

        confidence = min(confidence, 0.98)
        return Signal(
            post_id=post.id,
            post_text=post.content[:500],
            category=category,
            sentiment=sentiment,
            confidence=confidence,
            actions=actions,
            matched_keywords=[],
            llm_summary=reasoning,
        )

    def analyze(self, post: TruthPost) -> Signal | None:
        if post.id in self._cache:
            return self._cache[post.id]

        if not self._client:
            return None

        messages = self._build_messages(post)
        try:
            result = self._client.chat_completion(
                model=self.model,
                messages=messages,
            )
        except Exception as exc:
            logger.warning("LLM analyze failed for post %s: %s", post.id, exc)
            self._cache[post.id] = None
            return None

        if result is None:
            self._cache[post.id] = None
            return None

        signal = self._parse_response(post, result)
        self._cache[post.id] = signal
        return signal


class LLMRefiner:
    """LLM-based signal validator: takes a rule-based signal and adjusts confidence.
    Does NOT generate new symbols or categories — only validates the existing signal.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._cache: dict[str, dict] = {}
        self._client: OpenRouterClient | None = None
        if api_key:
            self._client = OpenRouterClient(api_key, base_url)

    @classmethod
    def from_config(cls, playbook, trading_config, llm_config) -> LLMRefiner:
        api_key = os.getenv(llm_config.api_key_env)
        if not api_key:
            logger.warning(
                "%s not set — LLM refiner will be disabled",
                llm_config.api_key_env,
            )
            return cls(model=llm_config.model, base_url=llm_config.base_url)
        return cls(
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_key=api_key,
        )

    def refine(self, post: TruthPost, signal: Signal) -> dict | None:
        if post.id in self._cache:
            return self._cache[post.id]

        if not self._client:
            return {"confidence": signal.confidence, "reasoning": "no LLM"}

        actions_str = "; ".join(f"{a.side.value} {a.symbol}" for a in signal.actions)
        prompt = (
            "You are a conservative trading signal validator. "
            "Given a Trump Truth Social post and a proposed trade signal, "
            "determine whether the signal is justified.\n\n"
            f"Post: {post.content[:1000]}\n\n"
            f"Proposed signal:\n"
            f"- Category: {signal.category}\n"
            f"- Confidence: {signal.confidence:.2f}\n"
            f"- Actions: {actions_str}\n"
            f"- Matched keywords: {', '.join(signal.matched_keywords)}\n\n"
            f"Respond with JSON only:\n"
            f'{{"confidence":0.0-1.0,"reasoning":"one sentence justification"}}\n\n'
            f"Guidelines:\n"
            f"- Return confidence >0.7 only if the post has DIRECT market-moving language\n"
            f"- Return confidence 0.0-0.3 if the post is vague, rhetorical, or irrelevant\n"
            f"- The confidence is your adjusted rating for this specific signal\n"
            f"- Do NOT suggest different trades — just evaluate this signal"
        )

        try:
            result = self._client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.warning("LLM refine failed for post %s: %s", post.id, exc)
            self._cache[post.id] = None
            return None

        if result is None:
            self._cache[post.id] = None
            return None

        self._cache[post.id] = result
        return result


class HybridAnalyzer:
    """Rule-based gatekeeper → LLM confidence refinement → Technical filter.
    Only calls the LLM when rule-based keywords match, saving ~80% of API costs.
    The LLM validates the rule-based signal and adjusts confidence — it does NOT
    suggest new symbols or categories.
    """

    def __init__(
        self,
        playbook: dict[str, Any],
        trading_config,
        llm_config,
    ) -> None:
        self.min_confidence = trading_config.min_confidence
        self.rule_analyzer = SignalAnalyzer(playbook, min_confidence=self.min_confidence)
        self.llm = LLMRefiner.from_config(playbook, trading_config, llm_config)
        self._technical = None
        self._enable_technical = False

    def enable_technical_filter(self, use_regime: bool = False) -> None:
        from sunshine.technical import RegimeFilter, TechnicalFilter

        regime = RegimeFilter() if use_regime else None
        self._technical = TechnicalFilter(regime_filter=regime)
        self._enable_technical = True

    def analyze(self, post: TruthPost) -> Signal | None:
        rule_signal = self.rule_analyzer.analyze(post)
        if rule_signal is None:
            return None

        if self.llm._client:
            refinement = self.llm.refine(post, rule_signal)
            if refinement is None:
                return None
            rule_signal.confidence = refinement.get("confidence", rule_signal.confidence)
            rule_signal.llm_summary = refinement.get("reasoning", "")
            rule_signal.confidence = min(rule_signal.confidence, 0.98)

        if rule_signal.confidence < self.min_confidence:
            return None

        if self._enable_technical and self._technical:
            filtered: list[TradeAction] = []
            for action in rule_signal.actions:
                passed, reason = self._technical.check(action.symbol, action.side, post.created_at)
                if passed:
                    filtered.append(action)
                else:
                    logger.info("Technical filter rejected %s %s: %s", action.side.value, action.symbol, reason)
            if not filtered:
                return None
            rule_signal.actions = filtered

        return rule_signal


def create_analyzer(
    playbook: dict,
    trading_config,
    llm_config=None,
) -> SignalAnalyzer | LLMAnalyzer | HybridAnalyzer:
    api_key_env = (llm_config.api_key_env or "OPENROUTER_API_KEY") if llm_config else "OPENROUTER_API_KEY"
    api_key = os.getenv(api_key_env)
    if api_key and llm_config:
        logger.info("Using LLMAnalyzer with model %s", llm_config.model)
        return LLMAnalyzer.from_config(playbook, trading_config, llm_config)
    logger.info("No LLM API key found — using rule-based SignalAnalyzer")
    return SignalAnalyzer(playbook, min_confidence=trading_config.min_confidence)
