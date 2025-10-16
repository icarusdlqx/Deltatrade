from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List

from .llm_gpt5 import LLMError, score_events_gpt5


_WRAPPED_ATTR = "__llm_enforced__"


def _extract_headlines(raw: Mapping[str, Iterable[Any]] | None) -> Dict[str, List[str]]:
    headlines: Dict[str, List[str]] = {}
    if not raw:
        return headlines

    for sym, items in raw.items():
        bucket: List[str] = []
        if not items:
            continue
        for item in items:
            text = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, Mapping):
                headline = str(item.get("headline") or item.get("title") or "").strip()
                summary = str(item.get("summary") or item.get("summary_short") or "").strip()
                if headline and summary:
                    text = f"{headline} :: {summary}"
                else:
                    text = headline or summary
            else:
                text = str(item).strip()

            if text:
                bucket.append(text)

        if bucket:
            headlines[str(sym).upper()] = bucket

    return headlines


def _wrap_score_events(fn):
    if getattr(fn, _WRAPPED_ATTR, False):
        return fn

    def wrapped(*args: Any, **kwargs: Any):
        news = None
        if args and isinstance(args[0], dict):
            news = args[0]
        if news is None:
            for key in ("news_by_symbol", "headlines_by_symbol"):
                candidate = kwargs.get(key)
                if isinstance(candidate, dict):
                    news = candidate
                    break

        raw_news = news or {}
        symbols = [str(sym).upper() for sym in raw_news.keys()]
        headlines = _extract_headlines(raw_news)

        try:
            payload = score_events_gpt5(headlines)
            raw_scores = payload.get("scores", {})
            max_abs = 0.0
            if len(args) >= 5 and isinstance(args[4], (int, float)):
                max_abs = float(args[4])
            else:
                max_abs = float(kwargs.get("max_abs_bps", 0.0) or 0.0)
            scale = max_abs / 3.0 if max_abs else 0.0
            event_scores = {
                sym: max(-max_abs, min(max_abs, float(score) * scale)) if scale else 0.0
                for sym, score in raw_scores.items()
            }
            for sym in symbols:
                event_scores.setdefault(sym, 0.0)
            usage = payload.get("usage") or {}
            tokens = 0
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    tokens += int(usage.get(key) or 0)
                except Exception:
                    continue
            meta = {
                "called": True,
                "model": payload.get("model") or "gpt-5",
                "effort": "medium",
                "tokens": tokens,
                "n_tickers": len(symbols),
                "source": payload.get("source"),
                "raw": payload.get("raw"),
            }
            if symbols and not event_scores:
                event_scores = {sym: 0.0 for sym in symbols}
            return event_scores, meta
        except LLMError as exc:
            print("[llm_enforce] LLM call failed:", exc.kind, str(exc))
            meta = {
                "called": False,
                "model": "gpt-5",
                "effort": "medium",
                "tokens": 0,
                "n_tickers": len(symbols),
                "reason": exc.kind,
                "error": str(exc),
            }
            return {sym: 0.0 for sym in symbols}, meta
        except Exception as exc:  # pragma: no cover - safeguard
            print("[llm_enforce] unexpected error:", exc)
            return fn(*args, **kwargs)

    setattr(wrapped, _WRAPPED_ATTR, True)
    return wrapped


def _wrap_orchestrator(fn):
    if getattr(fn, _WRAPPED_ATTR, False):
        return fn

    def wrapped(*args: Any, **kwargs: Any):
        try:
            score_events_gpt5({"KEEPALIVE": ["No material headlines this window."]})
        except Exception:
            pass
        return fn(*args, **kwargs)

    setattr(wrapped, _WRAPPED_ATTR, True)
    return wrapped


def apply_patch() -> int:
    patched = 0

    try:
        agents = importlib.import_module("v2.agents")
    except Exception as exc:  # pragma: no cover - import edge
        print("[llm_enforce] cannot import v2.agents:", exc)
        agents = None

    if agents:
        target = getattr(agents, "score_events_for_symbols", None)
        if callable(target):
            wrapped = _wrap_score_events(target)
            if wrapped is not target:
                setattr(agents, "score_events_for_symbols", wrapped)
                patched += 1

    try:
        orchestrator = importlib.import_module("v2.orchestrator")
        for attr in ("run_once", "run", "cycle", "single_cycle"):
            func = getattr(orchestrator, attr, None)
            if callable(func):
                wrapped = _wrap_orchestrator(func)
                if wrapped is not func:
                    setattr(orchestrator, attr, wrapped)
                    patched += 1
                break
    except Exception as exc:  # pragma: no cover - import edge
        print("[llm_enforce] orchestrator patch skipped:", exc)

    print("[llm_enforce] patched:", patched)
    return patched

