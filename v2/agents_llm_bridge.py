from __future__ import annotations
import json, os
from typing import Dict, List, Tuple

from .llm_client import chat_json, LLMError
from .event_cache import save as cache_save, recent as cache_recent, last_ok_within


def _compose(headlines_by_symbol: Dict[str, List[str]]) -> str:
    lines = []
    for sym, heads in (headlines_by_symbol or {}).items():
        if not heads:
            continue
        lines.append(f"{sym}:")
        for h in heads[:6]:
            lines.append(" - " + h.strip())
    return "\n".join(lines) if lines else "No headlines."


def _parse(txt: str) -> Dict[str, float]:
    try:
        obj = json.loads(txt)
        return {k.upper(): float(max(-3.0, min(3.0, float(v)))) for k, v in obj.items()}
    except Exception:
        return {}


def score_with_failover(headlines_by_symbol: Dict[str, List[str]]) -> Tuple[Dict[str, float], str]:
    ttl = int(os.getenv("EVENT_CACHE_TTL_MIN", "720"))
    try:
        raw = chat_json(_compose(headlines_by_symbol))
        scores = _parse(raw)
        if scores:
            cache_save(scores)
            return scores, "live"
        cache, ok = cache_recent(ttl)
        return (cache, "cache") if ok else ({}, "empty")
    except LLMError as e:
        cache, ok = cache_recent(ttl)
        return (cache, "cache") if ok and e.kind != "fatal" else ({}, f"error:{e.kind}")


def require_recent_assessment() -> bool:
    if os.getenv("REQUIRE_EVENT_ASSESSMENT", "0") != "1":
        return True
    return last_ok_within(int(os.getenv("EVENT_MAX_AGE_MIN", "120")))


def apply_patch_into_agents():
    # Wrap likely scoring functions in v2.agents to force this bridge.
    try:
        import importlib

        agents = importlib.import_module("v2.agents")
    except Exception as e:
        print("[llm_bridge] cannot import v2.agents:", e)
        return 0
    patched = 0

    def _wrap(fn):
        def f(*a, **k):
            hbys = k.get("headlines_by_symbol") or (a[0] if a and isinstance(a[0], dict) else None)
            if hbys is None:
                return fn(*a, **k)
            scores, source = score_with_failover(hbys)
            return {"scores": scores, "source": source}

        return f

    for name, obj in list(vars(agents).items()):
        if callable(obj) and any(w in name.lower() for w in ["event", "news", "score"]):
            try:
                setattr(agents, name, _wrap(obj))
                patched += 1
            except Exception:
                pass
    print("[llm_bridge] patched:", patched)
    return patched
