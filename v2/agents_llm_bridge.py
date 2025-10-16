from __future__ import annotations
import json, os
from typing import Dict, List, Any, Tuple
from .llm_client import chat_json, LLMError
from .event_cache import save_scores, get_recent_scores, record_failure, last_ok_within

def _compose_user_text(items: Dict[str, List[str]]) -> str:
    # Compact bullet list per ticker; LLM prompt instructs JSON output.
    lines = []
    for sym, heads in (items or {}).items():
        if not heads:
            continue
        lines.append(f"{sym}:")
        for h in heads[:6]:
            if isinstance(h, dict):
                text = h.get("headline") or h.get("title") or h.get("summary") or ""
            else:
                text = str(h)
            text = str(text).strip()
            if not text:
                continue
            lines.append(f" - {text}")
    return "\n".join(lines) if lines else "No headlines."

def _parse_scores(txt: str) -> Dict[str, float]:
    try:
        obj = json.loads(txt)
        out = {}
        for k, v in obj.items():
            try:
                out[k.upper()] = float(max(-3.0, min(3.0, float(v))))
            except Exception:
                continue
        return out
    except Exception:
        return {}

def score_events_with_failover(headlines_by_symbol: Dict[str, List[str]]) -> Tuple[Dict[str,float], str]:
    ttl = int(os.getenv("EVENT_CACHE_TTL_MIN", "720"))
    try:
        user_text = _compose_user_text(headlines_by_symbol)
        raw = chat_json(user_text)
        scores = _parse_scores(raw)
        if scores:
            save_scores(scores)
            return scores, "live"
        # Unexpected empty parse -> fallback to cache
        cached, ok = get_recent_scores(ttl)
        if ok:
            return cached, "cache"
        return {}, "empty"
    except LLMError as e:
        record_failure(e.err_type, e.message)
        cached, ok = get_recent_scores(ttl)
        if ok and e.err_type in ("transient","unknown"):
            return cached, "cache"
        return {}, f"error:{e.err_type}"

# -------- Runtime patch: force v2.agents event scoring through bridge ----
def apply_patch_into_agents():
    try:
        import importlib
        agents = importlib.import_module("v2.agents")
    except Exception as e:
        print("[llm_bridge] cannot import v2.agents:", e); return 0

    patched = 0

    def _wrap(fn):
        def f(*a, **k):
            # Try to extract {sym: [headlines]} from common calling patterns
            hbys = k.get("headlines_by_symbol") or k.get("items") or None
            if hbys is None and a and isinstance(a[0], dict):
                hbys = a[0]
            if hbys is None:
                # Fallback: if given list of texts, attribute to 'GEN'
                lst = k.get("headlines") or k.get("texts") or (a[0] if a and isinstance(a[0], list) else [])
                if lst:
                    hbys = {"GEN": lst}
            if hbys is None:
                # Unknown signature -> use original
                return fn(*a, **k)
            scores, source = score_events_with_failover(hbys)
            return {"scores": scores, "source": source}
        return f

    # Patch candidate names
    cand = []
    for name, obj in vars(agents).items():
        if callable(obj) and any(w in name.lower() for w in ["event","news","score"]):
            cand.append(name)
    for name in cand:
        try:
            old = getattr(agents, name)
            setattr(agents, name, _wrap(old))
            patched += 1
        except Exception:
            pass
    print(f"[llm_bridge] patched functions: {patched}")
    return patched

def require_recent_assessment() -> bool:
    if os.getenv("REQUIRE_EVENT_ASSESSMENT","0") != "1":
        return True
    max_age = int(os.getenv("EVENT_MAX_AGE_MIN","120"))
    return last_ok_within(max_age)
