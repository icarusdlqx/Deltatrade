from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

CACHE = Path("data/event_cache.json")
CACHE.parent.mkdir(parents=True, exist_ok=True)


def _now():
    return datetime.now(timezone.utc)


def _load():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            pass
    return {"last_success_iso": None, "scores": {}}


def _save(d):
    CACHE.write_text(json.dumps(d, indent=2))


def save(scores: dict):
    d = _load()
    ts = _now().isoformat(timespec="seconds")
    for k, v in (scores or {}).items():
        d["scores"][k.upper()] = {"score": float(v), "ts_iso": ts}
    d["last_success_iso"] = ts
    _save(d)


def recent(ttl_min: int):
    d = _load()
    out = {}
    ok = False
    ttl = timedelta(minutes=int(ttl_min))
    for sym, rec in d.get("scores", {}).items():
        try:
            t = datetime.fromisoformat(rec["ts_iso"])
        except Exception:
            continue
        if _now() - t <= ttl:
            out[sym] = float(rec["score"])
            ok = True
    return out, ok


def last_ok_within(mins: int):
    d = _load()
    ts = d.get("last_success_iso")
    if not ts:
        return False
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return False
    return (_now() - t).total_seconds() <= 60 * int(mins)
