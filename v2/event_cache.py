from __future__ import annotations
import json, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple

CACHE_FILE = Path("data/event_cache.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _now_utc():
    return datetime.now(timezone.utc)

def _load() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {"last_success_iso": None, "scores": {}, "failures": []}

def _save(obj: Dict[str, Any]):
    CACHE_FILE.write_text(json.dumps(obj, indent=2))

def save_scores(scores: Dict[str, float]):
    d = _load()
    ts = _now_utc().isoformat(timespec="seconds")
    for k, v in (scores or {}).items():
        d["scores"][k.upper()] = {"score": float(v), "ts_iso": ts}
    d["last_success_iso"] = ts
    _save(d)

def record_failure(err_type: str, message: str):
    d = _load()
    d["failures"].append({"ts_iso": _now_utc().isoformat(timespec="seconds"),
                          "type": err_type, "msg": message[:500]})
    # keep only last 50
    d["failures"] = d["failures"][-50:]
    _save(d)

def get_recent_scores(max_age_min: int) -> Tuple[Dict[str, float], bool]:
    d = _load()
    out = {}
    ok = False
    ttl = timedelta(minutes=int(max_age_min))
    for sym, rec in d.get("scores", {}).items():
        try:
            ts = datetime.fromisoformat(rec["ts_iso"])
        except Exception:
            continue
        if _now_utc() - ts <= ttl:
            out[sym] = float(rec["score"])
            ok = True
    return out, ok

def last_ok_within(max_age_min: int) -> bool:
    d = _load()
    ts = d.get("last_success_iso")
    if not ts: return False
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return False
    return (_now_utc() - t).total_seconds() <= 60*int(max_age_min)
