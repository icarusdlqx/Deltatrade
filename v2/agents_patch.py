"""
Hook into v2.agents to ensure all event scoring goes through llm_client.chat_json
and to record the timestamp of the last successful assessment.
"""
from __future__ import annotations
import importlib, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

LAST_FILE = Path("data/last_event_assessment.json")
LAST_FILE.parent.mkdir(parents=True, exist_ok=True)

def _save_last_ok():
    LAST_FILE.write_text(json.dumps({"ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds")}))

def _load_last_ok():
    if not LAST_FILE.exists(): return None
    try:
        return json.loads(LAST_FILE.read_text()).get("ts_iso")
    except Exception:
        return None

def _wrap_event_scorer(agents_mod) -> int:
    from .llm_client import chat_json
    patched = 0
    # look for likely call sites
    for name,obj in vars(agents_mod).items():
        # function style
        if callable(obj) and any(k in name.lower() for k in ("score","event","news")):
            def mk(fn: Callable):
                def f(*a, **k):
                    # force through our client if caller passes text/headlines
                    if "text" in k or (a and isinstance(a[0], str)):
                        user_text = k.get("text") or a[0]
                        res = chat_json(user_text)
                        _save_last_ok()
                        return res
                    return fn(*a, **k)
                return f
            try:
                setattr(agents_mod, name, mk(obj)); patched += 1
            except Exception:
                pass
        # class with method
        if isinstance(obj, type) and any(m for m in dir(obj) if "score" in m.lower()):
            try:
                meth = getattr(obj, [m for m in dir(obj) if "score" in m.lower()][0])
                def mk2(fn):
                    def f(self, *a, **k):
                        from .llm_client import chat_json
                        if "text" in k or (a and isinstance(a[0], str)):
                            user_text = k.get("text") or a[0]
                            res = chat_json(user_text)
                            _save_last_ok()
                            return res
                        return fn(self, *a, **k)
                    return f
                setattr(obj, meth.__name__, mk2(meth)); patched += 1
            except Exception:
                pass
    return patched

def last_ok_iso():
    return _load_last_ok()

def apply():
    try:
        agents = importlib.import_module("v2.agents")
    except Exception as e:
        print("[agents_patch] cannot import v2.agents:", e); return
    patched = _wrap_event_scorer(agents)
    print("[agents_patch] patched:", patched)
