"""
Hook into v2.agents to ensure all event scoring goes through llm_client.chat_json
and to record the timestamp of the last successful assessment.
"""
from __future__ import annotations
import importlib
from typing import Callable

from .event_gate import record_assessment, load_last_assessment_iso

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
                        record_assessment()
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
                            record_assessment()
                            return res
                        return fn(self, *a, **k)
                    return f
                setattr(obj, meth.__name__, mk2(meth)); patched += 1
            except Exception:
                pass
    return patched

def last_ok_iso():
    return load_last_assessment_iso()

def apply():
    try:
        agents = importlib.import_module("v2.agents")
    except Exception as e:
        print("[agents_patch] cannot import v2.agents:", e); return
    patched = _wrap_event_scorer(agents)
    print("[agents_patch] patched:", patched)
