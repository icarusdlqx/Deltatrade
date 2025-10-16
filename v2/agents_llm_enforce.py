from __future__ import annotations

import importlib
from typing import Any, Dict

from .llm_gpt5 import LLMError, score_events_gpt5


def apply_patch() -> int:
    patched = 0

    try:
        agents = importlib.import_module("v2.agents")
    except Exception as exc:  # pragma: no cover - import edge
        print("[llm_enforce] cannot import v2.agents:", exc)
        agents = None

    if agents:
        def _wrap(fn):
            def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                headlines_by_symbol = kwargs.get("headlines_by_symbol")
                if headlines_by_symbol is None and args and isinstance(args[0], dict):
                    headlines_by_symbol = args[0]

                try:
                    return score_events_gpt5(headlines_by_symbol or {})
                except LLMError as exc:
                    print("[llm_enforce] LLM call failed:", exc.kind, str(exc))
                    return {
                        "scores": {},
                        "raw": "",
                        "usage": {},
                        "model": "",
                        "source": f"error:{exc.kind}",
                    }

            return wrapped

        for name, obj in list(vars(agents).items()):
            if callable(obj) and any(token in name.lower() for token in ["event", "news", "score"]):
                try:
                    setattr(agents, name, _wrap(obj))
                    patched += 1
                except Exception:
                    pass

    try:
        orchestrator = importlib.import_module("v2.orchestrator")
        for attr in ("run_once", "run", "cycle", "single_cycle"):
            func = getattr(orchestrator, attr, None)
            if callable(func):
                def make_wrapper(original):
                    def inner(*args: Any, **kwargs: Any):
                        try:
                            score_events_gpt5({"KEEPALIVE": ["No material headlines this window."]})
                        except Exception:
                            pass
                        return original(*args, **kwargs)

                    return inner

                setattr(orchestrator, attr, make_wrapper(func))
                patched += 1
                break
    except Exception as exc:  # pragma: no cover - import edge
        print("[llm_enforce] orchestrator patch skipped:", exc)

    print("[llm_enforce] patched:", patched)
    return patched

