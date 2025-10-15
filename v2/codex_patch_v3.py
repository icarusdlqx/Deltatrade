from __future__ import annotations
import importlib, os
from typing import Callable
from .agents_llm_bridge import apply_patch_into_agents, require_recent_assessment
from .slippage_logger import record_fill
from .cost_overrides import estimate_pretrade_cost_bps

def _wrap_cost_functions(execution):
    patched = 0
    for name, fn in list(vars(execution).items()):
        if callable(fn) and "cost" in name.lower():
            def mk(orig):
                def f(*a, **k):
                    try:
                        return float(estimate_pretrade_cost_bps(*a, **k))
                    except Exception:
                        return orig(*a, **k)
                return f
            try:
                setattr(execution, name, mk(fn)); patched += 1
            except Exception:
                pass
    print("[codex_v3] cost fns patched:", patched)
    return patched

def _wrap_order_execution(execution):
    """
    Intercept common fill handlers to log realized slippage.
    """
    patched = 0
    cand = [n for n,o in vars(execution).items()
            if callable(o) and any(k in n.lower() for k in ("on_fill","handle_fill","record_fill","process_fill"))]
    for name in cand:
        old = getattr(execution, name)
        def mk(orig):
            def f(*a, **k):
                try:
                    # Try to grab useful fields from kwargs or args
                    sym = k.get("symbol") or (a[0] if a and isinstance(a[0], str) else None)
                    fill_price = k.get("fill_price") or k.get("price")
                    arrival = k.get("arrival_mid") or k.get("mid") or k.get("arrival")
                    side = k.get("side") or "NA"; qty = k.get("qty") or k.get("quantity") or 0
                    record_fill(sym, side, qty, arrival, fill_price, k.get("adv") or 0.0)
                except Exception:
                    pass
                return orig(*a, **k)
            return f
        try:
            setattr(execution, name, mk(old)); patched += 1
        except Exception:
            pass
    print("[codex_v3] fill hooks patched:", patched)
    return patched

def _wrap_risk_gate(agents):
    """
    Ensure net-edge gating uses symbol-aware costs (already handled upstream),
    and honor REQUIRE_EVENT_ASSESSMENT if enabled.
    """
    safety = float(os.getenv("COST_SAFETY","1.5"))
    min_net = float(os.getenv("MIN_NET_BPS","2.0"))
    patched = 0
    for name, cls in list(vars(agents).items()):
        if isinstance(cls, type) and "risk" in name.lower():
            for m, fn in list(vars(cls).items()):
                if callable(fn) and any(k in m.lower() for k in ("approve","gate","should")):
                    def mk(orig):
                        def f(self, *a, **k):
                            # Optional guard: require recent event assessment
                            try:
                                if not require_recent_assessment():
                                    return {"approved": False, "reasons": ["no_recent_event_assessment"]}
                            except Exception:
                                pass
                            res = orig(self, *a, **k)
                            try:
                                if isinstance(res, dict) and "expected_bps" in res and "cost_bps" in res:
                                    cost = float(res.get("cost_bps") or 0.0) * safety
                                    net = float(res.get("expected_bps") or 0.0) - cost
                                    res["net_bps"] = net
                                    res["approved"] = bool(net >= min_net)
                            except Exception:
                                pass
                            return res
                        return f
                    try:
                        setattr(cls, m, mk(fn)); patched += 1
                    except Exception:
                        pass
    print("[codex_v3] risk gate patched:", patched)
    return patched

def apply():
    # Patch agents (LLM bridge) + risk absolute exposures (separate module)
    apply_patch_into_agents()
    try:
        import v2.risk_patch as rp
        rp.apply()
    except Exception as e:
        print("[codex_v3] risk_abs patch failed:", e)

    # Patch execution (cost functions + fill logging)
    try:
        execution = importlib.import_module("v2.execution")
        _wrap_cost_functions(execution)
        _wrap_order_execution(execution)
    except Exception as e:
        print("[codex_v3] execution patch failed:", e)

    # Patch risk gate to use net edge with safety/min nets and optional guard
    try:
        agents = importlib.import_module("v2.agents")
        _wrap_risk_gate(agents)
    except Exception as e:
        print("[codex_v3] risk_gate patch failed:", e)

    print("[codex_v3] patch applied")

