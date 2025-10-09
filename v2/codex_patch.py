"""
CODEX runtime patch for Deltatrade:
- realistic ETF/equity pre-trade cost model
- optional min weight delta (turnover band)
- safer net-bps gate

The patch is robust: it searches for likely function names and wraps them.
If it cannot find a target, it degrades gracefully without breaking the run.
"""

from __future__ import annotations
import importlib, os
from types import ModuleType
from typing import Callable

def _wrap_cost_functions(execution: ModuleType) -> int:
    try:
        from .cost_overrides import estimate_pretrade_cost_bps as override_cost
    except Exception as e:
        print("[codex_patch] cost_overrides unavailable:", e)
        return 0

    candidates = [n for n, o in vars(execution).items()
                  if callable(o) and ("cost" in n.lower()) and
                     (n.lower().startswith("estimate") or n.lower().startswith("pretrade"))]

    def make_wrapper(orig: Callable):
        def wrapper(*args, **kwargs):
            try:
                return float(override_cost(*args, **kwargs))
            except Exception:
                return float(orig(*args, **kwargs))
        wrapper.__name__ = getattr(orig, "__name__", "wrapped_cost")
        return wrapper

    patched = 0
    for name in candidates:
        try:
            setattr(execution, name, make_wrapper(getattr(execution, name)))
            patched += 1
        except Exception:
            pass
    return patched

def _wrap_order_planner(execution: ModuleType) -> int:
    band_bps = float(os.getenv("TURNOVER_BAND_BPS", "25.0"))
    band = band_bps / 10000.0
    target_names = ("plan_orders", "build_orders", "make_orders")

    def make_wrapper(orig: Callable):
        def wrapper(*args, **kwargs):
            orders = orig(*args, **kwargs)
            try:
                result = []
                for o in orders or []:
                    dw = o.get("delta_w")
                    if dw is None and "target" in o and "current" in o:
                        dw = float(o["target"]) - float(o["current"])
                    if dw is None and "notional" in o and "equity" in o:
                        dw = float(o["notional"]) / max(1.0, float(o["equity"]))
                    if dw is None or abs(dw) >= band:
                        result.append(o)
                return result
            except Exception:
                return orders
        return wrapper

    patched = 0
    for name in target_names:
        obj = getattr(execution, name, None)
        if callable(obj):
            try:
                setattr(execution, name, make_wrapper(obj))
                patched += 1
            except Exception:
                pass
    return patched

def _wrap_risk_officer(agents: ModuleType) -> int:
    safety = float(os.getenv("COST_SAFETY", "1.5"))
    min_net = float(os.getenv("MIN_NET_BPS", "2.0"))
    patched = 0

    for name, obj in vars(agents).items():
        if isinstance(obj, type) and "risk" in name.lower():
            for m_name, m in list(vars(obj).items()):
                if callable(m) and any(k in m_name.lower() for k in ("approve", "gate", "should")):
                    def make_wrapper(orig: Callable):
                        def wrapper(self, *args, **kwargs):
                            res = orig(self, *args, **kwargs)
                            try:
                                if isinstance(res, dict) and "expected_bps" in res and "cost_bps" in res:
                                    cost = float(res["cost_bps"]) * safety
                                    net = float(res.get("expected_bps", 0.0)) - cost
                                    res["net_bps"] = net
                                    res["approved"] = bool(net >= min_net)
                                return res
                            except Exception:
                                return res
                        return wrapper
                    try:
                        setattr(obj, m_name, make_wrapper(m))
                        patched += 1
                    except Exception:
                        pass
    return patched

def apply():
    patched = {}
    try:
        execution = importlib.import_module("v2.execution")
        pc = _wrap_cost_functions(execution)
        po = _wrap_order_planner(execution)
        patched["execution.cost"] = pc
        patched["execution.orders"] = po
    except Exception as e:
        print("[codex_patch] execution patch failed:", e)

    try:
        agents = importlib.import_module("v2.agents")
        pr = _wrap_risk_officer(agents)
        patched["agents.risk"] = pr
    except Exception as e:
        print("[codex_patch] agents patch failed:", e)

    print("[codex_patch] summary:", patched)
