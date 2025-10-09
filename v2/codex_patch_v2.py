from __future__ import annotations
import importlib, os
from types import ModuleType
from typing import Callable, List, Dict, Any

def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        print("[codex_patch_v2]", fn, "failed:", e)
        return None

def _wrap_plan_orders(execution: ModuleType) -> int:
    """Gate planning to 3x/day, classify actions, and log."""
    from .schedule_control import should_run_now
    from .actions import classify_actions
    from .journal import log_run, log_items

    target_fn_names = ("plan_orders", "build_orders", "make_orders", "plan")
    patched = 0

    def make_wrapper(orig: Callable):
        def wrapper(*args, **kwargs):
            # schedule gate
            dec = should_run_now(mark=True)
            if not dec.run_now:
                log_run(status="skipped", slot="", reason=dec.reason)
                return []
            # run analysis
            orders = orig(*args, **kwargs)
            try:
                orders = list(orders or [])
            except Exception:
                pass
            orders = classify_actions(orders)
            # summarize
            n_buy = sum(1 for o in orders if o.get("action") == "BUY")
            n_sell = sum(1 for o in orders if o.get("action") == "SELL")
            n_hold = sum(1 for o in orders if o.get("action") == "HOLD")
            log_run(status="analyzed", slot=dec.slot_label, reason="scheduled_slot",
                    turnover=_guess_turnover(orders), n_orders=len(orders),
                    n_buy=n_buy, n_sell=n_sell, n_hold=n_hold)
            # per-item logging
            _attach_expected_costs(orders)  # if available on order objects
            log_items(dec.slot_label, orders)
            return orders
        return wrapper

    for name in target_fn_names:
        o = getattr(execution, name, None)
        if callable(o):
            try:
                setattr(execution, name, make_wrapper(o))
                patched += 1
            except Exception:
                pass
    return patched

def _attach_expected_costs(orders: List[Dict[str, Any]]):
    for o in orders or []:
        for k in ("expected_bps","cost_bps","net_bps"):
            if k not in o:
                o[k] = None

def _guess_turnover(orders: List[Dict[str, Any]]):
    tw = 0.0
    for o in orders or []:
        try:
            tw += abs(float(o.get("delta_w") or 0.0))
        except Exception:
            pass
    return tw if tw > 0 else None

def _wrap_risk_officer(agents: ModuleType) -> int:
    """Compute net bps and approval, and log the decision."""
    from .journal import log_run
    safety = float(os.getenv("COST_SAFETY", "1.5"))
    min_net = float(os.getenv("MIN_NET_BPS", "2.0"))
    patched = 0
    for name, obj in vars(agents).items():
        if isinstance(obj, type) and "risk" in name.lower():
            for m_name, m in list(vars(obj).items()):
                if callable(m) and any(k in m_name.lower() for k in ("approve","gate","should")):
                    def make_wrapper(orig: Callable):
                        def wrapper(self, *args, **kwargs):
                            res = orig(self, *args, **kwargs)
                            try:
                                if isinstance(res, dict) and "expected_bps" in res and "cost_bps" in res:
                                    cost = float(res["cost_bps"]) * safety
                                    net = float(res.get("expected_bps", 0.0)) - cost
                                    res["net_bps"] = net
                                    res["approved"] = bool(net >= min_net)
                                    log_run(status="risk_checked", slot="", reason="",
                                            expected_bps=float(res.get("expected_bps") or 0.0),
                                            cost_bps=cost, net_bps=net, approved=res["approved"])
                            except Exception:
                                pass
                            return res
                        return wrapper
                    try:
                        setattr(obj, m_name, make_wrapper(m)); patched += 1
                    except Exception:
                        pass
    return patched

def apply():
    patched = {}
    try:
        execution = importlib.import_module("v2.execution")
        patched["execution.planner"] = _wrap_plan_orders(execution)
    except Exception as e:
        print("[codex_patch_v2] execution patch failed:", e)
    try:
        agents = importlib.import_module("v2.agents")
        patched["agents.risk"] = _wrap_risk_officer(agents)
    except Exception as e:
        print("[codex_patch_v2] agents patch failed:", e)
    print("[codex_patch_v2] summary:", patched)
