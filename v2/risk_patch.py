from __future__ import annotations
import importlib, os
from typing import Dict, Any

def _extract_weights(*args, **kwargs) -> Dict[str, float]:
    # Best-effort: look in kwargs then args for dict-like weights, else derive from orders.
    for k in ("weights","target_weights","targets"):
        w = kwargs.get(k)
        if isinstance(w, dict): return {str(s).upper(): float(v) for s,v in w.items()}
    for a in args:
        if isinstance(a, dict):  # likely weights
            try: return {str(s).upper(): float(v) for s,v in a.items()}
            except Exception: pass
    # derive from orders list
    orders = kwargs.get("orders") or None
    if orders is None:
        for a in args:
            if isinstance(a, list): orders = a
    W = {}
    try:
        for o in orders or []:
            sym = (o.get("symbol") or o.get("ticker") or o.get("asset") or "").upper()
            tgt = o.get("target_w") if "target_w" in o else o.get("target")
            cur = o.get("current_w") if "current_w" in o else o.get("current")
            if tgt is None and "notional" in o and "equity" in o:
                tgt = float(o["notional"]) / max(1.0, float(o["equity"]))
                cur = 0.0
            if sym:
                W[sym] = float(tgt if tgt is not None else 0.0)
        if W: return W
    except Exception:
        pass
    return {}

def _exposures(weights: Dict[str, float]) -> Dict[str, float]:
    gross_long = sum(max(0.0, w) for w in weights.values())
    gross_short = sum(max(0.0, -w) for w in weights.values())
    gross = gross_long + gross_short
    net = gross_long - gross_short
    return {"gross": gross, "gross_long": gross_long, "gross_short": gross_short, "net": net}

def apply():
    patched = 0
    try:
        agents = importlib.import_module("v2.agents")
    except Exception as e:
        print("[risk_patch] cannot import v2.agents:", e); return patched

    MAX_GROSS = float(os.getenv("MAX_GROSS_EXPOSURE","1.10"))
    MAX_LONG = float(os.getenv("MAX_LONG_EXPOSURE","1.05"))
    MAX_SHORT = float(os.getenv("MAX_SHORT_EXPOSURE","0.35"))

    for name, cls in list(vars(agents).items()):
        if isinstance(cls, type) and "risk" in name.lower():
            for mname, m in list(vars(cls).items()):
                if callable(m) and any(k in mname.lower() for k in ("approve","gate","check","limits")):
                    def make_wrapper(orig):
                        def wrapper(self, *a, **k):
                            res = orig(self, *a, **k)
                            try:
                                # normalize dict-like result
                                if isinstance(res, dict):
                                    W = _extract_weights(*a, **k)
                                    ex = _exposures(W)
                                    res["exposures"] = ex
                                    breach = (
                                        ex["gross"] > MAX_GROSS or
                                        ex["gross_long"] > MAX_LONG or
                                        ex["gross_short"] > MAX_SHORT
                                    )
                                    if breach:
                                        res["approved"] = False
                                        res.setdefault("reasons", []).append("risk_abs_exposure_breach")
                                        res["limits"] = {"MAX_GROSS": MAX_GROSS, "MAX_LONG": MAX_LONG, "MAX_SHORT": MAX_SHORT}
                            except Exception:
                                pass
                            return res
                        return wrapper
                    try:
                        setattr(cls, mname, make_wrapper(m))
                        patched += 1
                    except Exception:
                        pass
    print("[risk_patch] patched methods:", patched)
    return patched
