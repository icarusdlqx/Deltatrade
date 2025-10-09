from __future__ import annotations
import os
from typing import List, Dict, Any
from .journal import last_actions_map

def classify_actions(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds 'action' to each order: BUY / SELL / HOLD.
    Uses two bands:
     - ACTION_BAND_BPS: below this |Δw| -> HOLD
     - ACTION_SWITCH_BPS: if last action was opposite, require larger |Δw| to flip
    """
    band = float(os.getenv("ACTION_BAND_BPS", "15")) / 10000.0
    switch = float(os.getenv("ACTION_SWITCH_BPS", "30")) / 10000.0
    prior = last_actions_map()
    out = []
    for o in orders or []:
        sym = o.get("symbol") or o.get("ticker") or o.get("asset")
        cur = _as_float(o.get("current")) or _as_float(o.get("current_w"))
        tgt = _as_float(o.get("target")) or _as_float(o.get("target_w"))
        if cur is None or tgt is None:
            # Try derive from notional/equity
            if "notional" in o and "equity" in o:
                tgt = float(o["notional"]) / max(1.0, float(o["equity"]))
                cur = 0.0
        dw = (tgt or 0.0) - (cur or 0.0)
        action = "HOLD"
        if abs(dw) >= band:
            action = "BUY" if dw > 0 else "SELL"
        last = prior.get(sym)
        if last and last != action and abs(dw) < switch:
            action = "HOLD"
        o["delta_w"] = dw
        o["target_w"] = tgt
        o["current_w"] = cur
        o["action"] = action
        out.append(o)
    return out

def _as_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None
