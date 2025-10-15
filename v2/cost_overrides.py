from __future__ import annotations
import os
from typing import Any, Dict
from .slippage_logger import get_symbol_slippage_bps

TIER1 = {"SPY","IVV","VOO","QQQ","QQQM"}
TIER2 = {"EEM","IEMG","IEFA","EFA","VTI","ITOT","ACWX","IWM","RSP","MDY","SOXX","SMH","XME"}

_PRESETS = {
    "TIER1":  dict(spread_bps=1.0, impact_a=12.0, impact_power=0.5, passive_fill=0.4, fee_bps=0.0),
    "TIER2":  dict(spread_bps=2.0, impact_a=18.0, impact_power=0.5, passive_fill=0.5, fee_bps=0.0),
    "DEFAULT":dict(spread_bps=5.0, impact_a=35.0, impact_power=0.6, passive_fill=0.6, fee_bps=0.0),
}

def _params_for(symbol: str) -> Dict[str, float]:
    s = (symbol or "").upper()
    if s in TIER1: return {**_PRESETS["TIER1"]}
    if s in TIER2: return {**_PRESETS["TIER2"]}
    return {**_PRESETS["DEFAULT"]}

def estimate_pretrade_cost_bps(*args: Any, **kwargs: Any) -> float:
    """
    Flexible signature; returns estimated cost in bps vs arrival mid.
    Inputs may include symbol, qty, notional, price, adv, participation, spread_bps.
    """
    symbol = kwargs.get("symbol") or (args[0] if args and isinstance(args[0], str) else "")
    qty = kwargs.get("qty") or kwargs.get("quantity") or kwargs.get("shares") or 0.0
    notional = kwargs.get("notional") or 0.0
    price = kwargs.get("price") or kwargs.get("last") or kwargs.get("mid") or 0.0
    adv = kwargs.get("adv") or kwargs.get("average_daily_volume") or 0.0
    participation = float(kwargs.get("participation") or 0.1)

    if not qty and notional and price:
        try: qty = float(notional) / float(price)
        except Exception: qty = 0.0

    p = _params_for(symbol)
    spread_bps = float(kwargs.get("spread_bps") or p["spread_bps"])
    passive_fill = float(p["passive_fill"])
    fee_bps = float(p["fee_bps"])
    a = float(p["impact_a"]); r = float(p["impact_power"])

    try:
        pct_adv = max(1e-9, min(1.0, float(qty) / float(adv))) if adv else max(1e-9, min(1.0, participation * 0.1))
    except Exception:
        pct_adv = 1e-4

    base_half_spread = passive_fill * spread_bps
    base_impact = a * (pct_adv ** r)
    base_total = base_half_spread + base_impact + fee_bps

    # Blend with realized slippage EMA if available
    meas = get_symbol_slippage_bps(symbol) or None
    w = float(os.getenv("DYNCOST_BLEND","0.5"))
    if meas is not None:
        blended = (1 - w) * base_total + w * float(meas)
    else:
        blended = base_total

    cap = float(os.getenv("COST_BPS_CAP","50"))
    return float(min(blended, cap))
