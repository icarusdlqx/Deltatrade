"""
Lightweight, conservative pre-trade cost model for very liquid ETFs/equities.
Replaces overly high defaults with symbol-aware spreads and concave impact.
Returns cost in bps vs. arrival mid.
"""

from __future__ import annotations
import os
from typing import Any, Dict

# Tiering is intentionally simple & conservative.
TIER1 = {"SPY","IVV","VOO","QQQ","QQQM"}
TIER2 = {"EEM","IEMG","IEFA","EFA","VTI","ITOT","ACWX","IWM","RSP","MDY","SOXX","SMH","XME"}

_PRESETS = {
    "TIER1":  dict(spread_bps=1.0, impact_a=12.0, impact_power=0.5, passive_fill=0.4, fee_bps=0.0),
    "TIER2":  dict(spread_bps=2.0, impact_a=18.0, impact_power=0.5, passive_fill=0.5, fee_bps=0.0),
    "DEFAULT":dict(spread_bps=5.0, impact_a=35.0, impact_power=0.6, passive_fill=0.6, fee_bps=0.0),
}

def _params_for(symbol: str) -> Dict[str, float]:
    s = (symbol or "").upper()
    if s in TIER1:
        return {**_PRESETS["TIER1"]}
    if s in TIER2:
        return {**_PRESETS["TIER2"]}
    return {**_PRESETS["DEFAULT"]}

def estimate_pretrade_cost_bps(*args: Any, **kwargs: Any) -> float:
    """
    Best-effort estimator. Accepts flexible signatures and ignores extra kwargs.
    Inputs (any subset):
      - symbol: str
      - qty / quantity / shares: number of shares
      - notional: dollar value of the order
      - price: price to convert notional -> shares
      - adv / average_daily_volume: ADV in shares (optional)
      - participation: 0..1 (optional; default 0.1)
      - spread_bps: override known spread (optional)
    Returns: float cost in basis points.
    """
    symbol = kwargs.get("symbol") or (args[0] if args and isinstance(args[0], str) else "")
    qty = kwargs.get("qty") or kwargs.get("quantity") or kwargs.get("shares") or 0.0
    notional = kwargs.get("notional") or 0.0
    price = kwargs.get("price") or kwargs.get("last") or kwargs.get("mid") or 0.0
    adv = kwargs.get("adv") or kwargs.get("average_daily_volume") or 0.0
    participation = float(kwargs.get("participation") or 0.1)

    if not qty and notional and price:
        try:
            qty = float(notional) / float(price)
        except Exception:
            qty = 0.0

    p = _params_for(symbol)
    spread_bps = float(kwargs.get("spread_bps") or p["spread_bps"])
    passive_fill = float(p["passive_fill"])
    fee_bps = float(p["fee_bps"])
    a = float(p["impact_a"])
    r = float(p["impact_power"])

    # If ADV is unknown, use a tiny proxy from participation; still conservative.
    try:
        pct_adv = max(1e-9, min(1.0, float(qty) / float(adv))) if adv else max(1e-9, min(1.0, participation * 0.1))
    except Exception:
        pct_adv = 1e-4

    half_spread_cost = passive_fill * spread_bps
    impact_bps = a * (pct_adv ** r)
    total_bps = half_spread_cost + impact_bps + fee_bps
    cap = float(os.getenv("COST_BPS_CAP", "50"))
    return float(min(total_bps, cap))
