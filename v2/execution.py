from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math, time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

@dataclass
class OrderPlan:
    symbol: str
    side: str       # "buy" or "sell"
    qty: float
    limit_price: float | None
    slices: int

def estimate_cost_bps(delta_notional: float, price: float, adv: float,
                      spread_bps: float, kappa: float, psi: float) -> float:
    if price <= 0 or adv <= 0:
        return abs(spread_bps)
    qty = abs(delta_notional) / price
    impact = kappa * (qty / adv) ** psi * 10000.0
    return abs(spread_bps) + impact

def build_order_plans(targets: Dict[str,float], current_mv: Dict[str,float],
                      prices: Dict[str,float], adv: Dict[str,float],
                      min_notional: float, max_slices: int,
                      spread_bps: float, kappa: float, psi: float) -> List[OrderPlan]:
    plans: List[OrderPlan] = []
    for s, tgt in targets.items():
        cur = current_mv.get(s, 0.0)
        delta = tgt - cur
        if abs(delta) < min_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        px = prices.get(s, 0.0)
        if px <= 0:
            continue
        qty_total = abs(delta) / px
        slices = max(1, min(max_slices, int(math.ceil(qty_total / max(1.0, qty_total / max_slices)))))
        plans.append(OrderPlan(symbol=s, side=side, qty=qty_total, limit_price=None, slices=slices))
    return plans

def place_orders_with_limits(client: TradingClient, plans: List[OrderPlan],
                             last_prices: Dict[str,float], peg_bps: int,
                             tif: TimeInForce = TimeInForce.DAY, fill_timeout_sec: int = 20) -> List[str]:
    order_ids: List[str] = []
    for p in plans:
        lp = last_prices[p.symbol]
        bump = (peg_bps/10000.0) * lp
        limit = (lp + bump) if p.side == "buy" else (lp - bump)
        qty_per = max(0.0001, p.qty / max(1, p.slices))
        qround = lambda q: max(0.0001, round(q, 4))
        limround = lambda x: round(x, 4)

        for _ in range(p.slices):
            try:
                req = LimitOrderRequest(
                    symbol=p.symbol,
                    qty=qround(qty_per),
                    side=OrderSide.BUY if p.side == "buy" else OrderSide.SELL,
                    time_in_force=tif,
                    limit_price=limround(limit)
                )
                o = client.submit_order(req)
                order_ids.append(o.id); time.sleep(0.5)
            except Exception:
                try:
                    req = MarketOrderRequest(
                        symbol=p.symbol,
                        qty=qround(qty_per),
                        side=OrderSide.BUY if p.side == "buy" else OrderSide.SELL,
                        time_in_force=tif,
                    )
                    o = client.submit_order(req)
                    order_ids.append(o.id); time.sleep(0.5)
                except Exception:
                    continue
    return order_ids
