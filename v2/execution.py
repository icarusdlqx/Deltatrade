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
    if abs(delta_notional) <= 1e-6:
        return 0.0
    if price <= 0 or adv <= 0:
        return abs(spread_bps)
    qty = abs(delta_notional) / price
    impact = kappa * (qty / adv) ** psi * 10000.0
    return abs(spread_bps) + impact

def _target_slice_notional(delta: float, adv_notional: float,
                           spread_bps: float, kappa: float, psi: float,
                           min_notional: float) -> float:
    """Infer a per-slice notional that keeps impact near the spread cost."""
    adv_notional = max(0.0, float(adv_notional))
    if adv_notional <= 0 or kappa <= 0 or psi <= 0:
        return max(min_notional, abs(delta))

    # Solve kappa * (slice/adv) ** psi * 10000 ~= spread_bps for slice
    try:
        participation = (abs(spread_bps) / (kappa * 10000.0)) ** (1.0 / psi)
    except ZeroDivisionError:
        participation = 0.0
    # Clamp participation to a sensible band so we do not spray tiny orders
    participation = max(0.02, min(0.25, participation))
    slice_notional = adv_notional * participation
    return max(min_notional, slice_notional)


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
        adv_notional = adv.get(s, 0.0)
        slice_notional = _target_slice_notional(delta, adv_notional, spread_bps, kappa, psi, min_notional)
        slices = max(1, int(math.ceil(abs(delta) / max(slice_notional, min_notional))))
        slices = min(slices, max_slices)
        # Keep a sensible limit anchor so downstream execution can respect cost inputs
        spread_fraction = abs(spread_bps) / 20000.0
        limit = px * (1 + spread_fraction) if side == "buy" else px * (1 - spread_fraction)
        plans.append(OrderPlan(symbol=s, side=side, qty=qty_total, limit_price=limit, slices=slices))
    return plans


def _asset_is_fractionable(client: TradingClient, symbol: str, cache: Dict[str, bool]) -> bool:
    if symbol in cache:
        return cache[symbol]
    getter = getattr(client, "get_asset", None)
    if getter is None:
        cache[symbol] = False
        return False
    try:
        asset = getter(symbol)
        frac = bool(getattr(asset, "fractionable", True))
    except Exception:
        frac = True
    cache[symbol] = frac
    return frac


def _round_order_qty(qty: float, fractionable: bool) -> float:
    qty = float(qty)
    if qty <= 0:
        return 0.0
    if fractionable:
        return max(0.0001, round(qty, 4))
    return float(max(0, math.floor(qty)))

def place_orders_with_limits(client: TradingClient, plans: List[OrderPlan],
                             last_prices: Dict[str,float], peg_bps: int,
                             tif: TimeInForce = TimeInForce.DAY, fill_timeout_sec: int = 20) -> List[str]:
    order_ids: List[str] = []
    fractionable_cache: Dict[str, bool] = {}
    for p in plans:
        lp = last_prices[p.symbol]
        bump = (peg_bps/10000.0) * lp
        limit_anchor = p.limit_price if p.limit_price is not None else lp
        limit = (limit_anchor + bump) if p.side == "buy" else (limit_anchor - bump)
        fractionable = _asset_is_fractionable(client, p.symbol, fractionable_cache)

        qty_total = max(0.0, p.qty)
        if qty_total <= 0:
            continue
        slices = max(1, p.slices)
        qty_remaining = qty_total

        for i in range(slices):
            if qty_remaining <= 0:
                break
            slices_left = max(1, slices - i)
            qty_target = qty_remaining / slices_left
            qty_slice = _round_order_qty(qty_target, fractionable)
            if qty_slice <= 0:
                if not fractionable:
                    break
                continue
            try:
                req = LimitOrderRequest(
                    symbol=p.symbol,
                    qty=qty_slice,
                    side=OrderSide.BUY if p.side == "buy" else OrderSide.SELL,
                    time_in_force=tif,
                    limit_price=round(limit, 4)
                )
                o = client.submit_order(req)
                order_ids.append(o.id)
                filled_qty = _await_limit_fill(client, o.id, qty_slice, fill_timeout_sec)
                qty_filled = min(qty_slice, filled_qty)
                qty_remaining = max(0.0, qty_remaining - qty_filled)
                remaining = max(0.0, qty_slice - qty_filled)
                if remaining > 1e-4:
                    _cancel_if_possible(client, o.id)
                    fallback = _submit_market(client, p.symbol, remaining, p.side, tif, fractionable, lp)
                    if fallback:
                        order_ids.append(fallback)
                        qty_remaining = max(0.0, qty_remaining - remaining)
            except Exception:
                fallback = _submit_market(client, p.symbol, qty_slice, p.side, tif, fractionable, lp)
                if fallback:
                    order_ids.append(fallback)
                    qty_remaining = max(0.0, qty_remaining - qty_slice)
            time.sleep(0.25)
    return order_ids


def _submit_market(client: TradingClient, symbol: str, qty: float, side: str,
                   tif: TimeInForce, fractionable: bool, last_price: float) -> str | None:
    side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
    if fractionable and last_price and last_price > 0:
        notional = round(max(1.0, last_price * qty), 2)
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side_enum,
                time_in_force=tif,
            )
            o = client.submit_order(req)
            return o.id
        except Exception:
            pass
    qty_adj = _round_order_qty(qty, fractionable)
    if qty_adj <= 0:
        return None
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty_adj,
            side=side_enum,
            time_in_force=tif,
        )
        o = client.submit_order(req)
        return o.id
    except Exception:
        return None


def _cancel_if_possible(client: TradingClient, order_id: str) -> None:
    cancel = getattr(client, "cancel_order_by_id", None)
    if cancel is None:
        cancel = getattr(client, "cancel_order", None)
    if cancel is None:
        return
    try:
        cancel(order_id)
    except Exception:
        return


def _await_limit_fill(client: TradingClient, order_id: str, qty: float, timeout: int) -> float:
    if timeout <= 0:
        return 0.0
    getter = getattr(client, "get_order_by_id", None)
    if getter is None:
        getter = getattr(client, "get_order", None)
    if getter is None:
        return 0.0

    deadline = time.time() + timeout
    latest_filled = 0.0
    while time.time() < deadline:
        try:
            order = getter(order_id)
        except Exception:
            break
        status = str(getattr(order, "status", "")).lower()
        filled_qty = float(getattr(order, "filled_qty", latest_filled) or latest_filled)
        latest_filled = max(latest_filled, filled_qty)
        if status in ("filled", "done_for_day"):
            return qty if filled_qty <= 0 else filled_qty
        if status in ("canceled", "expired", "stopped", "rejected"):
            return filled_qty
        if status == "partially_filled" and filled_qty >= qty - 1e-4:
            return filled_qty
        time.sleep(1.0)
    return latest_filled
