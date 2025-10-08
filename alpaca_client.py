from __future__ import annotations

import os
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def _client() -> TradingClient:
    api = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_V3")
    sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY_V3")
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    return TradingClient(api_key=api, secret_key=sec, paper=paper)


def get_clock() -> Dict[str, Any]:
    c = _client().get_clock()
    return {
        "is_open": bool(getattr(c, "is_open", False)),
        "next_open": str(getattr(c, "next_open", "")),
        "next_close": str(getattr(c, "next_close", "")),
    }


def list_positions() -> List[Dict[str, Any]]:
    client = _client()
    positions = client.get_all_positions()
    out: List[Dict[str, Any]] = []
    for pos in positions:
        qty_raw = getattr(pos, "qty", getattr(pos, "quantity", "0"))
        qty_available = getattr(pos, "qty_available", qty_raw)
        market_value = float(getattr(pos, "market_value", 0.0) or 0.0)
        current_price = getattr(pos, "current_price", None)
        if current_price is None and qty_raw not in ("0", 0, None):
            try:
                qty_f = float(qty_raw)
                current_price = market_value / qty_f if qty_f else 0.0
            except Exception:
                current_price = 0.0
        out.append(
            {
                "symbol": getattr(pos, "symbol", ""),
                "qty": qty_raw,
                "qty_available": qty_available,
                "avg_entry_price": float(getattr(pos, "avg_entry_price", 0.0) or 0.0),
                "market_value": market_value,
                "unrealized_pl": float(getattr(pos, "unrealized_pl", 0.0) or 0.0),
                "asset_class": getattr(pos, "asset_class", "us_equity"),
                "current_price": float(current_price or 0.0),
            }
        )
    return out


def get_account() -> Dict[str, Any]:
    acct = _client().get_account()
    return {
        "cash": float(getattr(acct, "cash", 0.0) or 0.0),
        "equity": float(getattr(acct, "equity", 0.0) or getattr(acct, "portfolio_value", 0.0) or 0.0),
        "buying_power": float(getattr(acct, "buying_power", 0.0) or 0.0),
    }


def _round_down_micro(qty_str: str) -> str:
    qty_decimal = Decimal(str(qty_str or "0"))
    return str(qty_decimal.quantize(Decimal("0.000001"), rounding=ROUND_DOWN))


def close_position(symbol: str) -> Dict[str, Any]:
    client = _client()
    try:
        order = client.close_position(symbol)
        return {
            "ok": True,
            "method": "close_position",
            "order_id": getattr(order, "id", None),
        }
    except APIError as err:
        primary_error = str(err)
        try:
            positions = {p["symbol"]: p for p in list_positions()}
            if symbol not in positions:
                return {"ok": False, "method": "close_position", "error": "no_position"}
            qty_raw = positions[symbol]["qty_available"]
            qty = _round_down_micro(qty_raw)
            if Decimal(qty) <= 0:
                return {"ok": False, "method": "close_position", "error": "zero_available"}
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            return {
                "ok": True,
                "method": "market_qty",
                "order_id": getattr(order, "id", None),
                "qty": qty,
                "fallback_error": primary_error,
            }
        except Exception as fallback_exc:
            return {
                "ok": False,
                "method": "market_qty",
                "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                "fallback_error": primary_error,
            }
    except Exception as exc:
        return {
            "ok": False,
            "method": "close_position",
            "error": f"{type(exc).__name__}: {exc}",
        }


def close_all_positions(cancel_open_orders: bool = True) -> Dict[str, Any]:
    client = _client()
    try:
        response = client.close_all_positions(cancel_orders=cancel_open_orders)
        normalized = []
        for item in response:
            symbol = getattr(item, "symbol", None)
            status = getattr(item, "status", None)
            if symbol is None and isinstance(item, dict):
                symbol = item.get("symbol")
                status = item.get("status")
            normalized.append({"symbol": symbol, "status": str(status or "")})
        return {"ok": True, "method": "close_all", "status": normalized}
    except Exception as exc:
        return {
            "ok": False,
            "method": "close_all",
            "error": f"{type(exc).__name__}: {exc}",
        }
