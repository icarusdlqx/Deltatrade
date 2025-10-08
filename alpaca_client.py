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
    clock = _client().get_clock()
    return {
        "is_open": bool(getattr(clock, "is_open", False)),
        "next_open": str(getattr(clock, "next_open", "")),
        "next_close": str(getattr(clock, "next_close", "")),
    }


def list_positions() -> List[Dict[str, Any]]:
    client = _client()
    positions = client.get_all_positions()
    out: List[Dict[str, Any]] = []
    for pos in positions:
        out.append(
            {
                "symbol": pos.symbol,
                "qty": pos.qty,
                "qty_available": getattr(pos, "qty_available", pos.qty),
                "avg_entry_price": float(getattr(pos, "avg_entry_price", 0) or 0),
                "market_value": float(getattr(pos, "market_value", 0) or 0),
                "unrealized_pl": float(getattr(pos, "unrealized_pl", 0) or 0),
                "current_price": float(getattr(pos, "current_price", 0) or 0),
                "asset_class": getattr(pos, "asset_class", "us_equity"),
            }
        )
    return out


def get_account() -> Dict[str, Any]:
    acct = _client().get_account()
    return {
        "cash": float(getattr(acct, "cash", 0) or 0),
        "equity": float(getattr(acct, "equity", 0) or 0),
        "buying_power": float(getattr(acct, "buying_power", 0) or 0),
    }


def _round_down_micro(qty_str: str) -> str:
    quantity = Decimal(qty_str)
    return str(quantity.quantize(Decimal("0.000001"), rounding=ROUND_DOWN))


def close_position(symbol: str) -> Dict[str, Any]:
    client = _client()
    try:
        order = client.close_position(symbol)
        return {
            "ok": True,
            "method": "close_position",
            "order_id": getattr(order, "id", None),
        }
    except APIError as exc:
        error_message = str(exc)
        try:
            positions = {p["symbol"]: p for p in list_positions()}
            if symbol not in positions:
                return {"ok": False, "method": "close_position", "error": "no_position"}
            available_raw = positions[symbol]["qty_available"]
            qty = _round_down_micro(available_raw)
            if Decimal(qty) <= 0:
                return {"ok": False, "method": "close_position", "error": "zero_available"}
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(request)
            return {
                "ok": True,
                "method": "market_qty",
                "order_id": getattr(order, "id", None),
                "qty": qty,
                "fallback_error": error_message,
            }
        except Exception as fallback_exc:
            return {
                "ok": False,
                "method": "market_qty",
                "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                "fallback_error": error_message,
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
        status: List[Dict[str, Any]] = []
        for item in response:
            if isinstance(item, dict):
                status.append(
                    {"symbol": item.get("symbol"), "status": str(item.get("status", ""))}
                )
            else:
                status.append(
                    {
                        "symbol": getattr(item, "symbol", None),
                        "status": str(getattr(item, "status", "")),
                    }
                )
        return {"ok": True, "method": "close_all", "status": status}
    except Exception as exc:
        return {
            "ok": False,
            "method": "close_all",
            "error": f"{type(exc).__name__}: {exc}",
        }
