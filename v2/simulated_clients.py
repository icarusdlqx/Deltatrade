from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SIM_STATE_PATH = Path("data/sim_trading_state.json")


def _ensure_data_dir() -> None:
    SIM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _default_symbols() -> List[str]:
    return [
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLY", "XLP",
        "XLE", "XLV", "XLI", "SMH", "SOXX", "TLT", "HYG"
    ]


class SimStockHistoricalDataClient:
    """Deterministic bar generator that mimics Alpaca's historical client."""

    def __init__(self, seed: int = 7, symbols: Iterable[str] | None = None) -> None:
        rng = np.random.default_rng(seed)
        self._symbols = list(symbols) if symbols is not None else _default_symbols()
        self._bars: Dict[str, pd.DataFrame] = {}
        horizon = 420
        end = datetime.now(timezone.utc)
        idx = pd.bdate_range(end=end, periods=horizon, tz=timezone.utc)
        base_levels = np.linspace(80, 320, num=len(self._symbols))
        for i, sym in enumerate(self._symbols):
            drift = 0.0003 * (i + 1)
            shocks = rng.normal(loc=drift, scale=0.012, size=len(idx))
            prices = 100.0 + base_levels[i] * 0.01
            closes = [prices]
            for shock in shocks:
                prices = max(5.0, prices * (1 + shock))
                closes.append(prices)
            closes = closes[1:]
            closes = np.array(closes)
            highs = closes * (1 + rng.uniform(0.001, 0.01, size=len(closes)))
            lows = closes * (1 - rng.uniform(0.001, 0.01, size=len(closes)))
            opens = closes / (1 + rng.uniform(-0.003, 0.003, size=len(closes)))
            volume = rng.integers(500_000, 5_000_000, size=len(closes))
            df = pd.DataFrame({
                "open": opens,
                "high": np.maximum.reduce([opens, highs, closes]),
                "low": np.minimum.reduce([opens, lows, closes]),
                "close": closes,
                "volume": volume.astype(float),
            }, index=idx)
            self._bars[sym] = df

    def _slice(self, sym: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        df = self._bars.get(sym)
        if df is None:
            return pd.DataFrame()
        out = df
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(timezone.utc)
            else:
                start_ts = start_ts.tz_convert(timezone.utc)
            out = out[out.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(timezone.utc)
            else:
                end_ts = end_ts.tz_convert(timezone.utc)
            out = out[out.index <= end_ts]
        return out

    def get_stock_bars(self, request) -> SimpleNamespace:
        syms = request.symbol_or_symbols
        if isinstance(syms, str):
            symbols = [syms]
        else:
            symbols = list(syms)
        start = getattr(request, "start", None)
        end = getattr(request, "end", None)
        frames = {}
        for sym in symbols:
            df = self._slice(sym, start, end)
            if not df.empty:
                frames[sym] = df
        if not frames:
            return SimpleNamespace(df=pd.DataFrame())
        combined = pd.concat(frames, names=["symbol", "timestamp"])
        combined.index = combined.index.set_names(["symbol", "timestamp"])
        return SimpleNamespace(df=combined)

    def latest_price(self, symbol: str) -> float:
        df = self._bars.get(symbol)
        if df is None or df.empty:
            return 0.0
        return float(df["close"].iloc[-1])

    def symbols(self) -> List[str]:
        return list(self._symbols)


@dataclass
class _SimOrder:
    id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_qty: float


class SimTradingClient:
    """Minimal TradingClient facsimile that settles immediately."""

    def __init__(self, data_client: SimStockHistoricalDataClient, starting_cash: float = 10_000.0) -> None:
        _ensure_data_dir()
        self._data = data_client
        self._state = self._load_state(starting_cash)
        self._orders: Dict[str, _SimOrder] = {}
        self._id = int(self._state.get("_last_order_id", 0))
        self.paper = True
        self.is_simulated = True

    # ----- Persistence -----
    def _load_state(self, starting_cash: float) -> Dict[str, dict]:
        if SIM_STATE_PATH.exists():
            try:
                return json.loads(SIM_STATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"cash": starting_cash, "positions": {}, "_last_order_id": 0}

    def _save_state(self) -> None:
        state = dict(self._state)
        state["_last_order_id"] = self._id
        SIM_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ----- Utilities -----
    def _next_id(self) -> str:
        self._id += 1
        return str(self._id)

    def _price(self, symbol: str, fallback: float | None = None) -> float:
        price = self._data.latest_price(symbol)
        if price > 0:
            return price
        return fallback if fallback is not None else 0.0

    # ----- API-like surface -----
    def get_all_positions(self) -> List[SimpleNamespace]:
        out = []
        for sym, pos in self._state.get("positions", {}).items():
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_price", 0.0))
            price = self._price(sym, avg)
            market_value = qty * price
            out.append(SimpleNamespace(
                symbol=sym,
                qty=qty,
                avg_entry_price=avg,
                market_value=market_value,
                current_price=price
            ))
        return out

    def get_account(self) -> SimpleNamespace:
        equity = self._state.get("cash", 0.0)
        for pos in self.get_all_positions():
            equity += pos.market_value
        return SimpleNamespace(equity=equity, cash=self._state.get("cash", 0.0), portfolio_value=equity)

    def submit_order(self, request) -> SimpleNamespace:
        symbol = request.symbol
        qty_attr = getattr(request, "qty", None)
        notional = getattr(request, "notional", None)
        side = "buy" if str(request.side).lower().startswith("buy") else "sell"
        limit_price = getattr(request, "limit_price", None)
        price = float(limit_price) if limit_price else self._price(symbol)
        if price <= 0:
            price = max(1.0, self._price(symbol, 100.0))
        if qty_attr is None or float(qty_attr or 0.0) <= 0:
            if notional is not None:
                qty = float(notional) / price if price else 0.0
            else:
                qty = 0.0
        else:
            qty = float(qty_attr)
        qty_signed = qty if side == "buy" else -qty
        self._apply_fill(symbol, qty_signed, price)
        order_id = self._next_id()
        order = _SimOrder(
            id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            status="filled",
            filled_qty=qty
        )
        self._orders[order_id] = order
        self._save_state()
        return SimpleNamespace(id=order_id, status="filled", filled_qty=qty)

    def get_asset(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(symbol=symbol, fractionable=True)

    def _apply_fill(self, symbol: str, qty_signed: float, price: float) -> None:
        positions = self._state.setdefault("positions", {})
        cash = float(self._state.get("cash", 0.0))
        pos = positions.get(symbol, {"qty": 0.0, "avg_price": price})
        qty_prev = float(pos.get("qty", 0.0))
        avg_prev = float(pos.get("avg_price", price))

        notional = qty_signed * price
        cash -= notional
        qty_new = qty_prev + qty_signed

        if abs(qty_new) < 1e-6:
            positions.pop(symbol, None)
        else:
            if qty_new > 0 and qty_prev >= 0:
                total_cost = qty_prev * avg_prev + qty_signed * price
                avg_price = total_cost / qty_new
            elif qty_new < 0 and qty_prev <= 0:
                total_cost = qty_prev * avg_prev + qty_signed * price
                avg_price = total_cost / qty_new if qty_new != 0 else price
            else:
                avg_price = price if qty_new >= 0 else -price
            positions[symbol] = {"qty": qty_new, "avg_price": avg_price}

        self._state["cash"] = cash

    def get_order_by_id(self, order_id: str) -> SimpleNamespace:
        order = self._orders.get(order_id)
        if order is None:
            return SimpleNamespace(id=order_id, status="canceled", filled_qty=0.0)
        return SimpleNamespace(id=order.id, status=order.status, filled_qty=order.filled_qty)

    def cancel_order_by_id(self, order_id: str) -> None:
        if order_id in self._orders:
            self._orders[order_id] = _SimOrder(
                id=order_id,
                symbol=self._orders[order_id].symbol,
                qty=self._orders[order_id].qty,
                side=self._orders[order_id].side,
                status="canceled",
                filled_qty=self._orders[order_id].filled_qty,
            )

    # Convenience for the web dashboard
    def snapshot(self) -> Dict[str, object]:
        acct = self.get_account()
        positions = []
        for pos in self.get_all_positions():
            positions.append({
                "symbol": pos.symbol,
                "qty": pos.qty,
                "avg_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
            })
        return {
            "cash": acct.cash,
            "equity": acct.equity,
            "positions": positions,
        }
