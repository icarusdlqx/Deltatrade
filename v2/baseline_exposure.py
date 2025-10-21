from __future__ import annotations
"""
Baseline exposure policy: maintain a 60–70% invested ratio (target 65%)
by topping up/paring a ballast ETF (default: SPY) after each analysis run.
Uses Alpaca TradingClient 'notional' market orders when outside the band.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_state(path: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(d: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _minutes_since(ts: float) -> float:
    return (time.time() - ts) / 60.0


def _account_metrics(tc: Optional[TradingClient]) -> Dict[str, float]:
    if not tc:
        return {"equity": 0.0, "cash": 0.0, "long_mv": 0.0, "invested": 0.0}
    acct = tc.get_account()
    equity = _safe_float(getattr(acct, "equity", 0.0))
    cash = _safe_float(getattr(acct, "cash", 0.0))
    long_mv = _safe_float(getattr(acct, "long_market_value", 0.0))
    invested = 0.0 if equity <= 0 else max(0.0, long_mv) / max(1e-9, equity)
    return {"equity": equity, "cash": cash, "long_mv": long_mv, "invested": invested}


def maybe_adjust_baseline_exposure(tc: Optional[TradingClient], cfg, logger=None) -> Optional[Dict[str, Any]]:
    """If invested ratio is outside configured band, trade ballast ETF toward target."""

    if not tc or not getattr(cfg, "BASELINE_ENABLE", True):
        return None

    state_path = getattr(cfg, "POLICY_STATE_PATH", "data/policy_state.json")
    st = _load_state(state_path)
    last_ts = float(st.get("baseline_last_ts", 0.0))
    cooldown_min = int(getattr(cfg, "BASELINE_ADJUST_COOLDOWN_MIN", 120))
    if last_ts and _minutes_since(last_ts) < cooldown_min:
        return {"skipped": True, "reason": f"cooldown_active_{cooldown_min}m"}

    metrics = _account_metrics(tc)
    equity, invested = metrics["equity"], metrics["invested"]
    if equity <= 0:
        return {"skipped": True, "reason": "no_equity"}

    sym = getattr(cfg, "BASELINE_TICKER", "SPY")
    min_r = float(getattr(cfg, "BASELINE_MIN", 0.60))
    max_r = float(getattr(cfg, "BASELINE_MAX", 0.70))
    tgt_r = float(getattr(cfg, "BASELINE_TARGET", 0.65))
    max_step = float(getattr(cfg, "BASELINE_MAX_STEP", 0.20))
    min_notional = float(getattr(cfg, "MIN_ORDER_NOTIONAL", 100.0))

    action: Optional[str] = None
    delta = 0.0
    if invested < min_r:
        delta = max(0.0, tgt_r - invested)
        action = "buy"
    elif invested > max_r:
        delta = max(0.0, invested - tgt_r)
        action = "sell"
    else:
        return {"skipped": True, "reason": "within_band", "invested": invested}

    notional = max(min_notional, min(delta * equity, max_step * equity))
    if notional < min_notional:
        return {"skipped": True, "reason": "too_small", "calc_notional": notional}

    side = OrderSide.BUY if action == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=sym,
        notional=round(notional, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = tc.submit_order(req)
        st["baseline_last_ts"] = time.time()
        _save_state(st, state_path)
        out = {
            "ok": True,
            "baseline_action": action,
            "symbol": sym,
            "notional": req.notional,
            "invested_before": invested,
            "equity": equity,
            "order_id": getattr(order, "id", None),
        }
        if logger:
            logger.info(
                f"[baseline] {action.upper()} {sym} notional ${req.notional:,} "
                f"(invested={invested:.2%} → target {tgt_r:.0%}) order_id={out['order_id']}"
            )
        return out
    except Exception as exc:  # pragma: no cover - network interaction
        if logger:
            logger.error(f"[baseline] submit_order failed: {exc}")
        return {
            "ok": False,
            "error": str(exc),
            "attempted": {"action": action, "symbol": sym, "notional": notional},
        }
