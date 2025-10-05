from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any, Dict
from pathlib import Path
from . import config as C

ALLOWED_KEYS = [
    "AUTOMATION_ENABLED","TRADING_WINDOWS_ET","WINDOW_TOL_MIN","AVOID_NEAR_CLOSE_MIN",
    "UNIVERSE_MODE","UNIVERSE_MAX","DATA_LOOKBACK_DAYS","MIN_BARS",
    "RESID_MOM_LOOKBACK","TREND_FAST","TREND_SLOW","REVERSAL_DAYS","WINSOR_PCT",
    "ENABLE_EVENT_SCORE","NEWS_LOOKBACK_DAYS","EVENT_TOP_K","EVENT_ALPHA_MULT",
    "ENABLE_VOL_TARGETING","TARGET_PORTFOLIO_VOL","VOL_TARGET_ANNUAL","LAMBDA_RISK","TURNOVER_PENALTY",
    "NAME_MAX","MAX_WEIGHT_PER_NAME","SECTOR_MAX","REBALANCE_BAND",
    "TARGET_POSITIONS","MAX_POSITIONS","TURNOVER_CAP","CASH_BUFFER",
    "SLEEVE_WEIGHTS",
    "ENABLE_COST_GATE","DRY_RUN","MIN_ORDER_NOTIONAL","MAX_SLICES","LIMIT_SLIP_BP",
    "COST_SPREAD_BPS","COST_IMPACT_KAPPA","COST_IMPACT_PSI","FILL_TIMEOUT_SEC",
    "MIN_NET_BPS_TO_TRADE",
    "ATR_STOP_MULT","TAKE_PROFIT_ATR","TIME_STOP_DAYS",
]

def _defaults() -> Dict[str, Any]:
    return {k: getattr(C, k) for k in dir(C) if k.isupper()}

def load_overrides() -> Dict[str, Any]:
    p = Path(C.SETTINGS_OVERRIDES_PATH)
    if not p.exists(): return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: raw[k] for k in ALLOWED_KEYS if k in raw}

def save_overrides(new_vals: Dict[str, Any]) -> None:
    p = Path(C.SETTINGS_OVERRIDES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = load_overrides()
    cur.update({k: new_vals[k] for k in new_vals if k in ALLOWED_KEYS})
    p.write_text(json.dumps(cur, indent=2), encoding="utf-8")

def get_cfg() -> SimpleNamespace:
    d = _defaults()
    ov = load_overrides()
    if "TRADING_WINDOWS_ET" in ov and isinstance(ov["TRADING_WINDOWS_ET"], str):
        ov["TRADING_WINDOWS_ET"] = [x.strip() for x in ov["TRADING_WINDOWS_ET"].split(",") if x.strip()]
    if "SLEEVE_WEIGHTS" in ov and isinstance(ov["SLEEVE_WEIGHTS"], dict):
        s = sum(max(0.0, float(v)) for v in ov["SLEEVE_WEIGHTS"].values()) or 1.0
        ov["SLEEVE_WEIGHTS"] = {k: max(0.0, float(v))/s for k,v in ov["SLEEVE_WEIGHTS"].items()}
    d.update(ov)
    return SimpleNamespace(**d)
