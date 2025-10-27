from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any, Dict
from pathlib import Path
from . import config as C

ALLOWED_KEYS = [
    "AUTOMATION_ENABLED","TRADING_WINDOWS_ET","WINDOW_TOL_MIN","AVOID_NEAR_CLOSE_MIN",
    "RUN_ONCE_PER_WINDOW","RUN_MARKERS_PATH",
    # Value-investor advisor settings exposed to UI
    "ENABLE_WEB_ADVISOR","WEB_ADVISOR_MODEL","WEB_ADVISOR_DOMAIN_ALLOWLIST",
    "WEB_ADVISOR_RECENCY_DAYS","WEB_ADVISOR_MAX_PAGES","ADVISOR_MAX_TRADES_PER_RUN",
    "EPISODES_MEMORY_LOOKBACK","MIN_HOLD_DAYS_BEFORE_SELL",
    # Baseline exposure policy and churn control
    "BASELINE_ENABLE","BASELINE_TICKER","BASELINE_MIN","BASELINE_MAX","BASELINE_TARGET",
    "BASELINE_MAX_STEP","BASELINE_ADJUST_COOLDOWN_MIN","POLICY_STATE_PATH",
    # News step (Step 2)
    "ENABLE_NEWS_CHECK","NEWS_SOURCES","NEWS_LOOKBACK_MIN","NEWS_MAX_PER_RUN",
    "NEWS_CACHE_PATH","NEWS_KEYWORDS_INCLUDE","NEWS_KEYWORDS_EXCLUDE",
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


def _normalize_settings(values: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(values or {})
    if "TARGET_POSITIONS" not in data and "MAX_POSITIONS" in data:
        try:
            data["TARGET_POSITIONS"] = int(data["MAX_POSITIONS"])
        except (TypeError, ValueError):
            data["TARGET_POSITIONS"] = data["MAX_POSITIONS"]
    if "TARGET_POSITIONS" in data:
        try:
            data["TARGET_POSITIONS"] = int(data["TARGET_POSITIONS"])
        except (TypeError, ValueError):
            pass
    return data


def _coerce_settings(values: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(values)
    tw = data.get("TRADING_WINDOWS_ET")
    if isinstance(tw, str):
        data["TRADING_WINDOWS_ET"] = [x.strip() for x in tw.split(",") if x.strip()]
    # Coerce lists provided as comma-separated strings
    for key in ("NEWS_SOURCES","NEWS_KEYWORDS_INCLUDE","NEWS_KEYWORDS_EXCLUDE","WEB_ADVISOR_DOMAIN_ALLOWLIST"):
        v = data.get(key)
        if isinstance(v, str):
            data[key] = [x.strip() for x in v.split(",") if x.strip()]
    sw = data.get("SLEEVE_WEIGHTS")
    if isinstance(sw, dict):
        total = sum(max(0.0, float(v)) for v in sw.values()) or 1.0
        data["SLEEVE_WEIGHTS"] = {k: max(0.0, float(v)) / total for k, v in sw.items()}
    return data

def _defaults() -> Dict[str, Any]:
    return {k: getattr(C, k) for k in dir(C) if k.isupper()}

def load_overrides() -> Dict[str, Any]:
    p = Path(C.SETTINGS_OVERRIDES_PATH)
    if not p.exists(): return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    filtered = {k: raw[k] for k in ALLOWED_KEYS if k in raw}
    return _normalize_settings(filtered)

def save_overrides(new_vals: Dict[str, Any]) -> None:
    p = Path(C.SETTINGS_OVERRIDES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = load_overrides()
    cur.update({k: new_vals[k] for k in new_vals if k in ALLOWED_KEYS})
    p.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def get_settings() -> Dict[str, Any]:
    defaults = _defaults()
    overrides = load_overrides()
    defaults.update(overrides)
    return _coerce_settings(defaults)


def get_cfg() -> SimpleNamespace:
    settings = get_settings()
    return SimpleNamespace(**settings)
