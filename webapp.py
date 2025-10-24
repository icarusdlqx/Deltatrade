from __future__ import annotations
"""
Compatibility patch: make run_once() safe whether callers assign a single value
    ep = run_once()
or try to unpack two values
    ep, summary = run_once()
This avoids 'ValueError: too many values to unpack (expected 2)' seen on manual runs.
"""

# --- START GPT-5 ENFORCER (medium effort, hard-coded prompt, logging) ---
try:
    from v2.agents_llm_enforce import apply_patch as _llm_enforce_apply

    _llm_enforce_apply()
except Exception as _e:
    print("[llm_enforce] not applied:", _e)
# --- END GPT-5 ENFORCER ---

# --- START CODEX PATCH HOOK v3 (LLM failover + abs risk + dyn cost) ---
try:
    from v2.codex_patch_v3 import apply as _codex_apply_v3
    _codex_apply_v3()
except Exception as _e:
    print("[codex_v3] not applied:", _e)
# --- END CODEX PATCH HOOK v3 ---

# --- START CODEX PATCH HOOK (schedule + actions + LLM) ---
import os
_codex_apply = None
try:
    from v2.codex_patch_v2 import apply as _codex_apply
except Exception:
    try:
        from v2.codex_patch import apply as _codex_apply  # fallback to prior patch if present
    except Exception:
        _codex_apply = None
if _codex_apply:
    _codex_apply()
try:
    if os.getenv("RUN_LLM_SMOKE", "1") == "1":
        from v2.llm_client import smoke_test as _llm_smoke
        _llm_smoke()
except Exception as _e:
    print("[codex] LLM smoke test skipped:", _e)
# --- END CODEX PATCH HOOK ---

import csv
import io
import json
import os
import logging
import gevent
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple
import pytz

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from alpaca_client import (
    close_all_positions as alp_close_all_positions,
    close_position as alp_close_position,
    get_account as alp_get_account,
    get_clock as alp_get_clock,
    list_positions as alp_list_positions,
)

from v2.config import MAX_LOG_ROWS
import v2.orchestrator as _orc_mod  # we'll wrap _orc_mod.run_once below
from v2.settings_bridge import get_cfg, load_overrides, save_overrides
from v2.utils import write_jsonl

# Configure logging for webapp
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "deltatrade-demo")


@app.context_processor
def inject_positions_count() -> Dict[str, int]:
    """Expose real-time open positions count to all templates."""
    count = 0
    try:
        positions = alp_list_positions()
        open_positions = [
            pos
            for pos in positions
            if float(getattr(pos, "qty", getattr(pos, "quantity", "0")) or 0.0) != 0.0
        ]
        count = len(open_positions)
    except Exception:
        count = 0
    return {"positions_open_count": count, "positions_count": count}

# ---------------------------------------------------------------------------
# Manual-run safety shim for run_once()
# ---------------------------------------------------------------------------
# Some code paths tried to do:   episode, summary = run_once()
# while run_once() historically returned a single dict. Unpacking a dict raises
# "too many values to unpack". This shim makes run_once() flexible:
#   - If caller unpacks, it yields (episode_dict, summary_dict).
#   - If caller assigns to a single var, it acts dict-like (ep.get(...), etc.).
try:
    _orig_run_once = getattr(_orc_mod, "run_once", None)
except Exception:
    _orig_run_once = None


def _make_summary_from_episode(ep: Dict[str, Any] | Mapping[str, Any]) -> Dict[str, str]:
    if isinstance(ep, Mapping):
        ep_dict = dict(ep)
    elif isinstance(ep, dict):
        ep_dict = ep
    else:
        ep_dict = {}
    try:
        summary = _summarize_last_run(ep_dict)
        text = summary.get("summary") or "run recorded"
        plain = summary.get("plain_english") or text
        return {"summary": text, "plain_english": plain}
    except Exception:
        return {"summary": "run recorded", "plain_english": "run recorded"}


class _EpisodeShim:
    """Dict-like wrapper that also cleanly unpacks into (episode, summary)."""

    __slots__ = ("_ep", "_summary")

    def __init__(self, ep: Any, summary: Dict[str, Any] | None = None):
        if isinstance(ep, Mapping):
            self._ep: Dict[str, Any] = dict(ep)
        elif isinstance(ep, dict):
            self._ep = ep
        else:
            self._ep = {}
        if isinstance(summary, Mapping):
            self._summary: Dict[str, Any] | None = dict(summary)
        else:
            self._summary = None

    # mapping-ish APIs used throughout the app
    def get(self, *a, **kw):
        return self._ep.get(*a, **kw)

    def __getitem__(self, k):
        return self._ep[k]

    def __contains__(self, k):
        return k in self._ep

    def keys(self):
        return self._ep.keys()

    def items(self):
        return self._ep.items()

    def values(self):
        return self._ep.values()

    def __len__(self):
        return len(self._ep)

    def __bool__(self):
        return bool(self._ep)

    def __repr__(self):
        return f"_EpisodeShim({self._ep!r})"

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._ep)

    def summary_dict(self) -> Dict[str, Any]:
        if self._summary is None:
            self._summary = _make_summary_from_episode(self._ep)
        return dict(self._summary)

    # when someone does: ep, summary = run_once()
    def __iter__(self):
        yield self._ep
        yield self.summary_dict()


def _compat_run_once(*args, **kwargs):
    if _orig_run_once is None:
        logging.getLogger(__name__).error("run_once() not found on v2.orchestrator")
        return _EpisodeShim({})
    res = _orig_run_once(*args, **kwargs)
    if isinstance(res, _EpisodeShim):
        return res
    if isinstance(res, tuple) and len(res) > 0:
        ep = res[0]
        summary = res[1] if len(res) >= 2 else None
        if isinstance(summary, str):
            summary = {"summary": summary, "plain_english": summary}
        return _EpisodeShim(ep, summary)
    return _EpisodeShim(res)


def _normalize_episode_result(result: Any) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    if isinstance(result, _EpisodeShim):
        return result.as_dict(), result.summary_dict()

    if isinstance(result, tuple):
        ep_raw = result[0] if len(result) > 0 else {}
        summary_raw = result[1] if len(result) > 1 else None

        ep_dict, _ = _normalize_episode_result(ep_raw)

        if isinstance(summary_raw, Mapping):
            summary_dict = dict(summary_raw)
        elif isinstance(summary_raw, str):
            summary_dict = {"summary": summary_raw, "plain_english": summary_raw}
        elif isinstance(summary_raw, _EpisodeShim):
            _, summary_dict = _normalize_episode_result(summary_raw)
        else:
            summary_dict = None

        if summary_dict is None:
            summary_dict = _make_summary_from_episode(ep_dict)

        return ep_dict, summary_dict

    if isinstance(result, Mapping):
        ep_dict = dict(result)
        return ep_dict, _make_summary_from_episode(ep_dict)

    return {}, None


# Monkey-patch module function and the local global name so ALL call sites use it
_orc_mod.run_once = _compat_run_once
globals()["run_once"] = _compat_run_once


NAV_ITEMS = [
    {"endpoint": "dashboard", "label": "Dashboard", "icon": "bi-speedometer2"},
    {"endpoint": "positions", "label": "Positions", "icon": "bi-graph-up"},
    {"endpoint": "log", "label": "Log", "icon": "bi-journal-text"},
    {"endpoint": "performance", "label": "Performance", "icon": "bi-bar-chart"},
    {"endpoint": "settings", "label": "Settings", "icon": "bi-sliders"},
    {"endpoint": "logout", "label": "Logout", "icon": "bi-box-arrow-right"},
]
def _environment_summary() -> Dict[str, Any]:
    alpaca_key = bool(os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_V3"))
    alpaca_secret = bool(os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY_V3"))
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    sim_mode_env = os.environ.get("SIM_MODE", "false").lower() in ("true", "1", "yes", "y")
    simulated = sim_mode_env or not (alpaca_key and alpaca_secret)
    paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
    mode_label = "Simulated" if simulated else ("Paper Trading" if paper else "Live Trading")
    return {
        "alpaca_key": alpaca_key,
        "alpaca_secret": alpaca_secret,
        "openai_key": openai_key,
        "paper": paper,
        "simulated": simulated,
        "mode_label": mode_label,
        "as_of": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
    }


def _load_episodes(path: str, limit: int = 200, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    episodes: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return episodes, 0
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return episodes, 0
    total = len(lines)
    if limit <= 0:
        limit = total
    offset = max(offset, 0)
    start = max(total - offset - limit, 0)
    end = max(total - offset, 0)
    subset = lines[start:end]
    for line in subset:
        try:
            episodes.append(json.loads(line))
        except Exception:
            continue
    episodes.reverse()
    return episodes, total


def _alpaca_credentials_available() -> bool:
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_V3")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY_V3")
    return bool(api_key and secret)


def insert_log(level: str, kind: str, data: Dict[str, Any]) -> None:
    try:
        cfg = get_cfg()
        record = {
            "as_of": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
            "level": level,
            "kind": kind,
            "payload": data,
        }
        write_jsonl(cfg.EPISODES_PATH, record)
    except Exception:
        logger.exception("Failed to record log entry", extra={"level": level, "kind": kind})


def _next_run_time(cfg) -> datetime | None:
    windows = getattr(cfg, "TRADING_WINDOWS_ET", []) or []
    if not windows:
        return None
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    upcoming: List[datetime] = []
    for win in windows:
        try:
            hour_str, minute_str = str(win).split(":", 1)
            hour, minute = int(hour_str), int(minute_str)
        except Exception:
            continue
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        upcoming.append(candidate)
    if not upcoming:
        return None
    return min(upcoming)


def _format_next_run(dt: datetime | None) -> str:
    if not dt:
        return "No run scheduled"
    return dt.strftime("%a %b %d · %I:%M %p ET")


def _mode_label(mode_code: str | None) -> str:
    if not mode_code:
        return "Unknown"
    mode_code = str(mode_code).lower()
    if mode_code in ("live", "realtime"):
        return "Live"
    if mode_code in ("sim", "simulated"):
        return "Simulated"
    return "Paper"


def _portfolio_snapshot_from_episode(latest: Dict[str, Any] | None) -> Dict[str, Any]:
    if not latest:
        return {"equity": 0.0, "cash": 0.0, "cash_pct": 0.0, "positions": 0, "target_gross": 0.0}
    gate = latest.get("gate", {}) or {}
    equity = float(gate.get("equity") or latest.get("investable", 0.0))
    cash = float(gate.get("cash") or 0.0)
    cash_frac = float(gate.get("cash_frac") or 0.0)
    positions = len(latest.get("targets", {}) or {})
    exposure = gate.get("exposure") or {}
    target_gross = float(exposure.get("sum_abs_weights") or exposure.get("target_gross") or 0.0)
    return {
        "equity": equity,
        "cash": cash,
        "cash_pct": cash_frac * 100.0,
        "positions": positions,
        "target_gross": target_gross,
    }


def _summarize_last_run(latest: Dict[str, Any] | None) -> Dict[str, Any]:
    if not latest:
        return {
            "summary": "No runs recorded yet.",
            "plain_english": "No runs recorded yet.",
            "net_bps": 0.0,
            "cost_bps": 0.0,
            "expected_bps": 0.0,
            "turnover_pct": 0.0,
            "order_count": 0,
            "reasons": [],
            "traded": False,
            "proceed_gate": False,
            "actual_orders": 0,
        }
    gate = latest.get("gate", {}) or {}
    expected = float(gate.get("expected_alpha_bps") or latest.get("expected_alpha_bps", 0.0) or 0.0)
    cost = float(gate.get("cost_bps") or latest.get("est_cost_bps", 0.0) or 0.0)
    net = float(gate.get("net_bps") or latest.get("net_edge_bps", 0.0) or 0.0)
    turnover_pct = float(gate.get("turnover_pct") or 0.0)
    order_count = int(gate.get("order_count") or latest.get("planned_orders_count", 0) or 0)
    reasons_raw = gate.get("friendly_reasons") or gate.get("reasons") or []
    if isinstance(reasons_raw, list):
        reasons = [str(r) for r in reasons_raw if r]
    elif reasons_raw:
        reasons = [str(reasons_raw)]
    else:
        reasons = []
    proceed_gate = bool(gate.get("proceed"))
    proceed_final = bool(gate.get("proceed_final") or latest.get("proceed"))
    orders_submitted = latest.get("orders_submitted") or []
    if isinstance(orders_submitted, list):
        actual_orders = [oid for oid in orders_submitted if oid != "DRY_RUN_NO_ORDERS"]
    else:
        actual_orders = []
    traded = len(actual_orders) > 0
    market_commentary = str(latest.get("market_commentary") or "").strip()

    if traded:
        summary = "Traded {count} orders · net {net:.1f} bps after {cost:.1f} bps costs · turnover {turnover:.1f}%".format(
            count=len(actual_orders), net=net, cost=cost, turnover=turnover_pct
        )
        plain = market_commentary or (
            "Placed {count} orders; turnover {turnover:.1f}%; exp {expected:.1f} bps → net {net:.1f} bps.".format(
                count=len(actual_orders), turnover=turnover_pct, expected=expected, net=net
            )
        )
    else:
        reason_text = ", ".join(reasons) if reasons else "none"
        summary = (
            "Skipped — reasons: {reasons}; gate math: exp {expected:.1f} bps, cost {cost:.1f} bps, net {net:.1f} bps"
        ).format(reasons=reason_text, expected=expected, cost=cost, net=net)
        plain = market_commentary or (
            "Skipped — reasons: {reasons}. Gate: exp {expected:.1f}, cost {cost:.1f}, net {net:.1f} bps."
        ).format(reasons=reason_text, expected=expected, cost=cost, net=net)

    advisor_summary = ""
    advisor_snippet = latest.get("advisor_summary_1p")
    if not advisor_snippet and isinstance(latest.get("advisor_report"), dict):
        advisor_snippet = latest["advisor_report"].get("world_state_summary")
    if advisor_snippet:
        advisor_summary = str(advisor_snippet).strip()
        if advisor_summary:
            summary = (summary + " Advisor: " + advisor_summary).strip()
            plain = (plain + " Advisor: " + advisor_summary).strip()
    return {
        "summary": summary,
        "plain_english": plain,
        "net_bps": net,
        "cost_bps": cost,
        "expected_bps": expected,
        "turnover_pct": turnover_pct,
        "order_count": order_count,
        "reasons": reasons,
        "traded": traded,
        "proceed_gate": proceed_gate,
        "proceed_final": proceed_final,
        "actual_orders": len(actual_orders),
        "advisor_summary": advisor_summary,
        "market_commentary": market_commentary,
    }


def _dashboard_metrics(episodes: List[Dict[str, Any]], total_count: int | None = None) -> Dict[str, Any]:
    if not episodes:
        return {
            "total_runs": total_count or 0,
            "proceed_rate": 0,
            "avg_expected": 0,
            "avg_cost": 0,
            "net_edge": 0,
        }
    # Use actual total count from file, not len(episodes) which may be limited
    total_runs = total_count if total_count is not None else len(episodes)
    proceed = sum(1 for ep in episodes if ep.get("proceed"))
    expected_vals: List[float] = []
    cost_vals: List[float] = []
    for ep in episodes:
        gate = ep.get("gate", {}) or {}
        expected_vals.append(float(gate.get("expected_alpha_bps", ep.get("expected_alpha_bps", 0.0))))
        cost_vals.append(float(gate.get("cost_bps", ep.get("est_cost_bps", 0.0))))
    net_edge = sum(e - c for e, c in zip(expected_vals, cost_vals)) / 10000.0
    return {
        "total_runs": total_runs,
        "proceed_rate": int(round(100 * proceed / len(episodes))) if episodes else 0,
        "avg_expected": round(mean(expected_vals), 2) if expected_vals else 0,
        "avg_cost": round(mean(cost_vals), 2) if cost_vals else 0,
        "net_edge": round(net_edge, 4),
    }


def _positions_snapshot(env: Dict[str, Any]) -> Dict[str, Any]:
    if env.get("simulated"):
        from v2.simulated_clients import SimStockHistoricalDataClient, SimTradingClient

        data_client = SimStockHistoricalDataClient()
        trade_client = SimTradingClient(data_client)
        snapshot = trade_client.snapshot()
        snapshot["simulated"] = True
        return snapshot

    if not _alpaca_credentials_available():
        return {"positions": [], "cash": 0.0, "equity": 0.0, "error": "Alpaca credentials not configured.", "simulated": False}

    try:
        raw_positions = alp_list_positions()
        positions: List[Dict[str, Any]] = []
        for pos in raw_positions:
            get = pos.get if isinstance(pos, Mapping) else None
            symbol = get("symbol") if get else getattr(pos, "symbol", "")
            qty = get("qty") if get else getattr(pos, "qty", 0.0)
            qty_available = get("qty_available") if get else getattr(pos, "qty_available", qty)
            avg_price = get("avg_entry_price") if get else getattr(pos, "avg_entry_price", 0.0)
            current_price = get("current_price") if get else getattr(pos, "current_price", 0.0)
            market_value = get("market_value") if get else getattr(pos, "market_value", 0.0)
            unrealized_pl = get("unrealized_pl") if get else getattr(pos, "unrealized_pl", 0.0)
            positions.append(
                {
                    "symbol": symbol,
                    "qty": float(qty or 0.0),
                    "qty_available": float(qty_available or qty or 0.0),
                    "avg_price": float(avg_price or 0.0),
                    "current_price": float(current_price or 0.0),
                    "market_value": float(market_value or 0.0),
                    "unrealized_pl": float(unrealized_pl or 0.0),
                }
            )
        account = alp_get_account()
        acc_get = account.get if isinstance(account, Mapping) else None
        cash_val = acc_get("cash") if acc_get else getattr(account, "cash", 0.0)
        equity_val = acc_get("equity") if acc_get else getattr(account, "equity", 0.0)
        return {
            "positions": positions,
            "cash": float(cash_val or 0.0),
            "equity": float(equity_val or 0.0),
            "simulated": False,
        }
    except Exception as exc:
        return {"positions": [], "cash": 0.0, "equity": 0.0, "error": str(exc), "simulated": False}


def _safe_clock(env: Dict[str, Any]) -> Dict[str, Any]:
    if env.get("simulated") or not _alpaca_credentials_available():
        return {"is_open": None, "next_open": None, "next_close": None, "error": "unavailable"}
    try:
        return alp_get_clock()
    except Exception as exc:
        return {"is_open": None, "next_open": None, "next_close": None, "error": str(exc)}


def _market_status_suffix(clock: Dict[str, Any]) -> str:
    if not clock:
        return ""
    is_open = clock.get("is_open")
    if is_open is None or is_open:
        return ""
    next_open = clock.get("next_open")
    if next_open:
        return f" Market closed until {next_open}."
    return " Market is currently closed."


def _normalize_error_code(error: str | None) -> str:
    if not error:
        return ""
    return str(error).split(":", 1)[0].strip()


def _format_error_reason(error: str | None, symbol: str | None) -> str:
    code = _normalize_error_code(error)
    mapping = {
        "missing_symbol": "No symbol provided.",
        "no_position": "No open position in {symbol}.",
        "zero_available": "No shares available to sell for {symbol}.",
        "missing_credentials": "Alpaca API credentials not configured.",
        "simulation_mode": "Manual sells are disabled in simulation mode.",
    }
    template = mapping.get(code)
    if template:
        if "{symbol}" in template:
            label = symbol or "the requested symbol"
            return template.format(symbol=label)
        return template
    return str(error or "Unknown error")


def _sell_success_message(symbol: str, result: Dict[str, Any], clock: Dict[str, Any]) -> str:
    method = result.get("method") or "close_position"
    base = f"Sell {symbol}: submitted via {method.replace('_', ' ')}."
    qty = result.get("qty")
    if qty:
        base += f" Qty {qty}."
    return base + _market_status_suffix(clock)


def _sell_failure_message(prefix: str, symbol: str | None, result: Dict[str, Any], clock: Dict[str, Any]) -> str:
    error_text = _format_error_reason(result.get("error"), symbol)
    fallback = result.get("fallback_error")
    if fallback and fallback != result.get("error"):
        error_text = f"{error_text} (fallback: {fallback})"
    return f"{prefix}{error_text}{_market_status_suffix(clock)}"


def _performance_series(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: List[str] = []
    expected: List[float] = []
    costs: List[float] = []
    cumulative: List[float] = []
    total = 0.0
    for ep in reversed(episodes):
        labels.append(ep.get("as_of", ""))
        gate = ep.get("gate", {}) or {}
        exp = float(gate.get("expected_alpha_bps", ep.get("expected_alpha_bps", 0.0)))
        cost = float(gate.get("cost_bps", ep.get("est_cost_bps", 0.0)))
        expected.append(exp)
        costs.append(cost)
        total += (exp - cost) / 10000.0
        cumulative.append(total)
    labels.reverse(); expected.reverse(); costs.reverse(); cumulative.reverse()
    return {"labels": labels, "expected": expected, "costs": costs, "cumulative": cumulative}


@app.context_processor
def inject_nav() -> Dict[str, Any]:
    return {"nav_items": NAV_ITEMS, "current_year": datetime.now(pytz.timezone("US/Eastern")).year}


@app.route("/")
def root():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    cfg = get_cfg()
    episodes, total = _load_episodes(cfg.EPISODES_PATH, limit=120)
    latest = episodes[0] if episodes else None
    env = _environment_summary()
    run_status = request.args.get("run_status")
    run_message = request.args.get("run_message")
    metrics = _dashboard_metrics(episodes, total_count=total)
    next_run_dt = _next_run_time(cfg)
    next_run_label = _format_next_run(next_run_dt) if next_run_dt else "No run scheduled"
    env_mode_code = "live" if not env.get("paper", True) else ("sim" if env.get("simulated") else "paper")
    gate_mode = (latest.get("gate", {}).get("mode") if latest else None) or None
    mode_code = gate_mode or env_mode_code
    status = {
        "mode": mode_code,
        "mode_label": _mode_label(mode_code),
        "enabled": bool(getattr(cfg, "AUTOMATION_ENABLED", False)),
        "next_run": next_run_label,
        "model": (latest.get("gate", {}).get("model") if latest else None) or (os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or getattr(cfg, "OPENAI_MODEL", "")),
        "effort": (latest.get("gate", {}).get("effort") if latest else None) or (os.getenv("REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or getattr(cfg, "OPENAI_REASONING_EFFORT", "")),
    }
    portfolio_snapshot = _portfolio_snapshot_from_episode(latest)
    last_run_summary = _summarize_last_run(latest)
    last_diag = latest.get("diag", {}) if latest else {}
    return render_template(
        "dashboard.html",
        cfg=cfg,
        episodes=episodes[:5],
        latest=latest,
        env=env,
        metrics=metrics,
        status=status,
        portfolio_snapshot=portfolio_snapshot,
        last_run=last_run_summary,
        last_diag=last_diag,
        run_status=run_status,
        run_message=run_message,
        active_page="dashboard",
    )


@app.route("/positions")
def positions():
    env = _environment_summary()
    snapshot_raw = _positions_snapshot(env)
    try:
        open_positions = [
            pos
            for pos in snapshot_raw.get("positions", [])
            if float(pos.get("qty") if isinstance(pos, dict) else getattr(pos, "qty", 0.0) or 0.0) != 0.0
        ]
    except Exception:
        open_positions = snapshot_raw.get("positions", [])
    snapshot = dict(snapshot_raw)
    snapshot["positions"] = open_positions
    snapshot["positions_count"] = len(open_positions)
    message = request.args.get("message")
    status = request.args.get("status", "success")
    return render_template(
        "positions.html",
        snapshot=snapshot,
        env=env,
        active_page="positions",
        message=message,
        status=status,
    )


@app.post("/positions/sell")
def sell_position():
    env = _environment_summary()
    payload = request.get_json(silent=True) or {}
    symbol = (request.form.get("symbol") if request.form else None) or payload.get("symbol") or ""
    symbol = str(symbol).strip().upper()
    clock = _safe_clock(env)
    status_code = 200

    if not symbol:
        result: Dict[str, Any] = {"ok": False, "error": "missing_symbol"}
        message = _sell_failure_message("Sell: failed — ", None, result, clock)
        status_code = 400
    elif env.get("simulated"):
        result = {"ok": False, "error": "simulation_mode"}
        message = _sell_failure_message(f"Sell {symbol}: failed — ", symbol, result, clock)
    elif not _alpaca_credentials_available():
        result = {"ok": False, "error": "missing_credentials"}
        message = _sell_failure_message(f"Sell {symbol}: failed — ", symbol, result, clock)
    else:
        try:
            result = alp_close_position(symbol)
        except Exception as exc:
            logger.exception("Error attempting to close position %s", symbol)
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if result.get("ok"):
            message = _sell_success_message(symbol, result, clock)
        else:
            message = _sell_failure_message(f"Sell {symbol}: failed — ", symbol, result, clock)

    log_payload = {
        "action": "manual_sell",
        "symbol": symbol,
        "clock": clock,
        "result": result,
        "message": message,
    }
    insert_log("ORDER", "manual_sell", log_payload)

    snapshot = _positions_snapshot(env)
    response = {
        "ok": bool(result.get("ok")),
        "result": result,
        "clock": clock,
        "snapshot": snapshot,
        "message": message,
    }
    return jsonify(response), status_code


@app.post("/positions/sell_all")
def sell_all_positions():
    env = _environment_summary()
    clock = _safe_clock(env)
    status_code = 200

    if env.get("simulated"):
        result: Dict[str, Any] = {"ok": False, "error": "simulation_mode"}
        message = _sell_failure_message("Sell all: failed — ", None, result, clock)
    elif not _alpaca_credentials_available():
        result = {"ok": False, "error": "missing_credentials"}
        message = _sell_failure_message("Sell all: failed — ", None, result, clock)
    else:
        try:
            result = alp_close_all_positions(cancel_open_orders=True)
        except Exception as exc:
            logger.exception("Error attempting to close all positions")
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if result.get("ok"):
            statuses = result.get("status") or []
            count = len(statuses)
            message = f"Sell all: submitted {count} close request{'s' if count != 1 else ''}."
            message += _market_status_suffix(clock)
        else:
            message = _sell_failure_message("Sell all: failed — ", None, result, clock)

    log_payload = {
        "action": "manual_sell_all",
        "clock": clock,
        "result": result,
        "message": message,
    }
    insert_log("ORDER", "manual_sell_all", log_payload)

    snapshot = _positions_snapshot(env)
    response = {
        "ok": bool(result.get("ok")),
        "result": result,
        "clock": clock,
        "snapshot": snapshot,
        "message": message,
    }
    return jsonify(response), status_code


@app.route("/log")
def log():
    cfg = get_cfg()
    limit = int(request.args.get("limit", MAX_LOG_ROWS))
    limit = max(1, min(limit, MAX_LOG_ROWS))
    page = max(int(request.args.get("page", 1) or 1), 1)
    offset = (page - 1) * limit
    episodes, total = _load_episodes(cfg.EPISODES_PATH, limit=limit, offset=offset)
    has_more = (offset + len(episodes)) < total
    start_index = offset + 1 if episodes else 0
    end_index = offset + len(episodes)
    total_pages = (total + limit - 1) // limit if total else 0
    env = _environment_summary()
    return render_template(
        "log.html",
        episodes=episodes,
        active_page="log",
        env=env,
        limit=limit,
        page=page,
        total=total,
        has_more=has_more,
        start_index=start_index,
        end_index=end_index,
        total_pages=total_pages,
    )


@app.route("/log.csv")
def log_csv():
    cfg = get_cfg()
    episodes, _ = _load_episodes(cfg.EPISODES_PATH, limit=MAX_LOG_ROWS)
    output = io.StringIO()
    fieldnames = [
        "timestamp",
        "mode",
        "model",
        "effort",
        "equity",
        "cash",
        "cash_frac",
        "order_count",
        "turnover_pct",
        "expected_alpha_bps",
        "cost_bps",
        "net_bps",
        "min_net_bps",
        "proceed_gate",
        "force_onboard",
        "passes_net_bps",
        "risk_officer_approved",
        "proceed_final",
        "reasons",
        "friendly_reasons",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for ep in episodes:
        gate = ep.get("gate", {}) or {}
        reasons_raw = gate.get("reasons") or ep.get("gate_reason") or []
        if isinstance(reasons_raw, list):
            reasons_str = ";".join(str(r) for r in reasons_raw if r)
        elif reasons_raw:
            reasons_str = str(reasons_raw)
        else:
            reasons_str = ""
        friendly_raw = gate.get("friendly_reasons") or ep.get("friendly_reasons") or []
        if isinstance(friendly_raw, list):
            friendly_str = ";".join(str(r) for r in friendly_raw if r)
        elif friendly_raw:
            friendly_str = str(friendly_raw)
        else:
            friendly_str = ""
        writer.writerow({
            "timestamp": ep.get("as_of", ""),
            "mode": gate.get("mode"),
            "model": gate.get("model"),
            "effort": gate.get("effort"),
            "equity": gate.get("equity"),
            "cash": gate.get("cash"),
            "cash_frac": gate.get("cash_frac"),
            "order_count": gate.get("order_count", ep.get("planned_orders_count", 0)),
            "turnover_pct": gate.get("turnover_pct"),
            "expected_alpha_bps": gate.get("expected_alpha_bps", ep.get("expected_alpha_bps", 0.0)),
            "cost_bps": gate.get("cost_bps", ep.get("est_cost_bps", 0.0)),
            "net_bps": gate.get("net_bps", ep.get("net_edge_bps", 0.0)),
            "min_net_bps": gate.get("min_net_bps"),
            "proceed_gate": gate.get("proceed"),
            "force_onboard": gate.get("force_onboard"),
            "passes_net_bps": gate.get("passes_net_bps"),
            "risk_officer_approved": gate.get("risk_officer_approved"),
            "proceed_final": ep.get("proceed"),
            "reasons": reasons_str,
            "friendly_reasons": friendly_str,
        })
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=log.csv"
    return response


@app.route("/performance")
def performance():
    cfg = get_cfg()
    episodes, total = _load_episodes(cfg.EPISODES_PATH, limit=180)
    series = _performance_series(episodes)
    metrics = _dashboard_metrics(episodes, total_count=total)
    env = _environment_summary()
    return render_template("performance.html", series=series, metrics=metrics, active_page="performance", env=env)


@app.route("/run-now", methods=["POST"])
def run_now():
    status = "success"
    message = "Manual analysis completed successfully."

    try:
        logger.info("Manual run_once triggered via web interface")

        # Use gevent.Timeout for safe timeout handling in web workers
        with gevent.Timeout(120):  # Will raise TimeoutError if exceeded
            episode_result = run_once()

        episode_data, summary = _normalize_episode_result(episode_result)
        if not summary and episode_data:
            summary = _summarize_last_run(episode_data)
        if summary:
            message = "Manual analysis complete — {summary_text}".format(summary_text=summary.get("summary", "run recorded"))

        logger.info("Manual run_once completed successfully")

    except gevent.Timeout:
        status = "error"
        message = "Manual analysis timed out after 2 minutes."
        logger.error("Manual run_once timed out after 2 minutes")
        insert_log(
            "ERROR",
            "manual_run_timeout",
            {"message": message},
        )
    except Exception as exc:
        status = "error"
        message = f"Manual analysis failed: {exc}"
        logger.exception("Manual run_once failed")
        insert_log(
            "ERROR",
            "manual_run_failed",
            {"message": message},
        )

    return redirect(url_for("dashboard", run_status=status, run_message=message))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data: Dict[str, Any] = {}
        g = request.form.get
        data["AUTOMATION_ENABLED"]   = g("AUTOMATION_ENABLED")   == "on"
        data["DRY_RUN"]              = g("DRY_RUN")              == "on"
        data["ENABLE_EVENT_SCORE"]   = g("ENABLE_EVENT_SCORE")   == "on"
        data["ENABLE_COST_GATE"]     = g("ENABLE_COST_GATE")     == "on"
        data["ENABLE_VOL_TARGETING"] = g("ENABLE_VOL_TARGETING") == "on"
        data["TRADING_WINDOWS_ET"]   = g("TRADING_WINDOWS_ET", "10:05,14:35,16:35")
        data["WINDOW_TOL_MIN"]       = int(g("WINDOW_TOL_MIN", "30"))
        data["AVOID_NEAR_CLOSE_MIN"] = int(g("AVOID_NEAR_CLOSE_MIN", "10"))
        data["ENABLE_WEB_ADVISOR"]   = g("ENABLE_WEB_ADVISOR") == "on"
        data["WEB_ADVISOR_MODEL"]    = g("WEB_ADVISOR_MODEL", "gpt-5")
        data["WEB_ADVISOR_RECENCY_DAYS"] = int(g("WEB_ADVISOR_RECENCY_DAYS", "7"))
        data["WEB_ADVISOR_MAX_PAGES"] = int(g("WEB_ADVISOR_MAX_PAGES", "12"))
        allowlist = g("WEB_ADVISOR_DOMAIN_ALLOWLIST", "").strip()
        if allowlist:
            data["WEB_ADVISOR_DOMAIN_ALLOWLIST"] = allowlist
        data["ADVISOR_MAX_TRADES_PER_RUN"] = int(g("ADVISOR_MAX_TRADES_PER_RUN", "6"))
        data["MIN_HOLD_DAYS_BEFORE_SELL"] = int(g("MIN_HOLD_DAYS_BEFORE_SELL", "30"))
        data["UNIVERSE_MODE"]        = g("UNIVERSE_MODE", "etfs_only")
        data["UNIVERSE_MAX"]         = int(g("UNIVERSE_MAX", "450"))
        data["DATA_LOOKBACK_DAYS"]   = int(g("DATA_LOOKBACK_DAYS", "260"))
        data["MIN_BARS"]             = int(g("MIN_BARS", "60"))
        data["RESID_MOM_LOOKBACK"]   = int(g("RESID_MOM_LOOKBACK", "63"))
        data["TREND_FAST"]           = int(g("TREND_FAST", "20"))
        data["TREND_SLOW"]           = int(g("TREND_SLOW", "50"))
        data["REVERSAL_DAYS"]        = int(g("REVERSAL_DAYS", "3"))
        data["WINSOR_PCT"]           = float(g("WINSOR_PCT", "0.02"))
        data["NEWS_LOOKBACK_DAYS"]   = int(g("NEWS_LOOKBACK_DAYS", "7"))
        data["EVENT_TOP_K"]          = int(g("EVENT_TOP_K", "50"))
        data["EVENT_ALPHA_MULT"]     = float(g("EVENT_ALPHA_MULT", "1.0"))
        data["TARGET_PORTFOLIO_VOL"] = float(g("TARGET_PORTFOLIO_VOL", "0.22"))
        data["LAMBDA_RISK"]          = float(g("LAMBDA_RISK", "8.0"))
        data["TURNOVER_PENALTY"]     = float(g("TURNOVER_PENALTY", "0.0005"))
        data["NAME_MAX"]             = float(g("NAME_MAX", "0.20"))
        data["SECTOR_MAX"]           = float(g("SECTOR_MAX", "0.30"))
        data["REBALANCE_BAND"]       = float(g("REBALANCE_BAND", "0.25"))
        tp = g("TARGET_POSITIONS")
        if tp is not None and tp != "":
            data["TARGET_POSITIONS"] = int(tp)
        val_max_positions = g("MAX_POSITIONS")
        if val_max_positions is not None and val_max_positions != "":
            data["MAX_POSITIONS"] = int(val_max_positions)
        for key in [
            "VOL_TARGET_ANNUAL",
            "GROSS_EXPOSURE_FLOOR",
            "MIN_ORDER_NOTIONAL",
            "MIN_NET_BPS_TO_TRADE",
        ]:
            val = g(key)
            if val is not None and val != "":
                data[key] = float(val)
        data["CASH_BUFFER"]          = float(g("CASH_BUFFER", "0.00"))
        sx = float(g("SLEEVE_XSEC", "0.6"))
        se = float(g("SLEEVE_EVENT", "0.4"))
        sleeve_total = max(0.0, sx) + max(0.0, se)
        sleeve_total = sleeve_total if sleeve_total > 0 else 1.0
        data["SLEEVE_WEIGHTS"]       = {"xsec": max(0.0, sx)/sleeve_total, "event": max(0.0, se)/sleeve_total}
        data["MAX_SLICES"]           = int(g("MAX_SLICES", "5"))
        data["LIMIT_SLIP_BP"]        = int(g("LIMIT_SLIP_BP", "10"))
        data["COST_SPREAD_BPS"]      = float(g("COST_SPREAD_BPS", "5.0"))
        data["COST_IMPACT_KAPPA"]    = float(g("COST_IMPACT_KAPPA", "0.10"))
        data["COST_IMPACT_PSI"]      = float(g("COST_IMPACT_PSI", "0.5"))
        data["FILL_TIMEOUT_SEC"]     = int(g("FILL_TIMEOUT_SEC", "20"))
        data["ATR_STOP_MULT"]        = float(g("ATR_STOP_MULT", "2.5"))
        data["TAKE_PROFIT_ATR"]      = float(g("TAKE_PROFIT_ATR", "2.0"))
        data["TIME_STOP_DAYS"]       = int(g("TIME_STOP_DAYS", "10"))
        save_overrides(data)
        return redirect(url_for("settings"))
    cfg = get_cfg()
    env = _environment_summary()
    return render_template("settings.html", cfg=cfg, ov=load_overrides(), active_page="settings", env=env)


@app.route("/api/health", methods=["GET"])
def api_health():
    """Diagnostics endpoint showing broker connectivity and exposure."""

    try:
        account = alp_get_account()
        clock = alp_get_clock()
        positions = alp_list_positions()
        open_positions = [
            pos
            for pos in positions
            if float(getattr(pos, "qty", getattr(pos, "quantity", 0)) or 0.0) != 0.0
        ]
        return jsonify(
            {
                "ok": True,
                "equity": getattr(account, "equity", None),
                "cash": getattr(account, "cash", None),
                "open_positions": len(open_positions),
                "market_is_open": bool(getattr(clock, "is_open", False)),
                "next_open": str(getattr(clock, "next_open", "")),
                "next_close": str(getattr(clock, "next_close", "")),
            }
        ), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/logout")
def logout():
    env = _environment_summary()
    return render_template("logout.html", env=env, active_page="logout")


@app.route("/health")
def health():
    """Health check endpoint for deployment verification"""
    try:
        # Basic configuration check
        cfg = get_cfg()
        env = _environment_summary()
        
        # Return minimal health information (don't expose sensitive details)
        return jsonify({
            "ok": True,
            "status": "healthy",
            "timestamp": env["as_of"],
            "simulation_mode": env["simulated"]
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "ok": False,
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
