from __future__ import annotations

import csv
import io
import json
import os
import logging
import gevent
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple
import pytz

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from v2.config import MAX_LOG_ROWS
from v2.orchestrator import run_once
from v2.settings_bridge import get_cfg, load_overrides, save_overrides

# Configure logging for webapp
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "deltatrade-demo")



NAV_ITEMS = [
    {"endpoint": "dashboard", "label": "Dashboard", "icon": "bi-speedometer2"},
    {"endpoint": "positions", "label": "Positions", "icon": "bi-graph-up"},
    {"endpoint": "log", "label": "Log", "icon": "bi-journal-text"},
    {"endpoint": "performance", "label": "Performance", "icon": "bi-bar-chart"},
    {"endpoint": "settings", "label": "Settings", "icon": "bi-sliders"},
    {"endpoint": "logout", "label": "Logout", "icon": "bi-box-arrow-right"},
]


def _environment_summary() -> Dict[str, Any]:
    alpaca_key = bool(os.environ.get("ALPACA_API_KEY"))
    alpaca_secret = bool(os.environ.get("ALPACA_SECRET_KEY"))
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
    reasons_raw = gate.get("reasons") or []
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
    if traded:
        summary = "Traded {count} orders · net {net:.1f} bps after {cost:.1f} bps costs · turnover {turnover:.1f}%".format(
            count=len(actual_orders), net=net, cost=cost, turnover=turnover_pct
        )
        plain = "Placed {count} orders; turnover {turnover:.1f}%; exp {expected:.1f} bps → net {net:.1f} bps.".format(
            count=len(actual_orders), turnover=turnover_pct, expected=expected, net=net
        )
    else:
        reason_text = ", ".join(reasons) if reasons else "none"
        summary = (
            "Skipped — reasons: {reasons}; gate math: exp {expected:.1f} bps, cost {cost:.1f} bps, net {net:.1f} bps"
        ).format(reasons=reason_text, expected=expected, cost=cost, net=net)
        plain = (
            "Skipped — reasons: {reasons}. Gate: exp {expected:.1f}, cost {cost:.1f}, net {net:.1f} bps."
        ).format(reasons=reason_text, expected=expected, cost=cost, net=net)
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

    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        return {"positions": [], "cash": 0.0, "equity": 0.0, "error": "Alpaca credentials not configured.", "simulated": False}

    try:
        paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
        client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper)
        positions = []
        for pos in client.get_all_positions():
            positions.append({
                "symbol": pos.symbol,
                "qty": float(getattr(pos, "qty", getattr(pos, "quantity", 0.0))),
                "avg_price": float(getattr(pos, "avg_entry_price", 0.0)),
                "current_price": float(getattr(pos, "current_price", 0.0)),
                "market_value": float(getattr(pos, "market_value", 0.0)),
            })
        account = client.get_account()
        return {
            "positions": positions,
            "cash": float(getattr(account, "cash", 0.0)),
            "equity": float(getattr(account, "equity", 0.0)),
            "simulated": False,
        }
    except Exception as exc:
        return {"positions": [], "cash": 0.0, "equity": 0.0, "error": str(exc), "simulated": False}


def _trading_client(env: Dict[str, Any]):
    if env.get("simulated"):
        from v2.simulated_clients import SimStockHistoricalDataClient, SimTradingClient

        data_client = SimStockHistoricalDataClient()
        return SimTradingClient(data_client)

    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        raise RuntimeError("Alpaca credentials not configured.")

    paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes", "y")
    return TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper)


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
        active_page="dashboard",
    )


@app.route("/positions")
def positions():
    env = _environment_summary()
    snapshot = _positions_snapshot(env)
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


@app.route("/positions/sell", methods=["POST"])
def sell_position():
    env = _environment_summary()
    symbol = (request.form.get("symbol") or "").strip().upper()
    if not symbol:
        return redirect(url_for("positions", status="error", message="No symbol provided."))

    snapshot = _positions_snapshot(env)
    position = next(
        (p for p in snapshot.get("positions", []) if str(p.get("symbol", "")).upper() == symbol),
        None,
    )
    if not position:
        return redirect(url_for("positions", status="error", message=f"Position {symbol} not found."))

    qty = abs(float(position.get("qty") or 0.0))
    if qty <= 0:
        return redirect(url_for("positions", status="error", message=f"Position {symbol} has no quantity to sell."))

    qty = round(qty, 6)

    try:
        client = _trading_client(env)
    except Exception as exc:
        logger.exception("Failed to initialise trading client for sell %s", symbol)
        return redirect(url_for("positions", status="error", message=f"Unable to sell {symbol}: {exc}"))

    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        client.submit_order(req)
    except Exception as exc:
        logger.exception("Error submitting sell order for %s", symbol)
        return redirect(url_for("positions", status="error", message=f"Failed to sell {symbol}: {exc}"))

    message = f"Submitted sell order for {symbol} ({qty:.4f} shares)."
    return redirect(url_for("positions", status="success", message=message))


@app.route("/positions/sell_all", methods=["POST"])
def sell_all_positions():
    env = _environment_summary()
    snapshot = _positions_snapshot(env)
    positions = [p for p in snapshot.get("positions", []) if abs(float(p.get("qty") or 0.0)) > 0]

    if not positions:
        return redirect(url_for("positions", status="success", message="No positions to sell."))

    try:
        client = _trading_client(env)
    except Exception as exc:
        logger.exception("Failed to initialise trading client for sell all")
        return redirect(url_for("positions", status="error", message=f"Unable to sell positions: {exc}"))

    failures: List[str] = []
    for pos in positions:
        symbol = str(pos.get("symbol", "")).upper()
        qty = round(abs(float(pos.get("qty") or 0.0)), 6)
        if qty <= 0:
            continue
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            client.submit_order(req)
        except Exception:
            logger.exception("Error submitting sell order for %s", symbol)
            failures.append(symbol)

    if failures:
        failed_list = ", ".join(failures)
        return redirect(
            url_for(
                "positions",
                status="error",
                message=f"Failed to sell: {failed_list}.",
            )
        )

    return redirect(url_for("positions", status="success", message="Submitted sell orders for all positions."))


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
    try:
        logger.info("Manual run_once triggered via web interface")
        
        # Use gevent.Timeout for safe timeout handling in web workers
        with gevent.Timeout(120):  # Will raise TimeoutError if exceeded
            run_once()
        
        logger.info("Manual run_once completed successfully")
            
    except gevent.Timeout:
        logger.error("Manual run_once timed out after 2 minutes")
    except Exception as e:
        logger.error(f"Manual run_once failed: {e}")
        
    return redirect(url_for("dashboard"))


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
        data["TRADING_WINDOWS_ET"]   = g("TRADING_WINDOWS_ET", "10:05,14:35")
        data["WINDOW_TOL_MIN"]       = int(g("WINDOW_TOL_MIN", "30"))
        data["AVOID_NEAR_CLOSE_MIN"] = int(g("AVOID_NEAR_CLOSE_MIN", "10"))
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
        data["TARGET_POSITIONS"]     = int(g("TARGET_POSITIONS", "10"))
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
