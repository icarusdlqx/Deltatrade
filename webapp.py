from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from flask import Flask, jsonify, redirect, render_template, request, url_for

from alpaca.trading.client import TradingClient

from v2.orchestrator import run_once
from v2.settings_bridge import get_cfg, load_overrides, save_overrides


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
        "as_of": datetime.utcnow().isoformat() + "Z",
    }


def _load_episodes(path: str, limit: int = 200) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return episodes
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return episodes
    for line in lines[-limit:]:
        try:
            episodes.append(json.loads(line))
        except Exception:
            continue
    episodes.reverse()
    return episodes


def _dashboard_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {
            "total_runs": 0,
            "proceed_rate": 0,
            "avg_expected": 0,
            "avg_cost": 0,
            "net_edge": 0,
        }
    total_runs = len(episodes)
    proceed = sum(1 for ep in episodes if ep.get("proceed"))
    expected_vals = [float(ep.get("expected_alpha_bps", 0.0)) for ep in episodes]
    cost_vals = [float(ep.get("est_cost_bps", 0.0)) for ep in episodes]
    net_edge = sum(e - c for e, c in zip(expected_vals, cost_vals)) / 10000.0
    return {
        "total_runs": total_runs,
        "proceed_rate": int(round(100 * proceed / total_runs)) if total_runs else 0,
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


def _performance_series(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: List[str] = []
    expected: List[float] = []
    costs: List[float] = []
    cumulative: List[float] = []
    total = 0.0
    for ep in reversed(episodes):
        labels.append(ep.get("as_of", ""))
        exp = float(ep.get("expected_alpha_bps", 0.0))
        cost = float(ep.get("est_cost_bps", 0.0))
        expected.append(exp)
        costs.append(cost)
        total += (exp - cost) / 10000.0
        cumulative.append(total)
    labels.reverse(); expected.reverse(); costs.reverse(); cumulative.reverse()
    return {"labels": labels, "expected": expected, "costs": costs, "cumulative": cumulative}


@app.context_processor
def inject_nav() -> Dict[str, Any]:
    return {"nav_items": NAV_ITEMS, "current_year": datetime.utcnow().year}


@app.route("/")
def root():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    cfg = get_cfg()
    episodes = _load_episodes(cfg.EPISODES_PATH, limit=120)
    latest = episodes[0] if episodes else None
    env = _environment_summary()
    metrics = _dashboard_metrics(episodes)
    return render_template("dashboard.html", cfg=cfg, episodes=episodes[:5], latest=latest,
                           env=env, metrics=metrics, active_page="dashboard")


@app.route("/positions")
def positions():
    env = _environment_summary()
    snapshot = _positions_snapshot(env)
    return render_template("positions.html", snapshot=snapshot, env=env, active_page="positions")


@app.route("/log")
def log():
    cfg = get_cfg()
    episodes = _load_episodes(cfg.EPISODES_PATH, limit=300)
    env = _environment_summary()
    return render_template("log.html", episodes=episodes, active_page="log", env=env)


@app.route("/performance")
def performance():
    cfg = get_cfg()
    episodes = _load_episodes(cfg.EPISODES_PATH, limit=180)
    series = _performance_series(episodes)
    metrics = _dashboard_metrics(episodes)
    env = _environment_summary()
    return render_template("performance.html", series=series, metrics=metrics, active_page="performance", env=env)


@app.route("/run-now", methods=["POST"])
def run_now():
    run_once()
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
        data["TARGET_PORTFOLIO_VOL"] = float(g("TARGET_PORTFOLIO_VOL", "0.12"))
        data["LAMBDA_RISK"]          = float(g("LAMBDA_RISK", "8.0"))
        data["TURNOVER_PENALTY"]     = float(g("TURNOVER_PENALTY", "0.0005"))
        data["NAME_MAX"]             = float(g("NAME_MAX", "0.20"))
        data["SECTOR_MAX"]           = float(g("SECTOR_MAX", "0.30"))
        data["REBALANCE_BAND"]       = float(g("REBALANCE_BAND", "0.25"))
        data["TARGET_POSITIONS"]     = int(g("TARGET_POSITIONS", "10"))
        data["CASH_BUFFER"]          = float(g("CASH_BUFFER", "0.00"))
        sx = float(g("SLEEVE_XSEC", "0.6"))
        se = float(g("SLEEVE_EVENT", "0.4"))
        sleeve_total = max(0.0, sx) + max(0.0, se)
        sleeve_total = sleeve_total if sleeve_total > 0 else 1.0
        data["SLEEVE_WEIGHTS"]       = {"xsec": max(0.0, sx)/sleeve_total, "event": max(0.0, se)/sleeve_total}
        data["MIN_ORDER_NOTIONAL"]   = float(g("MIN_ORDER_NOTIONAL", "25"))
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
    env = _environment_summary()
    return jsonify({
        "ok": True,
        "alpaca_api_key_present": env["alpaca_key"],
        "openai_api_key_present": env["openai_key"],
        "paper": env["paper"],
        "simulated": env["simulated"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
