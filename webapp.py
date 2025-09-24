from __future__ import annotations
import os, json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
from v2.settings_bridge import get_cfg, save_overrides, load_overrides
from v2.orchestrator import run_once

app = Flask(__name__)

@app.route("/")
def root():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    cfg = get_cfg()
    episodes = []
    try:
        with open(cfg.EPISODES_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines()[-50:]:
                try:
                    episodes.append(json.loads(line))
                except Exception:
                    continue
        episodes.reverse()
    except FileNotFoundError:
        pass
    health = {
        "alpaca_paper": os.environ.get("ALPACA_PAPER","true"),
        "openai_present": bool(os.environ.get("OPENAI_API_KEY")),
        "as_of": datetime.utcnow().isoformat()+"Z"
    }
    return render_template("dashboard.html", cfg=cfg, episodes=episodes, health=health)

@app.route("/run-now", methods=["POST"])
def run_now():
    run_once()
    return redirect(url_for("dashboard"))

@app.route("/settings", methods=["GET","POST"])
def settings():
    if request.method == "POST":
        data = {}
        g = request.form.get
        # toggles
        data["AUTOMATION_ENABLED"]  = g("AUTOMATION_ENABLED")  == "on"
        data["DRY_RUN"]             = g("DRY_RUN")             == "on"
        data["ENABLE_EVENT_SCORE"]  = g("ENABLE_EVENT_SCORE")  == "on"
        data["ENABLE_COST_GATE"]    = g("ENABLE_COST_GATE")    == "on"
        data["ENABLE_VOL_TARGETING"]= g("ENABLE_VOL_TARGETING")== "on"
        # windows
        data["TRADING_WINDOWS_ET"]  = g("TRADING_WINDOWS_ET","10:05,14:35")
        data["WINDOW_TOL_MIN"]      = int(g("WINDOW_TOL_MIN","30"))
        data["AVOID_NEAR_CLOSE_MIN"]= int(g("AVOID_NEAR_CLOSE_MIN","10"))
        # universe
        data["UNIVERSE_MODE"]       = g("UNIVERSE_MODE","etfs_only")
        data["UNIVERSE_MAX"]        = int(g("UNIVERSE_MAX","450"))
        data["DATA_LOOKBACK_DAYS"]  = int(g("DATA_LOOKBACK_DAYS","260"))
        data["MIN_BARS"]            = int(g("MIN_BARS","60"))
        # features & events
        data["RESID_MOM_LOOKBACK"]  = int(g("RESID_MOM_LOOKBACK","63"))
        data["TREND_FAST"]          = int(g("TREND_FAST","20"))
        data["TREND_SLOW"]          = int(g("TREND_SLOW","50"))
        data["REVERSAL_DAYS"]       = int(g("REVERSAL_DAYS","3"))
        data["WINSOR_PCT"]          = float(g("WINSOR_PCT","0.02"))
        data["NEWS_LOOKBACK_DAYS"]  = int(g("NEWS_LOOKBACK_DAYS","7"))
        data["EVENT_TOP_K"]         = int(g("EVENT_TOP_K","50"))
        data["EVENT_ALPHA_MULT"]    = float(g("EVENT_ALPHA_MULT","1.0"))
        # optimizer & risk
        data["TARGET_PORTFOLIO_VOL"]= float(g("TARGET_PORTFOLIO_VOL","0.12"))
        data["LAMBDA_RISK"]         = float(g("LAMBDA_RISK","8.0"))
        data["TURNOVER_PENALTY"]    = float(g("TURNOVER_PENALTY","0.0005"))
        data["NAME_MAX"]            = float(g("NAME_MAX","0.20"))
        data["SECTOR_MAX"]          = float(g("SECTOR_MAX","0.30"))
        data["REBALANCE_BAND"]      = float(g("REBALANCE_BAND","0.25"))
        data["TARGET_POSITIONS"]    = int(g("TARGET_POSITIONS","10"))
        data["CASH_BUFFER"]         = float(g("CASH_BUFFER","0.00"))
        # sleeves
        sx = float(g("SLEEVE_XSEC","0.6")); se = float(g("SLEEVE_EVENT","0.4"))
        s  = max(0.0, sx) + max(0.0, se)
        if s <= 0: s = 1.0
        data["SLEEVE_WEIGHTS"]      = {"xsec": sx/s, "event": se/s}
        # execution & costs
        data["MIN_ORDER_NOTIONAL"]  = float(g("MIN_ORDER_NOTIONAL","25"))
        data["MAX_SLICES"]          = int(g("MAX_SLICES","5"))
        data["LIMIT_SLIP_BP"]       = int(g("LIMIT_SLIP_BP","10"))
        data["COST_SPREAD_BPS"]     = float(g("COST_SPREAD_BPS","5.0"))
        data["COST_IMPACT_KAPPA"]   = float(g("COST_IMPACT_KAPPA","0.10"))
        data["COST_IMPACT_PSI"]     = float(g("COST_IMPACT_PSI","0.5"))
        data["FILL_TIMEOUT_SEC"]    = int(g("FILL_TIMEOUT_SEC","20"))
        # stops
        data["ATR_STOP_MULT"]       = float(g("ATR_STOP_MULT","2.5"))
        data["TAKE_PROFIT_ATR"]     = float(g("TAKE_PROFIT_ATR","2.0"))
        data["TIME_STOP_DAYS"]      = int(g("TIME_STOP_DAYS","10"))
        save_overrides(data)
        return redirect(url_for("settings"))
    return render_template("settings.html", cfg=get_cfg(), ov=load_overrides())

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "alpaca_api_key_present": bool(os.environ.get("ALPACA_API_KEY")),
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "paper": os.environ.get("ALPACA_PAPER","true")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","8000")))
