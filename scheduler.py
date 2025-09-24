from __future__ import annotations
import os, time
import pytz
from datetime import datetime
from alpaca.trading.client import TradingClient
from v2.settings_bridge import get_cfg
from v2.orchestrator import run_once

def in_window_et(now, windows, tol_min):
    et = now.astimezone(pytz.timezone("US/Eastern"))
    for hhmm in windows:
        try:
            h, m = map(int, str(hhmm).split(":"))
            target = et.replace(hour=h, minute=m, second=0, microsecond=0)
            if abs((et - target).total_seconds())/60.0 <= tol_min:
                return True
        except Exception:
            continue
    return False

def near_close_guard(trading_client, avoid_min):
    try:
        clock = trading_client.get_clock()
        if not clock.is_open:
            return True
        now = datetime.now(pytz.utc)
        return (clock.next_close - now).total_seconds()/60.0 <= avoid_min
    except Exception:
        return False

if __name__ == "__main__":
    cfg = get_cfg()
    
    # Check if API keys are available
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
    
    if alpaca_key and alpaca_secret:
        tc = TradingClient(alpaca_key, alpaca_secret,
                           paper=os.environ.get("ALPACA_PAPER","true").lower() in ("true","1","yes","y"))
        print("Deltatrade V1 scheduler starting with Alpaca API…")
    else:
        # Run without trading client for simulated mode
        tc = None
        print("Deltatrade V1 scheduler starting in simulated mode (no API keys)…")
    while True:
        cfg = get_cfg()  # re-read overrides each loop
        now = datetime.now(pytz.utc)
        if cfg.AUTOMATION_ENABLED and in_window_et(now, cfg.TRADING_WINDOWS_ET, int(cfg.WINDOW_TOL_MIN)) and not (tc and near_close_guard(tc, int(cfg.AVOID_NEAR_CLOSE_MIN))):
            try:
                ep = run_once()
                print("Episode:", ep.get("as_of"), "proceed=", ep.get("proceed"), "orders=", len(ep.get("orders_submitted",[])))
            except Exception as e:
                print("Run error:", e)
            time.sleep(60)
        else:
            time.sleep(15)
