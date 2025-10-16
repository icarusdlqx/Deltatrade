from __future__ import annotations

# --- START GPT-5 ENFORCER (medium effort, hard-coded prompt, logging) ---
try:
    from v2.agents_llm_enforce import apply_patch as _llm_enforce_apply

    _llm_enforce_apply()
    try:
        from v2.llm_gpt5 import score_events_gpt5

        score_events_gpt5(
            {"SMOKE": ["Scheduler startup health-check. Reply with JSON."]}
        )
    except Exception as _e:
        print("[llm_enforce] boot smoke skipped:", _e)
except Exception as _e:
    print("[llm_enforce] not applied:", _e)
# --- END GPT-5 ENFORCER ---

# --- START CODEX PATCH HOOK v3 (LLM failover + abs risk + dyn cost) ---
try:
    from v2.codex_patch_v3 import apply as _codex_apply_v3
    _codex_apply_v3()
    # Smoke test so you can see token usage immediately
    try:
        from v2.llm_client import smoke_test as _llm_smoke
        _llm_smoke()
    except Exception as _e:
        print("[codex_v3] LLM smoke skipped:", _e)
except Exception as _e:
    print("[codex_v3] not applied:", _e)
# --- END CODEX PATCH HOOK v3 ---

# --- START LLM BOOTSTRAP (prompt+logging+smoke+guard) ---

try:
    from v2.codex_bootstrap_llm import apply as _codex_apply_llm
    _codex_apply_llm()
except Exception as _e:
    print("[codex] LLM bootstrap not applied:", _e)
# --- END LLM BOOTSTRAP ---

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
import os, time, signal, json
import pytz
from datetime import datetime
from pathlib import Path
from alpaca.trading.client import TradingClient
from v2.settings_bridge import get_cfg
from v2.orchestrator import run_once
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

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


def _match_window_et(now, windows, tol_min):
    """
    Return the matched window string (e.g., '10:05') if now is within tol_min of it, else None.
    """
    et = now.astimezone(pytz.timezone("US/Eastern"))
    for hhmm in windows or []:
        try:
            h, m = map(int, str(hhmm).split(":"))
            target = et.replace(hour=h, minute=m, second=0, microsecond=0)
            if abs((et - target).total_seconds())/60.0 <= tol_min:
                return f"{h:02d}:{m:02d}"
        except Exception:
            continue
    return None


def _today_str(now_et=None):
    now_et = now_et or datetime.now(pytz.timezone("US/Eastern"))
    return now_et.strftime("%Y-%m-%d")


def _load_markers(path):
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_markers(markers, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(markers, indent=2), encoding="utf-8")


def _has_run_today(win_label, now_et, path):
    markers = _load_markers(path)
    ran = set(markers.get(_today_str(now_et), []))
    return str(win_label) in ran


def _mark_run_today(win_label, now_et, path):
    markers = _load_markers(path)
    key = _today_str(now_et)
    lst = list(markers.get(key, []))
    if str(win_label) not in lst:
        lst.append(str(win_label))
        markers[key] = lst
        _save_markers(markers, path)

def near_close_guard(trading_client, avoid_min):
    try:
        clock = trading_client.get_clock()
        if clock is None:
            return False

        if not getattr(clock, "is_open", False):
            # When the market is already closed we should not block automation.
            return False

        et_tz = pytz.timezone("US/Eastern")
        now = datetime.now(et_tz)

        next_close = getattr(clock, "next_close", None)
        if next_close is None:
            return False

        if next_close.tzinfo is None:
            next_close = pytz.utc.localize(next_close)
        next_close = next_close.astimezone(et_tz)

        minutes_to_close = (next_close - now).total_seconds() / 60.0
        return minutes_to_close <= avoid_min
    except Exception as e:
        logger.warning(f"Failed to check market close guard: {e}")
        return False

if __name__ == "__main__":
    logger.info("Deltatrade V1 scheduler initializing...")
    
    try:
        cfg = get_cfg()
        logger.info("Configuration loaded successfully")
        
        # Check if API keys are available
        alpaca_key = os.environ.get("ALPACA_API_KEY")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
        
        if alpaca_key and alpaca_secret:
            logger.info("Initializing Alpaca trading client...")
            tc = TradingClient(alpaca_key, alpaca_secret,
                               paper=os.environ.get("ALPACA_PAPER","true").lower() in ("true","1","yes","y"))
            logger.info("Deltatrade V1 scheduler starting with Alpaca API")
        else:
            # Run without trading client for simulated mode
            tc = None
            logger.info("Deltatrade V1 scheduler starting in simulated mode (no API keys)")
            
    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {e}")
        raise
    # Main scheduler loop with enhanced error handling
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            cfg = get_cfg()  # re-read overrides each loop
            now_et = datetime.now(pytz.timezone("US/Eastern"))
            win = _match_window_et(now_et, getattr(cfg, "TRADING_WINDOWS_ET", []), int(getattr(cfg, "WINDOW_TOL_MIN", 0)))
            once_only = bool(getattr(cfg, "RUN_ONCE_PER_WINDOW", True))
            markers_path = getattr(cfg, "RUN_MARKERS_PATH", "data/run_markers.json")
            should_run = bool(cfg.AUTOMATION_ENABLED) and bool(win) and (
                not once_only or not _has_run_today(win, now_et, markers_path)
            )

            # Block if too close to market close (when live TradingClient is present)
            if should_run and (tc and near_close_guard(tc, int(cfg.AVOID_NEAR_CLOSE_MIN))):
                should_run = False

            if should_run:
                logger.info(f"Trading window '{win}' active, executing run_once...")

                # Set up timeout for run_once (5 minutes max)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minute timeout

                try:
                    ep = run_once()
                    signal.alarm(0)  # Cancel timeout
                    logger.info(
                        f"Episode completed: {ep.get('as_of')} proceed={ep.get('proceed')} orders={len(ep.get('orders_submitted',[]))}"
                    )
                    # Mark this window as completed for today to prevent additional runs in the tolerance band
                    if once_only and win:
                        _mark_run_today(win, now_et, markers_path)
                    consecutive_errors = 0  # Reset error counter on success
                    # Sleep past the current minute to avoid double-trigger in the same window
                    time.sleep(65)

                except TimeoutException:
                    signal.alarm(0)
                    logger.error("run_once() timed out after 5 minutes")
                    consecutive_errors += 1

                except Exception as e:
                    signal.alarm(0)
                    logger.error(f"Run error: {e}")
                    consecutive_errors += 1

            # Check for too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Too many consecutive errors ({consecutive_errors}), entering safe mode")
                time.sleep(300)  # Wait 5 minutes before retrying
                consecutive_errors = 0

            # Short idle sleep; loop quickly to catch windows
            time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(30)  # Wait before retrying
