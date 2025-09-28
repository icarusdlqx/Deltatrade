from __future__ import annotations
import os, time, signal
import pytz
from datetime import datetime
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
            now = datetime.now(pytz.timezone("US/Eastern"))
            
            if cfg.AUTOMATION_ENABLED and in_window_et(now, cfg.TRADING_WINDOWS_ET, int(cfg.WINDOW_TOL_MIN)) and not (tc and near_close_guard(tc, int(cfg.AVOID_NEAR_CLOSE_MIN))):
                logger.info("Trading window active, executing run_once...")
                
                # Set up timeout for run_once (5 minutes max)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minute timeout
                
                try:
                    ep = run_once()
                    signal.alarm(0)  # Cancel timeout
                    logger.info(f"Episode completed: {ep.get('as_of')} proceed={ep.get('proceed')} orders={len(ep.get('orders_submitted',[]))}")
                    consecutive_errors = 0  # Reset error counter on success
                    
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
                    
                time.sleep(60)
            else:
                logger.debug(f"Trading window inactive, sleeping... (automation_enabled={cfg.AUTOMATION_ENABLED})")
                time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(30)  # Wait before retrying
