from __future__ import annotations

# ===== Canonical defaults =====

# Automation & windows
AUTOMATION_ENABLED = True
TRADING_WINDOWS_ET = ["10:05", "14:35"]
WINDOW_TOL_MIN = 30
AVOID_NEAR_CLOSE_MIN = 10

# Universe
UNIVERSE_MODE = "etfs_only"   # "etfs_only" (default) or "sp500_etfs" if data/sp500.csv present
UNIVERSE_MAX = 450
DATA_LOOKBACK_DAYS = 260
MIN_BARS = 60

# Features
RESID_MOM_LOOKBACK = 63
TREND_FAST = 20
TREND_SLOW = 50
REVERSAL_DAYS = 3
WINSOR_PCT = 0.02

# Event scoring (AI)
ENABLE_EVENT_SCORE = True
NEWS_LOOKBACK_DAYS = 7
EVENT_TOP_K = 50
EVENT_BPS_PER_DAY = {"low": 3, "med": 7, "high": 12}
EVENT_ALPHA_MULT = 1.00

# Optimizer / risk
ENABLE_VOL_TARGETING = True
TARGET_PORTFOLIO_VOL = 0.12
LAMBDA_RISK = 8.0
TURNOVER_PENALTY = 0.0005
NAME_MAX = 0.20
SECTOR_MAX = 0.30
REBALANCE_BAND = 0.25
TARGET_POSITIONS = 10
CASH_BUFFER = 0.00

# Sleeves
SLEEVES = ["xsec", "event"]
SLEEVE_WEIGHTS = {"xsec": 0.60, "event": 0.40}

# Execution & costs
ENABLE_COST_GATE = True
DRY_RUN = False
MIN_ORDER_NOTIONAL = 25.0
MAX_SLICES = 5
LIMIT_SLIP_BP = 10
COST_SPREAD_BPS = 5.0
COST_IMPACT_KAPPA = 0.10
COST_IMPACT_PSI = 0.5
FILL_TIMEOUT_SEC = 20

# Stops / TTL (stored; not auto-placed)
ATR_STOP_MULT = 2.5
TAKE_PROFIT_ATR = 2.0
TIME_STOP_DAYS = 10

# Paths
EPISODES_PATH = "data/episodes_v2.jsonl"
STATE_PATH = "data/state.json"
SETTINGS_OVERRIDES_PATH = "data/settings_overrides.json"

# API & model
OPENAI_MODEL = "gpt-5"
OPENAI_REASONING_EFFORT = "medium"
