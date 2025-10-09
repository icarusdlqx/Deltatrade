from __future__ import annotations
import csv, os
from pathlib import Path
from datetime import datetime, timedelta, timezone

LOG_DIR = Path("data")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = LOG_DIR / "analysis_log.csv"
ITEM_LOG = LOG_DIR / "analysis_items.csv"

def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _ensure_headers():
    if not RUN_LOG.exists():
        with RUN_LOG.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_iso","slot","status","reason","turnover","expected_bps","cost_bps","net_bps","approved","n_orders","n_buy","n_sell","n_hold"])
    if not ITEM_LOG.exists():
        with ITEM_LOG.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_iso","slot","symbol","action","target_w","current_w","delta_w","expected_bps","cost_bps","net_bps"])

def log_run(status: str, slot: str = "", reason: str = "", turnover: float = None,
            expected_bps: float = None, cost_bps: float = None, net_bps: float = None,
            approved: bool = None, n_orders: int = None, n_buy: int = None, n_sell: int = None, n_hold: int = None):
    _ensure_headers()
    with RUN_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([_utcnow_iso(), slot, status, reason,
                    _fmt(turnover), _fmt(expected_bps), _fmt(cost_bps), _fmt(net_bps),
                    _fmt(int(approved) if approved is not None else None),
                    _fmt(n_orders), _fmt(n_buy), _fmt(n_sell), _fmt(n_hold)])

def log_items(slot: str, rows):
    if not rows:
        return
    _ensure_headers()
    with ITEM_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        ts = _utcnow_iso()
        for r in rows:
            w.writerow([ts, slot,
                        r.get("symbol") or r.get("ticker") or r.get("asset"),
                        r.get("action"),
                        _fmt(r.get("target_w")), _fmt(r.get("current_w")), _fmt(r.get("delta_w")),
                        _fmt(r.get("expected_bps")), _fmt(r.get("cost_bps")), _fmt(r.get("net_bps"))])

def last_actions_map(lookback_days: int = 7):
    """Return dict[symbol] -> last recorded action within lookback window."""
    out = {}
    if not ITEM_LOG.exists():
        return out
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    with ITEM_LOG.open("r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                ts = datetime.fromisoformat(row["ts_iso"].replace("Z","")).replace(tzinfo=None)
            except Exception:
                ts = cutoff
            if ts < cutoff:
                continue
            sym = row["symbol"]
            out[sym] = row["action"]
    return out

def _fmt(x):
    return "" if x is None else x
