from __future__ import annotations
import csv, json, os
from pathlib import Path
from datetime import datetime, timezone

FILLS_LOG = Path("logs/fills.csv"); FILLS_LOG.parent.mkdir(parents=True, exist_ok=True)
STATS_FILE = Path("data/slippage_stats.json"); STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load_stats():
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_stats(d):
    STATS_FILE.write_text(json.dumps(d, indent=2))

def record_fill(symbol: str, side: str, qty: float, arrival_mid: float, fill_price: float, adv_shares: float = 0.0):
    if not symbol or not arrival_mid or not fill_price:
        return
    # absolute mid slippage in bps
    try:
        slip_bps = abs((float(fill_price) - float(arrival_mid)) / float(arrival_mid)) * 1e4
    except Exception:
        return
    new = not FILLS_LOG.exists()
    with FILLS_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_iso","symbol","side","qty","arrival_mid","fill_price","slip_bps","adv"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    symbol.upper(), side, qty, arrival_mid, fill_price, f"{slip_bps:.4f}", adv_shares or ""])

    # Update EMA per symbol
    stats = _load_stats()
    s = stats.get(symbol.upper(), {"ema_bps": slip_bps, "count": 0})
    alpha = 0.2
    ema = float(s.get("ema_bps", slip_bps))
    ema = (1 - alpha) * ema + alpha * slip_bps
    cnt = int(s.get("count", 0)) + 1
    stats[symbol.upper()] = {"ema_bps": ema, "count": cnt, "updated_iso": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save_stats(stats)

def get_symbol_slippage_bps(symbol: str) -> float | None:
    d = _load_stats()
    rec = d.get((symbol or "").upper())
    try:
        return float(rec["ema_bps"]) if rec else None
    except Exception:
        return None
