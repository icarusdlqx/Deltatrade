from __future__ import annotations
import os, json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback: naive times

STATE = Path("data/last_run.json")
STATE.parent.mkdir(parents=True, exist_ok=True)

def _tz():
    tzname = os.getenv("RUN_TZ", "America/New_York")
    return ZoneInfo(tzname) if ZoneInfo else None

def _now():
    z = _tz()
    return datetime.now(z) if z else datetime.utcnow()

def _slots_for_today():
    now = _now()
    base = now.replace(hour=0, minute=0, second=0, microsecond=0)
    times = [s.strip() for s in os.getenv("RUN_TIMES_LOCAL", "10:00,13:30,15:50").split(",") if s.strip()]
    slots = []
    for t in times:
        hh, mm = [int(x) for x in t.split(":")]
        slots.append(base.replace(hour=hh, minute=mm))
    return slots

def _load_state(today_str: str):
    if STATE.exists():
        try:
            st = json.loads(STATE.read_text())
        except Exception:
            st = {}
    else:
        st = {}
    if st.get("date") != today_str:
        st = {"date": today_str, "done": []}
    return st

def _save_state(st):
    STATE.write_text(json.dumps(st))

@dataclass
class ScheduleDecision:
    run_now: bool
    slot_label: str
    reason: str

def should_run_now(mark: bool = True) -> ScheduleDecision:
    """
    Returns whether we are inside a run window and the slot hasn't executed yet.
    If mark=True and run_now=True, marks the slot as executed immediately (to prevent duplicates).
    """
    win_min = int(os.getenv("RUN_WINDOW_MIN", "10"))
    now = _now()
    today_str = (now.date()).isoformat()
    slots = _slots_for_today()
    st = _load_state(today_str)
    for s in slots:
        label = f"{today_str}_{s.strftime('%H:%M')}"
        if label in st.get("done", []):
            continue
        delta = abs((now - s).total_seconds()) / 60.0
        if delta <= win_min:
            if mark:
                st["done"].append(label)
                _save_state(st)
            return ScheduleDecision(True, label, "inside_window")
    # not in any window
    return ScheduleDecision(False, "", "outside_window")
