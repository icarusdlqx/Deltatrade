from __future__ import annotations
import os
from datetime import datetime, timezone
from .agents_patch import last_ok_iso

def must_have_recent_event_assessment() -> bool:
    if os.getenv("REQUIRE_EVENT_ASSESSMENT","0") != "1":
        return True
    max_age_min = int(os.getenv("EVENT_MAX_AGE_MIN","90"))
    ts = last_ok_iso()
    if not ts: return False
    try:
        t = datetime.fromisoformat(ts.replace("Z",""))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        else:
            t = t.astimezone(timezone.utc)
    except Exception:
        return False
    delta = (datetime.now(timezone.utc) - t).total_seconds()/60.0
    return delta <= max_age_min
