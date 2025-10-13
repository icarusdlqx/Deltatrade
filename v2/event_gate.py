from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LAST_FILE = Path("data/last_event_assessment.json")
LAST_FILE.parent.mkdir(parents=True, exist_ok=True)


def record_assessment(ts: Optional[datetime] = None) -> None:
    """Persist the timestamp of the latest successful event assessment."""
    try:
        when = ts or datetime.now(timezone.utc)
        LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_FILE.write_text(
            json.dumps({"ts_iso": when.isoformat(timespec="seconds")})
        )
    except Exception:
        # Failing to persist the timestamp should never break trading flow.
        pass


def load_last_assessment_iso() -> Optional[str]:
    """Return the ISO timestamp of the last saved assessment, if present."""
    try:
        if not LAST_FILE.exists():
            return None
        data = json.loads(LAST_FILE.read_text())
        ts = data.get("ts_iso")
        if isinstance(ts, str) and ts:
            return ts
    except Exception:
        return None
    return None
