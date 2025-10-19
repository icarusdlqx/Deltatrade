# infra/health_gate.py
import json, pathlib
from datetime import datetime, timezone, timedelta

HB = pathlib.Path("logs/heartbeat.ndjson")


def api_healthy(window_min=5, max_err_rate=0.1, max_p95_ms=5000, min_samples=5):
    if not HB.exists(): return False, "no_heartbeat"
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    latencies, errors, total = [], 0, 0
    for ln in HB.read_text(encoding="utf-8").splitlines()[-400:]:
        if not ln.strip(): continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        try:
            ts = datetime.fromisoformat(rec["ts"])
        except Exception:
            continue
        if ts < cutoff: continue
        total += 1
        if rec.get("status") != "ok": errors += 1
        lm = rec.get("latency_ms")
        if isinstance(lm, int): latencies.append(lm)
    if total < min_samples: return False, "insufficient_samples"
    err_rate = errors / total
    p95 = sorted(latencies)[int(0.95*len(latencies))-1] if latencies else None
    if err_rate > max_err_rate: return False, f"high_err_rate:{err_rate:.2f}"
    if p95 and p95 > max_p95_ms: return False, f"high_p95:{p95}ms"
    return True, "ok"


def log_gate_decision(ok: bool, why: str):
    from .heartbeat import write_heartbeat
    write_heartbeat("gate", "pre_trade", "CHECK", "ok" if ok else "error",
                    latency_ms=0, req_meta={"window":"pre_cycle"}, resp_meta={"why":why})
