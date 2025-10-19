# infra/heartbeat.py
import time, json, os, threading
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.getenv("HEARTBEAT_LOG_PATH", "logs/heartbeat.ndjson"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_heartbeat(api_type, endpoint, method="POST", status="ok",
                    latency_ms=None, req_meta=None, resp_meta=None):
    rec = {
        "ts": _now_iso(),
        "api_type": api_type,
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "latency_ms": latency_ms,
        "req": req_meta or {},
        "resp": resp_meta or {},
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "pid": os.getpid(),
    }
    line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
    with _lock:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def wrap_heartbeat(api_type, endpoint, method="POST", req_meta_fn=lambda *a, **k: {}):
    def deco(fn):
        def inner(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                req_meta = req_meta_fn(*args, **kwargs) or {}
            except Exception:
                req_meta = {}
            status, http_code, tokens_in, tokens_out, err = "ok", None, None, None, None
            try:
                res = fn(*args, **kwargs)
                http_code = getattr(res, "status_code", None) or getattr(res, "status", None)
                usage = getattr(res, "usage", None) or {}
                tokens_in  = usage.get("prompt_tokens")
                tokens_out = usage.get("completion_tokens")
                return res
            except Exception as e:
                status = "error"; err = str(e)[:400]
                raise
            finally:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                write_heartbeat(api_type, endpoint, method, status, latency_ms,
                                req_meta=req_meta,
                                resp_meta={"http_code": http_code, "tokens_in": tokens_in,
                                           "tokens_out": tokens_out, "err": err})
        return inner
    return deco


def tail_heartbeat(n=50, status_only=False):
    try:
        lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()[-n:]
    except FileNotFoundError:
        print("No heartbeat yet."); return
    for ln in lines:
        if not ln.strip(): continue
        rec = json.loads(ln)
        if status_only:
            print(rec["ts"], rec["api_type"], rec["endpoint"], rec["status"], f'{rec.get("latency_ms","?")}ms')
        else:
            print(ln)
