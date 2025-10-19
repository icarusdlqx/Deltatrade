# codex_one_click_heartbeat_patch.py
# Run once from your repo root: python codex_one_click_heartbeat_patch.py
import os, re, json, time, shutil
from pathlib import Path

ROOT = Path.cwd()


def write_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = p.read_text(encoding="utf-8")
        if old.strip() == content.strip():
            print(f"[=] Unchanged {p}")
            return
    p.write_text(content, encoding="utf-8")
    print(f"[+] Wrote {p}")


def safe_insert_before_def(path: Path, import_line: str, deco_line: str, def_name_hint: str):
    """
    Adds import if missing; adds decorator above the first def *hint* match.
    Idempotent: skips if decorator already present.
    """
    if not path.exists():
        print(f"[!] Skip: {path} not found")
        return

    txt = path.read_text(encoding="utf-8")
    changed = False

    # import
    if import_line not in txt:
        txt = import_line + "\n" + txt
        changed = True

    # already decorated?
    if deco_line.strip() in txt:
        pass
    else:
        # find a def line with hint in name
        pat = re.compile(rf"^(\s*)def\s+([a-zA-Z0-9_]*{re.escape(def_name_hint)}[a-zA-Z0-9_]*)\s*\(", re.M)
        m = pat.search(txt)
        if m:
            indent = m.group(1)
            inject = f"{indent}{deco_line}\n"
            start = m.start()
            # insert decorator right before def
            # find start of line
            bol = txt.rfind("\n", 0, start) + 1
            txt = txt[:bol] + inject + txt[bol:]
            changed = True
        else:
            print(f"[!] Could not find a '{def_name_hint}' function in {path}; skipping decorator.")

    if changed:
        path.write_text(txt, encoding="utf-8")
        print(f"[~] Patched {path}")
    else:
        print(f"[=] No changes to {path}")


def add_reasoning_effort_medium(path: Path, model_key_hint="model", effort="medium"):
    """
    Best-effort: add reasoning={'effort':'medium'} to OpenAI chat/completions calls.
    Looks for 'client.chat.completions.create(' or 'openai.chat.completions.create(' blocks.
    """
    if not path.exists():
        print(f"[!] Skip: {path} not found")
        return
    txt = path.read_text(encoding="utf-8")
    if "reasoning={'effort':'medium'}" in txt or '"effort":"medium"' in txt:
        print(f"[=] Reasoning effort already present in {path}")
        return

    # naive insertion just after the opening parenthesis of create(
    patterns = [
        r"(\.chat\.completions\.create\(\s*)",
        r"(\.responses\.create\(\s*)"  # in case using new Responses API
    ]
    changed = False
    for pat in patterns:
        txt_new, n = re.subn(pat, r"\1reasoning={'effort':'medium'}, ", txt)
        if n > 0:
            txt = txt_new
            changed = True

    if changed:
        path.write_text(txt, encoding="utf-8")
        print(f"[~] Added reasoning effort=medium in {path}")
    else:
        print(f"[!] Could not detect OpenAI create(...) calls in {path}; skipped effort injection.")


def add_pretrade_gate_to_runner(path: Path):
    """
    Inserts a pre-trade health check gate call at top of each cycle tick.
    Looks for functions containing 'run_cycle' or 'run_trading_cycle' or 'main'.
    """
    if not path.exists():
        print(f"[!] Skip: {path} not found")
        return
    txt = path.read_text(encoding="utf-8")
    import_line = "from infra.health_gate import api_healthy, log_gate_decision"
    if import_line not in txt:
        txt = import_line + "\n" + txt

    if "api_healthy(" in txt and "log_gate_decision(" in txt:
        print(f"[=] Health gate already present in {path}")
        path.write_text(txt, encoding="utf-8")
        return

    # inject at start of first function that looks like a trading cycle
    func_pat = re.compile(r"^(\s*)def\s+(run_(trading_)?cycle|main|tick)\s*\([^\)]*\)\s*:", re.M)
    m = func_pat.search(txt)
    if not m:
        print(f"[!] Could not find a cycle function in {path}; skipped gate injection.")
        path.write_text(txt, encoding="utf-8")
        return

    indent = m.group(1)
    body_start = m.end()
    # Find insertion point: first non-empty line after the def:
    nl = txt.find("\n", body_start)
    insert_pos = nl + 1 if nl != -1 else body_start

    gate_code = f"""{indent}    ok, why = api_healthy(window_min=5, max_err_rate=0.1, max_p95_ms=5000)
{indent}    log_gate_decision(ok, why)
{indent}    if not ok:
{indent}        # Fail closed this cycle
{indent}        return {{'status':'skipped','reason':why}}
"""
    txt = txt[:insert_pos] + gate_code + txt[insert_pos:]
    path.write_text(txt, encoding="utf-8")
    print(f"[~] Injected pre-trade health gate into {path}")


HEARTBEAT_PY = r'''# infra/heartbeat.py
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
'''


HEALTH_GATE_PY = r'''# infra/health_gate.py
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
'''


SMOKE_PY = r'''# scripts/heartbeat_smoke.py
"""
Quick smoke: pings GPT-5 (dry prompt) and Alpaca (account/status).
Run: python scripts/heartbeat_smoke.py
Requires your usual env vars for OpenAI + Alpaca.
"""
import os, time
from infra.heartbeat import wrap_heartbeat

# Optional: you can swap this to your actual clients.
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

import requests


@wrap_heartbeat("openai","chat.completions", req_meta_fn=lambda **k: {"model":k.get("model")})
def ping_gpt(model="gpt-5", messages=None, **opts):
    if client is None: raise RuntimeError("OpenAI client not available")
    messages = messages or [{"role":"user","content":"Health check. Reply 'pong'."}]
    # Medium reasoning effort
    return client.chat.completions.create(model=model, messages=messages, reasoning={'effort':'medium'}, **opts)


@wrap_heartbeat("alpaca","v2/account", method="GET")
def ping_alpaca_account():
    base = os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets")
    key = os.getenv("APCA_API_KEY_ID"); sec = os.getenv("APCA_API_SECRET_KEY")
    r = requests.get(f"{base}/v2/account", headers={"APCA-API-KEY-ID":key or "", "APCA-API-SECRET-KEY":sec or ""}, timeout=15)
    class Resp: pass
    resp = Resp(); resp.status_code = r.status_code; resp.usage = {}
    if r.status_code >= 400: raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:120]}")
    return resp


if __name__ == "__main__":
    try:
        ping_gpt()
        print("GPT-5 OK")
    except Exception as e:
        print("GPT-5 FAIL:", e)
    try:
        ping_alpaca_account()
        print("Alpaca OK")
    except Exception as e:
        print("Alpaca FAIL:", e)
    print("Tail last 20:")
    from infra.heartbeat import tail_heartbeat
    tail_heartbeat(20, True)
'''


def main():
    # 1) Write heartbeat + health gate + smoke
    write_file(ROOT / "infra" / "heartbeat.py", HEARTBEAT_PY)
    write_file(ROOT / "infra" / "health_gate.py", HEALTH_GATE_PY)
    write_file(ROOT / "scripts" / "heartbeat_smoke.py", SMOKE_PY)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # 2) Patch likely clients
    openai_candidates = [
        ROOT / "openai_client.py",
        ROOT / "clients" / "openai_client.py",
        ROOT / "services" / "openai_client.py",
        ROOT / "deltatrade" / "openai_client.py",
    ]
    alpaca_candidates = [
        ROOT / "alpaca_client.py",
        ROOT / "brokers" / "alpaca_client.py",
        ROOT / "services" / "alpaca_client.py",
        ROOT / "deltatrade" / "alpaca_client.py",
    ]
    runner_candidates = [
        ROOT / "trader.py",
        ROOT / "engine.py",
        ROOT / "runner.py",
        ROOT / "deltatrade" / "trader.py",
        ROOT / "deltatrade" / "engine.py",
        ROOT / "deltatrade" / "runner.py",
    ]

    # decorate OpenAI call
    for p in openai_candidates:
        safe_insert_before_def(
            p,
            "from infra.heartbeat import wrap_heartbeat",
            "@wrap_heartbeat('openai','chat.completions', req_meta_fn=lambda **k: {'model': k.get('model'), 'run_id': k.get('run_id')})",
            def_name_hint="gpt"  # def ...gpt...( ... )
        )
        # add reasoning effort=medium
        add_reasoning_effort_medium(p)

    # decorate Alpaca order call
    for p in alpaca_candidates:
        safe_insert_before_def(
            p,
            "from infra.heartbeat import wrap_heartbeat",
            "@wrap_heartbeat('alpaca','v2/orders', req_meta_fn=lambda **k: {'symbol': k.get('symbol'), 'side': k.get('side')})",
            def_name_hint="order"  # def ...order...( ... )
        )

    # add pre-trade gate to runner
    for p in runner_candidates:
        add_pretrade_gate_to_runner(p)

    print("\nDone.\nNext steps:")
    print("1) Run: python scripts/heartbeat_smoke.py   # quick sanity; expect 'OK' lines and heartbeats")
    print("2) Before each 10:05 / 14:35 / 16:35 ET cycle, your bot will now fail-closed if APIs look bad.")
    print("3) Tail the log: tail -n 50 logs/heartbeat.ndjson")


if __name__ == "__main__":
    main()
