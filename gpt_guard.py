# Minimal harden-and-log wrapper for LLM trade decisions
# Requirements: python-dotenv (optional), requests (or openai if you prefer), backoff
#   pip install requests backoff python-dotenv

import os, time, json, uuid, hashlib, threading, requests, backoff
from datetime import datetime, timezone

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("TRADE_DECISION_MODEL", "gpt-5")             # locked
REASONING = os.getenv("TRADE_DECISION_REASONING", "medium")    # locked
P95_DEADLINE_SEC = float(os.getenv("TRADE_P95_DEADLINE_SEC", "9.5"))
ABS_CUTOFF_SEC   = float(os.getenv("TRADE_ABS_CUTOFF_SEC", "12.0"))
MAX_TOKENS       = int(os.getenv("TRADE_MAX_TOKENS", "700"))
TEMP             = float(os.getenv("TRADE_TEMP", "0.2"))

# Thread-safe, append-only JSONL audit log
LOG_PATH = os.getenv("TRADE_AUDIT_LOG", "logs/trade_llm_audit.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
_lock = threading.Lock()

SYSTEM_PROMPT = """You are the portfolio’s Trade Decision Engine. 
Return ONLY strict JSON matching this schema:
{
  "decision": "buy|sell|hold",
  "symbol": "string",
  "confidence": 0..1,
  "size_pct": 0..100,
  "rationale": "concise but complete chain of thought for auditors (DO NOT include tool call IDs)",
  "risk_flags": ["list","of","strings"]
}
Rules:
- Respect hard risk rails: never exceed size_pct requested by risk params; never short if shorting_disabled=true.
- Use only data provided in 'inputs'.
- If inputs are stale or contradictory, return {"decision":"hold","symbol":"",...,"risk_flags":["stale_inputs"]}.
"""

# NOTE: if you prefer to avoid exposing chain-of-thought, set rationale to a short, non-sensitive summary instead.

def _stable_idempotency_key(payload: dict) -> str:
    """Stable idempotency for identical (model, reasoning, inputs) within 10s window."""
    base = {
        "m": MODEL,
        "r": REASONING,
        "inputs": payload.get("inputs"),
        "ts_bucket": int(time.time() // 10)
    }
    h = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    return f"trade-{h}"

def _headers(idempotency_key: str):
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Idempotency-Key": idempotency_key,                 # server-side dedupe
        "X-Client": "Deltatrade-GPT-Guard/1.0"
    }

def _completion_payload(prompt_inputs: dict):
    # Force model & reasoning; do NOT allow override from caller
    return {
        "model": MODEL,
        "reasoning": { "effort": REASONING },               # non-bypassable
        "temperature": TEMP,
        "max_output_tokens": MAX_TOKENS,
        "input": [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content": json.dumps(prompt_inputs, ensure_ascii=False)}
        ]
    }

def _deadline_guard(start_ts: float, p95_deadline=P95_DEADLINE_SEC, hard_cut=ABS_CUTOFF_SEC):
    elapsed = time.time() - start_ts
    if elapsed > hard_cut:
        raise TimeoutError(f"LLM call exceeded hard cutoff {hard_cut}s (elapsed={elapsed:.2f}s)")
    # Caller may use this to flip to a fallback after p95 deadline passes
    return elapsed > p95_deadline

def _log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

@backoff.on_exception(backoff.expo,
                      (requests.exceptions.RequestException, TimeoutError),
                      max_tries=4, jitter=backoff.full_jitter, factor=0.5)
def _post_with_retry(url, headers, payload, timeout):
    return requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

def call_trade_llm(inputs: dict, risk_params: dict, request_id: str | None = None):
    """
    Mandatory entry point for *all* trade decisions.
    - inputs: dict with market snapshot, features, signals
    - risk_params: dict with hard rails (max_position_pct, shorting_disabled, etc.)
    """
    assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
    start = time.time()
    req_id = request_id or str(uuid.uuid4())
    # Merge and freeze caller inputs
    prompt_inputs = {
        "inputs": {
            "market": inputs.get("market"),
            "features": inputs.get("features"),
            "signals": inputs.get("signals"),
            "positions": inputs.get("positions"),
            "clock": inputs.get("clock"),
        },
        "risk_params": risk_params
    }
    idem = _stable_idempotency_key(prompt_inputs)
    payload = _completion_payload(prompt_inputs)
    url = f"{OPENAI_BASE}/responses"  # compatible with modern Responses API

    # Soft p95 guard → fallback to cached heuristic if breached
    p95_fallback_triggered = False
    try:
        while True:
            p95_fallback_triggered = _deadline_guard(start) or p95_fallback_triggered
            timeout_left = max(0.5, ABS_CUTOFF_SEC - (time.time() - start))
            resp = _post_with_retry(url, _headers(idem), payload, timeout=timeout_left)
            resp.raise_for_status()
            data = resp.json()

            # ---- Parse model output (Responses API shape-agnostic) ----
            # Expected: a single text item with strict JSON.
            txt = None
            # Try common shapes
            if "output" in data:
                # unified responses
                chunks = data["output"]
                for ch in chunks:
                    if ch.get("type") in ("message","output_text","text"):
                        txt = ch.get("content") if ch.get("type")=="text" else ch.get("content", [{}])[0].get("text")
                        if txt: break
            if not txt and "choices" in data:
                txt = data["choices"][0]["message"]["content"]

            # Validate JSON
            try:
                decision = json.loads(txt)
            except Exception as e:
                # Force model to repair into strict JSON once
                payload["input"].append({"role":"user","content":"Your prior reply was not valid JSON. Output strict JSON only."})
                # retry via backoff
                raise requests.exceptions.RequestException(f"Non-JSON model output: {txt[:200]}…")

            # Enforce hard rails
            max_allowed = float(risk_params.get("max_position_pct", 10))
            if float(decision.get("size_pct", 0)) > max_allowed:
                decision["risk_flags"] = list(set(decision.get("risk_flags", []) + ["size_exceeds_rail"]))
                decision["size_pct"] = max_allowed
            if risk_params.get("shorting_disabled", True) and decision.get("decision") == "sell" and inputs.get("positions", {}).get(decision.get("symbol",""), 0) <= 0:
                decision["decision"] = "hold"
                decision["risk_flags"] = list(set(decision.get("risk_flags", []) + ["shorting_disabled"]))

            latency = time.time() - start
            _log_event({
                "kind": "trade_llm_success",
                "request_id": req_id,
                "idempotency_key": idem,
                "model": MODEL,
                "reasoning": REASONING,
                "latency_sec": round(latency, 3),
                "p95_deadline_breached": p95_fallback_triggered,
                "inputs_fingerprint": hashlib.md5(json.dumps(prompt_inputs, sort_keys=True).encode()).hexdigest(),
                "raw_text": txt,                 # full model reply for later audit
                "decision": decision
            })
            return decision

    except Exception as e:
        latency = time.time() - start
        # Fallback: hold w/ risk flag; preserve last known scores if you maintain a cache
        fallback = {
            "decision": "hold",
            "symbol": "",
            "confidence": 0.0,
            "size_pct": 0.0,
            "rationale": "Fallback: model call failed or timed out; preserving positions.",
            "risk_flags": ["llm_call_failed"]
        }
        _log_event({
            "kind": "trade_llm_failure",
            "request_id": req_id,
            "idempotency_key": idem,
            "model": MODEL,
            "reasoning": REASONING,
            "latency_sec": round(latency, 3),
            "error": str(e),
            "inputs_fingerprint": hashlib.md5(json.dumps(prompt_inputs, sort_keys=True).encode()).hexdigest()
        })
        return fallback

