from __future__ import annotations
import os, csv, json, math
from pathlib import Path
from datetime import datetime, timezone

BASE_PROMPT = os.getenv("LLM_BASE_PROMPT",
  "You are Deltatrade’s risk-aware news scorer. "
  "Read headlines/snippets and return a compact JSON object mapping each mentioned "
  "ticker to a score in [-3,3] (-3 strong bearish, +3 strong bullish). "
  "If unsure, use 0. Do not invent tickers."
)

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LLM_LOG = LOG_DIR / "llm_calls.csv"

class LLMError(Exception):
    def __init__(self, err_type: str, message: str):
        super().__init__(f"{err_type}: {message}")
        self.err_type = err_type
        self.message = message

def _log_tokens(model: str, prompt: str, prompt_tokens=None, completion_tokens=None, note=""):
    if os.getenv("LLM_LOG_TOKENS","1") != "1": return
    new = not LLM_LOG.exists()
    with LLM_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_iso","model","prompt_excerpt","prompt_tokens","completion_tokens","note"])
        excerpt = prompt[:240] + ("…" if len(prompt)>240 else "")
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    model, excerpt, prompt_tokens or "", completion_tokens or "", note])

def _client_new():
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        return None

def _client_legacy():
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return openai
    except Exception:
        return None

def _classify_error_text(e: Exception) -> str:
    s = str(e).lower()
    if any(k in s for k in ["rate limit", "overloaded", "timeout", "temporar", "server error", "unavailable"]):
        return "transient"
    if any(k in s for k in ["invalid api key", "authentication", "billing", "payment"]):
        return "fatal"
    return "unknown"

def chat_json(user_text: str, system_prompt: str = None, model: str = None) -> str:
    system_prompt = system_prompt or BASE_PROMPT
    model = model or os.getenv("OPENAI_MODEL","gpt-4o-mini")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("fatal","OPENAI_API_KEY not set")

    cli = _client_new()
    if cli:
        try:
            r = cli.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_text}],
                temperature=0.1
            )
            msg = r.choices[0].message.content
            u = getattr(r, "usage", None)
            _log_tokens(model, system_prompt+"\n\n"+user_text,
                        getattr(u,"prompt_tokens",None), getattr(u,"completion_tokens",None))
            return msg
        except Exception as e:
            _log_tokens(model, system_prompt+"\n\n"+user_text, note=f"error:{type(e).__name__}:{e}")
            raise LLMError(_classify_error_text(e), str(e))

    cli = _client_legacy()
    if cli:
        try:
            r = cli.ChatCompletion.create(
                model=model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_text}],
                temperature=0.1
            )
            msg = r["choices"][0]["message"]["content"]
            u = r.get("usage", {})
            _log_tokens(model, system_prompt+"\n\n"+user_text,
                        u.get("prompt_tokens"), u.get("completion_tokens"))
            return msg
        except Exception as e:
            _log_tokens(model, system_prompt+"\n\n"+user_text, note=f"error:{type(e).__name__}:{e}")
            raise LLMError(_classify_error_text(e), str(e))

    raise LLMError("unknown","No OpenAI client available")

def smoke_test():
    try:
        _ = chat_json("Reply with OK.", "You are a health check.")
        print("[llm] Smoke test: OK")
    except LLMError as e:
        print("[llm] Smoke test failed:", e.err_type, e.message)
