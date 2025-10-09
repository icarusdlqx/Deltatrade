from __future__ import annotations
import os, csv
from pathlib import Path
from datetime import datetime, timezone

BASE_PROMPT = os.getenv("LLM_BASE_PROMPT",
    "You are Deltatrade’s risk-aware news scorer. For each item, output a JSON list "
    "of numbers in the range [-3,3] where -3 is strongly bearish and +3 strongly bullish. "
    "Use only information in the text; if uncertain, return 0. Keep the output compact.")

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LLM_LOG = LOG_DIR / "llm_calls.csv"

def _log_tokens(model: str, prompt_tokens: int = None, completion_tokens: int = None):
    if os.getenv("LLM_LOG_TOKENS", "1") != "1":
        return
    if not LLM_LOG.exists():
        with LLM_LOG.open("w", newline="") as f:
            csv.writer(f).writerow(["ts_iso","model","prompt_tokens","completion_tokens"])
    with LLM_LOG.open("a", newline="") as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                model, prompt_tokens or "", completion_tokens or ""])

def _client_new():
    # New SDK style
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

def chat_once(system_prompt: str, user_text: str, model: str = None) -> str:
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Try new SDK first
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
            usage = getattr(r, "usage", None)
            if usage:
                _log_tokens(model, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
            return msg
        except Exception as _e:
            pass
    # Fallback to legacy
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
            usage = r.get("usage", {})
            _log_tokens(model, usage.get("prompt_tokens"), usage.get("completion_tokens"))
            return msg
        except Exception as _e:
            pass
    raise RuntimeError("OpenAI client not available or call failed.")

def smoke_test():
    """Run a tiny call at boot so you can see whether tokens are used."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[llm] OPENAI_API_KEY not set; skipping smoke test.")
        return
    try:
        txt = chat_once("You are a health check.", "Reply with OK.")
        print("[llm] OpenAI API smoke test:", txt.strip()[:40])
    except Exception as e:
        print("[llm] OpenAI API smoke test failed:", repr(e))
