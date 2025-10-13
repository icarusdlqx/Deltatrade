from __future__ import annotations
import os, csv
from pathlib import Path
from datetime import datetime, timezone

BASE_PROMPT = os.getenv("LLM_BASE_PROMPT",
  "You are Deltatrade’s risk-aware news scorer. "
  "Given one or more headlines/snippets about equities, return a compact JSON "
  "object mapping each mentioned ticker to a score in [-3,3] "
  "(-3 strong bearish, +3 strong bullish). If uncertain, use 0. "
  "Only use information in the text; do not invent tickers."
)

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "llm_calls.csv"

def _log(model: str, prompt: str, prompt_tokens=None, completion_tokens=None, note=""):
    if os.getenv("LLM_LOG_TOKENS","1") != "1": return
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_iso","model","prompt_excerpt","prompt_tokens","completion_tokens","note"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model, (prompt[:240] + ("…" if len(prompt)>240 else "")),
            prompt_tokens or "", completion_tokens or "", note
        ])

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

def chat_json(user_text: str, system_prompt: str = None, model: str = None) -> str:
    system_prompt = system_prompt or BASE_PROMPT
    model = model or os.getenv("OPENAI_MODEL","gpt-4o-mini")

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
            u = getattr(r,"usage",None)
            _log(model, system_prompt + "\n\n" + user_text,
                 getattr(u,"prompt_tokens",None), getattr(u,"completion_tokens",None))
            return msg
        except Exception as e:
            _log(model, system_prompt + "\n\n" + user_text, note=f"error:{e}")

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
            u = r.get("usage",{})
            _log(model, system_prompt + "\n\n" + user_text,
                 u.get("prompt_tokens"), u.get("completion_tokens"))
            return msg
        except Exception as e:
            _log(model, system_prompt + "\n\n" + user_text, note=f"error:{e}")
    raise RuntimeError("OpenAI client not available or call failed.")

def smoke_test():
    if os.getenv("RUN_LLM_SMOKE","1") != "1":
        return
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[llm] OPENAI_API_KEY not set; skipping smoke test.")
        _log(os.getenv("OPENAI_MODEL","gpt-4o-mini"), "smoke", note="no_key")
        return
    try:
        txt = chat_json("Reply only with OK.", "You are a health check.")
        print("[llm] Smoke test:", str(txt)[:40])
    except Exception as e:
        print("[llm] Smoke test failed:", repr(e))
