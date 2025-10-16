from __future__ import annotations
import os, csv, json
from pathlib import Path
from datetime import datetime, timezone

BASE_PROMPT = (
    "You are Deltatrade’s event scorer (prompt_v1). "
    "Output ONLY JSON: {\"TICKER\": score, ...} where score is a number in [-3,3]. "
    "Use the provided headlines; if unsure, use 0. Do not add commentary or code fences."
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LLM_LOG = LOG_DIR / "llm_calls.csv"


class LLMError(Exception):
    def __init__(self, kind, msg):
        super().__init__(msg)
        self.kind = kind


def _log(model, prompt, pt=None, ct=None, note=""):
    if os.getenv("LLM_LOG_TOKENS", "1") != "1":
        return
    new = not LLM_LOG.exists()
    with LLM_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_iso", "model", "prompt_excerpt", "prompt_tokens", "completion_tokens", "note"])
        ex = prompt[:240] + ("…" if len(prompt) > 240 else "")
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model,
                ex,
                pt or "",
                ct or "",
                note,
            ]
        )


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


def chat_json(user_text: str, system_prompt: str = BASE_PROMPT, model: str = None) -> str:
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("fatal", "OPENAI_API_KEY missing")
    cli = _client_new()
    if cli:
        try:
            r = cli.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
                temperature=0.1,
            )
            msg = r.choices[0].message.content
            u = getattr(r, "usage", None)
            _log(
                model,
                system_prompt + "\n\n" + user_text,
                getattr(u, "prompt_tokens", None),
                getattr(u, "completion_tokens", None),
            )
            return msg
        except Exception as e:
            _log(model, system_prompt + "\n\n" + user_text, note=f"error:{e}")
            raise LLMError(_classify(str(e)), str(e))
    cli = _client_legacy()
    if cli:
        try:
            r = cli.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
                temperature=0.1,
            )
            msg = r["choices"][0]["message"]["content"]
            u = r.get("usage", {})
            _log(
                model,
                system_prompt + "\n\n" + user_text,
                u.get("prompt_tokens"),
                u.get("completion_tokens"),
            )
            return msg
        except Exception as e:
            _log(model, system_prompt + "\n\n" + user_text, note=f"error:{e}")
            raise LLMError(_classify(str(e)), str(e))
    raise LLMError("unknown", "No OpenAI client")


def _classify(s: str) -> str:
    s = (s or "").lower()
    if any(k in s for k in ["rate", "limit", "timeout", "temporar", "unavailable"]):
        return "transient"
    if any(k in s for k in ["auth", "invalid api key", "billing", "payment"]):
        return "fatal"
    return "unknown"


def smoke_test():
    try:
        _ = chat_json("Reply with OK.", "You are a health check.")
        print("[llm] Smoke test: OK")
    except LLMError as e:
        print("[llm] Smoke test failed:", e.kind, str(e)[:80])
