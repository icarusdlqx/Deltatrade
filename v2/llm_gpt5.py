from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROMPT_VERSION = os.getenv("LLM_PROMPT_VERSION", "event_v1")
BASE_PROMPT = os.getenv(
    "LLM_BASE_PROMPT",
    "You are Deltatrade’s event scorer ("
    + PROMPT_VERSION
    + "). "
    "Read the per-ticker headlines and output ONLY JSON: "
    '{"TICKER": score, ...} where each score is a number in [-3,3]. '
    "-3 = strongly bearish, +3 = strongly bullish, 0 = uncertain/neutral. "
    "Use ONLY the supplied headlines; never invent tickers; no prose, no code fences.",
)


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RAW_DIR = LOG_DIR / "llm_responses"
RAW_DIR.mkdir(exist_ok=True)
CALLS_CSV = LOG_DIR / "llm_calls.csv"


class LLMError(Exception):
    def __init__(self, kind: str, msg: str) -> None:
        super().__init__(msg)
        self.kind = kind


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_call(
    model: str,
    prompt: str,
    user_excerpt: str,
    input_tokens: Any = None,
    output_tokens: Any = None,
    note: str = "",
) -> None:
    if os.getenv("LLM_LOG_TOKENS", "1") != "1":
        return

    new = not CALLS_CSV.exists()
    with CALLS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(
                [
                    "ts_iso",
                    "model",
                    "effort",
                    "prompt_version",
                    "prompt_sha1",
                    "user_excerpt",
                    "input_tokens",
                    "output_tokens",
                    "note",
                ]
            )
        sha = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()[:12]
        writer.writerow(
            [
                _ts(),
                model,
                "medium",
                PROMPT_VERSION,
                sha,
                user_excerpt[:240],
                input_tokens or "",
                output_tokens or "",
                note,
            ]
        )


def _save_raw(payload: Dict[str, Any]) -> None:
    if os.getenv("LLM_SAVE_RAW", "1") != "1":
        return

    fname = RAW_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    try:
        fname.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


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


def _compose_user_text(headlines_by_symbol: Dict[str, List[str]]) -> str:
    if not headlines_by_symbol:
        return "NO_HEADLINES:\n - No material headlines in the last window."

    lines: List[str] = []
    for sym, heads in sorted(headlines_by_symbol.items()):
        if not heads:
            continue
        lines.append(f"{sym.upper()}:")
        for h in heads[:6]:
            h = (h or "").replace("\n", " ").strip()
            if h:
                lines.append(" - " + h)

    return "\n".join(lines) if lines else "NO_HEADLINES:\n - (empty list)"


def _parse_scores(txt: str) -> Dict[str, float]:
    try:
        obj = json.loads(txt)
        out: Dict[str, float] = {}
        for key, value in obj.items():
            try:
                out[key.upper()] = float(max(-3.0, min(3.0, float(value))))
            except Exception:
                continue
        return out
    except Exception:
        return {}


def score_events_gpt5(headlines_by_symbol: Dict[str, List[str]]) -> Dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("fatal", "OPENAI_API_KEY missing")

    user_text = _compose_user_text(headlines_by_symbol)
    user_excerpt = user_text[:200] + ("…" if len(user_text) > 200 else "")

    client_new = _client_new()
    if client_new:
        try:
            response = client_new.responses.create(
                model=model,
                input=
                [
                    {"role": "developer", "content": BASE_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                reasoning={"effort": "medium"},
            )

            raw = ""
            if hasattr(response, "output"):
                for item in response.output:  # type: ignore[attr-defined]
                    if hasattr(item, "content"):
                        for content in item.content:  # type: ignore[attr-defined]
                            if hasattr(content, "text"):
                                raw += content.text

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None) if usage else None
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            _log_call(
                model,
                BASE_PROMPT,
                user_excerpt,
                input_tokens,
                output_tokens,
                note="responses",
            )
            scores = _parse_scores(raw)
            payload = {
                "scores": scores,
                "raw": raw,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "model": model,
                "source": "live",
            }
            _save_raw(
                {
                    "prompt_version": PROMPT_VERSION,
                    "developer_prompt": BASE_PROMPT,
                    "user_text": user_text,
                    "response_text": raw,
                    "usage": payload["usage"],
                }
            )
            return payload
        except Exception as exc:  # pragma: no cover - network
            _log_call(model, BASE_PROMPT, user_excerpt, note=f"responses_error:{exc}")

    client_legacy = _client_legacy()
    if client_legacy:
        try:
            response = client_legacy.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": BASE_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                reasoning={"effort": "medium"},
                temperature=0.1,
            )

            raw = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            _log_call(
                model,
                BASE_PROMPT,
                user_excerpt,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                note="chat",
            )
            scores = _parse_scores(raw)
            payload = {
                "scores": scores,
                "raw": raw,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                },
                "model": model,
                "source": "live",
            }
            _save_raw(
                {
                    "prompt_version": PROMPT_VERSION,
                    "developer_prompt": BASE_PROMPT,
                    "user_text": user_text,
                    "response_text": raw,
                    "usage": payload["usage"],
                }
            )
            return payload
        except Exception as exc:  # pragma: no cover - network
            _log_call(model, BASE_PROMPT, user_excerpt, note=f"chat_error:{exc}")

    raise LLMError("unknown", "No OpenAI client available or both endpoints failed")

