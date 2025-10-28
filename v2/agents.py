from __future__ import annotations
import os, json, logging
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

log = logging.getLogger(__name__)

from .event_gate import record_assessment

class EventScore(BaseModel):
    symbol: str
    direction: int = Field(description="-1 short, 0 neutral, +1 long")
    magnitude: str = Field(description="low|med|high")
    confidence: float = Field(ge=0, le=1)
    half_life_days: int
    rationale: List[str] = []
    risks: List[str] = []

def _client():
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def _extract_text(resp) -> str:
    # Prefer modern Responses API helpers
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(resp, "output", None)
    if output:
        chunks: List[str] = []
        for item in output:  # type: ignore[assignment]
            content = getattr(item, "content", None) or []
            for part in content:
                parsed = getattr(part, "parsed", None)
                if parsed is not None:
                    try:
                        return json.dumps(parsed)
                    except Exception:
                        pass
                json_payload = getattr(part, "json", None)
                if json_payload is not None:
                    try:
                        return json.dumps(json_payload)
                    except Exception:
                        pass
                schema_payload = getattr(part, "json_schema", None)
                if isinstance(schema_payload, dict):
                    parsed_payload = schema_payload.get("parsed")
                    if parsed_payload is not None:
                        try:
                            return json.dumps(parsed_payload)
                        except Exception:
                            pass
                    content_text = schema_payload.get("text")
                    if isinstance(content_text, str) and content_text.strip():
                        chunks.append(content_text)
                        continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text)
        if chunks:
            return "".join(chunks)

    # Fallback to Chat Completions style payloads
    try:
        return resp["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:
        return "{}"

def score_events_for_symbols(news_by_symbol: Dict[str, List[dict]],
                             model: str, reasoning_effort: str,
                             bps_map: Dict[str, int], max_abs_bps: int) -> Tuple[Dict[str, float], Dict[str, object]]:
    symbols = list(news_by_symbol.keys())
    n_tickers = len(symbols)
    model_name = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or model or "gpt-5"
    effort = os.getenv("REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or reasoning_effort or "medium"
    meta: Dict[str, object] = {
        "called": False,
        "model": model_name,
        "effort": effort,
        "tokens": 0,
        "n_tickers": n_tickers,
        "reason": None,
    }

    details: Dict[str, Dict[str, object]] = {}
    summaries: List[str] = []

    def record_detail(sym: str,
                      *,
                      bps: float = 0.0,
                      direction: int = 0,
                      magnitude: str = "low",
                      confidence: float = 0.0,
                      half_life_days: int = 0,
                      rationale: List[str] | None = None,
                      risks: List[str] | None = None,
                      raw_response: str | None = None,
                      note: str | None = None) -> None:
        action = "Neutral"
        if direction > 0:
            action = "Buy tilt"
        elif direction < 0:
            action = "Sell tilt"
        conf_pct = int(round(max(0.0, min(1.0, confidence)) * 100))
        rationale_list = [str(r).strip() for r in (rationale or []) if str(r).strip()]
        risks_list = [str(r).strip() for r in (risks or []) if str(r).strip()]
        rationale_txt = "; ".join(rationale_list) if rationale_list else "No explicit rationale provided."
        risks_txt = "; ".join(risks_list) if risks_list else "No key risks highlighted."
        note_txt = (note or "LLM provided structured score.").strip()
        summary = (
            f"{sym}: {action} {bps:+.1f} bps (magnitude {magnitude}, confidence {conf_pct}%, "
            f"half-life {half_life_days}d). {note_txt} Rationale: {rationale_txt}. Risks: {risks_txt}."
        )
        detail = {
            "symbol": sym,
            "bps": float(bps),
            "direction": int(direction),
            "magnitude": str(magnitude),
            "confidence": float(confidence),
            "half_life_days": int(half_life_days),
            "rationale": rationale_list,
            "risks": risks_list,
            "raw_response": raw_response,
            "summary": summary,
            "note": note_txt,
        }
        details[sym] = detail
        summaries.append(summary)

    def neutral_meta(reason: str) -> Tuple[Dict[str, float], Dict[str, object]]:
        for sym in symbols:
            record_detail(sym, note=f"{reason} Score held neutral (0.0 bps).")
        meta["details"] = details
        meta["summaries"] = summaries
        return {s: 0.0 for s in symbols}, meta

    if not os.getenv("OPENAI_API_KEY"):
        meta["reason"] = "no_api_key"
        log.info("llm_skip", extra={"why": meta["reason"], "n_tickers": n_tickers})
        return neutral_meta("Skipped LLM scoring (no API key).")

    try:
        cli = _client()
    except Exception:
        meta["reason"] = "client_init_failed"
        log.info("llm_skip", extra={"why": meta["reason"], "n_tickers": n_tickers})
        return neutral_meta("Skipped LLM scoring (OpenAI client unavailable).")

    out: Dict[str, float] = {}
    schema = {
        "name": "EventScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "integer"},
                "magnitude": {"type": "string", "enum": ["low", "med", "high"]},
                "confidence": {"type": "number"},
                "half_life_days": {"type": "integer"},
                "rationale": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symbol", "direction", "magnitude", "confidence", "half_life_days"],
        },
    }
    total_tokens = 0
    for sym, items in news_by_symbol.items():
        if not items:
            out[sym] = 0.0
            record_detail(sym, note="No recent news items to score. Score held neutral (0.0 bps).")
            continue
        text = "\n".join([f"- {i.get('headline','')} :: {i.get('summary','')}" for i in items[:20]])
        try:
            # Prefer Responses API with JSON schema
            resp = cli.responses.create(
                model=model_name,
                reasoning={"effort": effort},
                input=[
                    {
                        "role": "system",
                        "content": "You are an equity event analyst. Return a calibrated, directionally correct score for near-term stock impact.",
                    },
                    {
                        "role": "user",
                        "content": f"Symbol: {sym}\nConsider the news items (most recent first):\n{text}\nReturn EventScore JSON.",
                    },
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            raw = _extract_text(resp)
            js = json.loads(raw)
            es = EventScore.model_validate(js)
            bps = es.direction * float(es.confidence) * float(bps_map.get(es.magnitude, 0))
            bps = max(-max_abs_bps, min(max_abs_bps, bps))
            out[sym] = float(bps)
            record_detail(
                sym,
                bps=bps,
                direction=es.direction,
                magnitude=es.magnitude,
                confidence=es.confidence,
                half_life_days=es.half_life_days,
                rationale=list(es.rationale or []),
                risks=list(es.risks or []),
                raw_response=raw,
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:
                token_fields = [
                    getattr(usage, "total_tokens", None),
                    getattr(usage, "output_tokens", None),
                ]
                for value in token_fields:
                    if value is not None:
                        total_tokens += int(value)
                        break
            meta["called"] = True
            log.info("event_score_detail %s", details[sym]["summary"])
        except Exception as exc:
            out[sym] = 0.0
            log.warning("event_score_failed for %s: %s", sym, exc)
            record_detail(
                sym,
                note=f"Failed to parse LLM response ({type(exc).__name__}). Score held neutral (0.0 bps).",
                raw_response=None,
            )
    if meta.get("called"):
        meta["tokens"] = total_tokens
        record_assessment()
        log.info(
            "llm_call",
            extra={
                "model": model_name,
                "effort": effort,
                "tokens": total_tokens,
                "n_tickers": n_tickers,
            },
        )
    else:
        meta["reason"] = meta.get("reason") or "no_calls"
        log.info(
            "llm_skip",
            extra={"why": meta["reason"], "model": model_name, "n_tickers": n_tickers},
        )
    meta["details"] = details
    meta["summaries"] = summaries
    return out, meta

def risk_officer_check(proposed: Dict[str, float], sector_map: Dict[str, str], sector_max: float, name_max: float) -> Dict[str, str]:
    import collections
    if not proposed:
        return {"approved":"true","reason":"no exposure"}
    sector_tot = collections.defaultdict(float)
    for s, w in proposed.items():
        if w > name_max * 1.10:
            return {"approved":"false","reason":f"name cap breached {s}: {w:.2%} > {name_max:.2%}"}
        sec = sector_map.get(s, "UNKNOWN")
        sector_tot[sec] += max(0.0, w)
    for sec, tot in sector_tot.items():
        if tot > sector_max * 1.10:
            return {"approved":"false","reason":f"sector cap {sec} breached: {tot:.2%} > {sector_max:.2%}"}
    return {"approved":"true","reason":"within bounds"}
