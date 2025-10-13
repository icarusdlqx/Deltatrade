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
    # Try Responses API
    try:
        return resp.output[0].content[0].text
    except Exception:
        pass
    # Try chat completions
    try:
        return resp.choices[0].message["content"]
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

    if not os.getenv("OPENAI_API_KEY"):
        meta["reason"] = "no_api_key"
        log.info("llm_skip", extra={"why": meta["reason"], "n_tickers": n_tickers})
        return {s: 0.0 for s in symbols}, meta

    try:
        cli = _client()
    except Exception:
        meta["reason"] = "client_init_failed"
        log.info("llm_skip", extra={"why": meta["reason"], "n_tickers": n_tickers})
        return {s: 0.0 for s in symbols}, meta

    out: Dict[str, float] = {}
    schema = {
      "name": "EventScore",
      "schema": {
        "type": "object",
        "properties": {
          "symbol": {"type": "string"},
          "direction": {"type": "integer"},
          "magnitude": {"type": "string", "enum": ["low","med","high"]},
          "confidence": {"type": "number"},
          "half_life_days": {"type": "integer"},
          "rationale": {"type": "array", "items": {"type":"string"}},
          "risks": {"type": "array", "items": {"type":"string"}}
        },
        "required": ["symbol","direction","magnitude","confidence","half_life_days"]
      }
    }
    total_tokens = 0
    for sym, items in news_by_symbol.items():
        if not items:
            out[sym] = 0.0
            continue
        text = "\n".join([f"- {i.get('headline','')} :: {i.get('summary','')}" for i in items[:20]])
        try:
            # Prefer Responses API with JSON schema
            resp = cli.responses.create(
                model=model_name,
                reasoning={"effort": effort},
                input=[
                    {"role":"system","content":"You are an equity event analyst. Return a calibrated, directionally correct score for near-term stock impact."},
                    {"role":"user","content": f"Symbol: {sym}\nConsider the news items (most recent first):\n{text}\nReturn EventScore JSON."}
                ],
                response_format={"type":"json_schema","json_schema":schema}
            )
            raw = _extract_text(resp)
            js = json.loads(raw)
            es = EventScore.model_validate(js)
            bps = es.direction * float(es.confidence) * float(bps_map.get(es.magnitude, 0))
            bps = max(-max_abs_bps, min(max_abs_bps, bps))
            out[sym] = float(bps)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                tokens = getattr(usage, "total_tokens", None)
                if tokens is not None:
                    total_tokens += int(tokens)
            meta["called"] = True
        except Exception:
            out[sym] = 0.0
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
