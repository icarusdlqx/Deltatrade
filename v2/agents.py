from __future__ import annotations
import os, json
from typing import Dict, List
from pydantic import BaseModel, Field
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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
                             bps_map: Dict[str, int], max_abs_bps: int) -> Dict[str, float]:
    try:
        cli = _client()
    except Exception:
        return {s: 0.0 for s in news_by_symbol.keys()}
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
    for sym, items in news_by_symbol.items():
        if not items:
            out[sym] = 0.0
            continue
        text = "\n".join([f"- {i.get('headline','')} :: {i.get('summary','')}" for i in items[:20]])
        try:
            # Prefer Responses API with JSON schema
            resp = cli.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
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
        except Exception:
            out[sym] = 0.0
    return out

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
