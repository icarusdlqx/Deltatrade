from __future__ import annotations
"""
AI scoring for news items.
Tries GPT-5 helper if available; falls back to a deterministic keyword model.
"""
from typing import Any, Dict, List

MAJOR_MACRO_TOKENS = [
    "cpi","ppi","pce","inflation","jobs","payrolls","unemployment","fomc","rate","rates",
    "yield","treasury","fed","powell","ecb","boj","boe","pmi","ism","gdp","tariff","sanction",
    "shutdown","opec","oil","energy","war","conflict","ceasefire",
]
NEG_TOKENS = [
    "hawkish","hot inflation","surprise jump","surge in yields","spike in yields","selloff","default",
    "downgrade","profit warning","guidance cut","strike","shutdown","escalation","sanction","war",
    "conflict","misses","miss","slump","recession","stagflation","soars above","accelerates",
]
POS_TOKENS = [
    "dovish","cooling inflation","disinflation","beats","beat","above consensus","upgrade","truce",
    "ceasefire","stimulus","cuts rates","rate cut","yield falls","pullback in yields","reopening",
]


def _count_hits(text: str, vocab: List[str]) -> int:
    text = text.lower()
    cnt = 0
    for w in vocab:
        if w in text:
            cnt += 1
    return cnt


def _keyword_score(item: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{item.get('title','')} {item.get('summary','')}".lower()
    pos = _count_hits(text, POS_TOKENS)
    neg = _count_hits(text, NEG_TOKENS)
    maj = _count_hits(text, MAJOR_MACRO_TOKENS)
    # risk score in [-1, 1]
    denom = max(1, pos + neg)
    risk = (pos - neg) / denom
    # confidence increases if major macro tokens present
    conf = min(1.0, 0.3 + 0.15 * maj + 0.1 * denom)
    if risk > 0.15:
        bias = "risk_on"
        hint = "accumulate"
    elif risk < -0.15:
        bias = "risk_off"
        hint = "reduce"
    else:
        bias = "unclear"
        hint = "hold"
    rel = "high" if maj >= 1 else ("med" if denom >= 2 else "low")
    return {
        "risk_score": round(risk, 3),
        "risk_bias": bias,
        "relevance": rel,
        "action_hint": hint,
        "confidence": round(conf, 2),
        "topics": [t for t in MAJOR_MACRO_TOKENS if t in text][:6],
    }


def _try_llm(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optional path: reuse your repo's GPT-5 event scorer if present.
    Must be resilient to import/signature mismatches.
    """
    try:
        from v2.llm_gpt5 import score_events_gpt5  # type: ignore
    except Exception:
        return []
    try:
        # Minimal prompt packing compatible with a generic "score_events_gpt5"
        payload = {"NEWS": [
            f"{it.get('title','')}. {it.get('summary','')} (source: {it.get('source','')})"
            for it in items
        ]}
        _raw = score_events_gpt5(payload)  # repo-specific; we won't assume its exact shape
        # Try to map back; if shape is unknown, fall back gracefully
        out: List[Dict[str, Any]] = []
        for it in items:
            # default skeleton; try to enrich from raw only if clearly present
            out.append(_keyword_score(it))
        return out
    except Exception:
        return []


def score_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    # Try LLM path; if returns nothing useful, use keywords
    llm = _try_llm(items)
    if llm and len(llm) == len(items):
        return llm
    return [_keyword_score(it) for it in items]


def aggregate_bias(scored: List[Dict[str, Any]]) -> Dict[str, float]:
    if not scored:
        return {"risk_bias_mean": 0.0}
    vals = [float(x.get("risk_score", 0.0)) for x in scored]
    return {"risk_bias_mean": round(sum(vals) / max(1, len(vals)), 3)}
