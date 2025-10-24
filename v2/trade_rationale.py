from __future__ import annotations
"""Attach human-readable rationales to submitted orders on the episode."""
from typing import Any, Dict


def _from_web_report(ep: Dict[str, Any], sym: str) -> str:
    rep = (ep or {}).get("web_search_report", {})
    report = rep.get("report") if isinstance(rep, dict) else {}
    if not isinstance(report, dict):
        return ""
    actions = report.get("actions") or []
    try:
        for action in actions:
            if str(action.get("ticker", "")).upper() == sym.upper():
                rationale = action.get("rationale") or ""
                sources = action.get("sources") or []
                src_txt = f" Sources: {', '.join(sources[:3])}." if sources else ""
                return (rationale.strip() + src_txt).strip()
    except Exception:
        pass
    summary = report.get("world_state_summary") or ""
    return summary[:220] if summary else ""


def _from_news(ep: Dict[str, Any]) -> str:
    rep = (ep or {}).get("news_report", {})
    agg = (rep.get("aggregate") or {}).get("risk_bias_mean", 0.0) if isinstance(rep, dict) else 0.0
    if agg > 0.15:
        return "Risk-on tilt per aggregated macro/news signals."
    if agg < -0.15:
        return "Risk-off tilt per aggregated macro/news signals."
    return "Neutral macro/news stance."


def _from_long_term(ep: Dict[str, Any], sym: str) -> str:
    analysis = (ep or {}).get("long_term_analysis") or (ep or {}).get("diag", {}).get("long_term_analysis")
    if not isinstance(analysis, dict):
        return ""
    theses = analysis.get("theses") if isinstance(analysis.get("theses"), dict) else {}
    thesis = theses.get(sym.upper()) or theses.get(sym)
    if isinstance(thesis, dict):
        summary = thesis.get("summary") or ""
        if summary:
            return summary[:600]
    return ""


def _from_model(ep: Dict[str, Any], sym: str) -> str:
    sig = (ep or {}).get("signals", {}).get(sym.upper())
    if isinstance(sig, dict):
        bits = []
        if "score" in sig:
            bits.append(f"score={sig['score']:.2f}")
        if "trend" in sig:
            bits.append(f"trend={sig['trend']}")
        if bits:
            return "Model signals: " + ", ".join(bits)
    return ""


def attach_order_rationales(ep: Dict[str, Any]) -> Dict[str, Any]:
    try:
        orders = (ep or {}).get("orders_submitted") or []
        if not isinstance(orders, list) or not orders:
            return ep
        rationales = {}
        for od in orders:
            sym = str(od.get("symbol") or od.get("ticker") or "").upper()
            if not sym:
                continue
            why = _from_long_term(ep, sym) or _from_web_report(ep, sym) or _from_model(ep, sym) or _from_news(ep)
            if not why:
                why = "Order placed based on portfolio signals and constraints."
            rationales[sym] = why[:600]
        ep = dict(ep)
        ep["order_rationales"] = rationales
        return ep
    except Exception:
        return ep
