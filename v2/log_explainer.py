"""
Build a plain-English narrative for the Log page with a value-investor focus.
Summarizes macro context, gate math, and per-symbol actions with valuation color.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _planned_vs_executed(ep: Dict[str, Any]) -> str:
    planned = ep.get("orders_planned") or ep.get("orders_plan") or []
    executed = ep.get("orders_submitted") or []
    if isinstance(planned, dict):
        planned_ct = len(planned)
    elif isinstance(planned, list):
        planned_ct = len(planned)
    else:
        planned_ct = 0
    exec_ids: List[Any]
    if isinstance(executed, dict):
        exec_ids = [oid for oid in executed.values() if oid != "DRY_RUN_NO_ORDERS"]
    elif isinstance(executed, list):
        exec_ids = [oid for oid in executed if oid != "DRY_RUN_NO_ORDERS"]
    else:
        exec_ids = []
    executed_ct = len(exec_ids)
    status = "EXECUTED" if executed_ct > 0 else "HELD"
    gate_txt = ""
    gate = ep.get("gate") or {}
    if isinstance(gate, dict):
        reasons = gate.get("friendly_reasons") or gate.get("reasons")
        if isinstance(reasons, list) and reasons:
            gate_txt = " Gate: " + ", ".join(str(r) for r in reasons if r)
        elif isinstance(reasons, str) and reasons:
            gate_txt = f" Gate: {reasons}"
    return f"Planned {planned_ct} order(s), executed {executed_ct} — status: {status}.{gate_txt}"


def _explain_gate_math(ep: Dict[str, Any]) -> str:
    gate = ep.get("gate") or {}
    if not isinstance(gate, dict):
        return ""
    expected = float(gate.get("expected_alpha_bps") or ep.get("expected_alpha_bps") or 0.0)
    cost = float(gate.get("cost_bps") or ep.get("est_cost_bps") or 0.0)
    net = float(gate.get("net_bps") or ep.get("net_edge_bps") or (expected - cost))
    turnover = gate.get("turnover_pct")
    pieces = [f"Gate math: exp {expected:.1f} bps, cost {cost:.1f} bps, net {net:.1f} bps."]
    if turnover not in (None, ""):
        try:
            pieces.append(f"Turnover {float(turnover):.1f}%.")
        except (TypeError, ValueError):
            pass
    proceed = gate.get("proceed") or gate.get("proceed_final") or ep.get("proceed")
    if proceed:
        pieces.append("Proceed = yes.")
    else:
        pieces.append("Proceed = no.")
    return " ".join(pieces)


def _explain_world_state(ep: Dict[str, Any]) -> str:
    report = ep.get("advisor_report") if isinstance(ep.get("advisor_report"), dict) else None
    if not isinstance(report, dict):
        return ""
    summary = (report.get("world_state_summary") or "").strip()
    if not summary:
        return ""
    cites = report.get("citations") or []
    cite_txt = ""
    if isinstance(cites, list) and cites:
        urls = []
        for item in cites[:3]:
            if isinstance(item, dict) and item.get("url"):
                urls.append(str(item["url"]))
            elif isinstance(item, str):
                urls.append(item)
        if urls:
            cite_txt = " Sources: " + ", ".join(urls)
    return f"Market & macro view: {summary}{cite_txt}"


def _actions_bullets(ep: Dict[str, Any]) -> List[str]:
    report = ep.get("advisor_report") if isinstance(ep.get("advisor_report"), dict) else None
    if not isinstance(report, dict):
        return []
    bullets: List[str] = []
    for action in (report.get("actions") or [])[:8]:
        if not isinstance(action, dict):
            continue
        ticker = str(action.get("ticker") or "").upper()
        if not ticker:
            continue
        act = str(action.get("action") or "").upper()
        rationale = (action.get("rationale") or "").strip()
        bits = [f"{ticker}: {act} — {rationale}" if rationale else f"{ticker}: {act}"]
        valuation = action.get("valuation") or {}
        if isinstance(valuation, dict):
            val_bits: List[str] = []
            mapping = (
                ("pe_forward", "fwd P/E"),
                ("ev_ebitda", "EV/EBITDA"),
                ("fcf_yield_pct", "FCF yield"),
                ("rev_cagr_3y_pct", "3y rev CAGR"),
                ("op_margin_trend", "margin"),
            )
            for key, label in mapping:
                val = valuation.get(key)
                if val in (None, ""):
                    continue
                val_bits.append(f"{label}={val}")
            if val_bits:
                bits.append("(" + ", ".join(val_bits) + ")")
        sources = action.get("sources") or []
        if isinstance(sources, list) and sources:
            srcs = [str(src) for src in sources[:3]]
            bits.append("Sources: " + ", ".join(srcs))
        bullets.append(" ".join(bits))
    return bullets


def build_human_log(ep: Dict[str, Any]) -> str:
    parts: List[str] = []
    world = _explain_world_state(ep)
    if world:
        parts.append(world)
    parts.append(_planned_vs_executed(ep))
    gate = _explain_gate_math(ep)
    if gate:
        parts.append(gate)
    actions = _actions_bullets(ep)
    if actions:
        parts.append("Portfolio actions (12-month view): " + " | ".join(actions))
    return " ".join(filter(None, parts)) or "Run recorded."
