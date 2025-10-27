from __future__ import annotations

"""
Value-investor advisor with web search.
Focuses on 12-month theses, valuation color, and explicit citations.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

import pytz

from .investor_memory import build_position_memory
from .settings_bridge import get_cfg

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

ET = pytz.timezone("US/Eastern")

SCHEMA = {
    "name": "deltatrade_advisor_v1",
    "schema": {
        "type": "object",
        "properties": {
            "as_of": {"type": "string"},
            "world_state_summary": {"type": "string", "maxLength": 1200},
            "risk_bias": {"type": "number", "minimum": -1, "maximum": 1},
            "investment_ratio_target": {"type": "number", "minimum": 0, "maximum": 1},
            "theses": {
                "type": "array",
                "minItems": 3,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "tickers": {"type": "array", "items": {"type": "string"}},
                        "horizon_months": {"type": "integer"},
                    },
                    "required": ["title", "summary"],
                },
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "asset_type": {"type": "string", "enum": ["stock", "etf"]},
                        "action": {"type": "string", "enum": ["buy", "sell", "hold", "no_change"]},
                        "weight_target": {"type": "number"},
                        "rationale": {"type": "string"},
                        "valuation": {
                            "type": "object",
                            "properties": {
                                "pe_forward": {"type": "number"},
                                "ev_ebitda": {"type": "number"},
                                "fcf_yield_pct": {"type": "number"},
                                "rev_cagr_3y_pct": {"type": "number"},
                                "op_margin_trend": {"type": "string"},
                                "moat": {"type": "string"},
                                "risks": {"type": "string"},
                                "catalysts": {"type": "string"},
                            },
                        },
                        "sources": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ticker", "asset_type", "action", "rationale"],
                },
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["as_of", "world_state_summary", "risk_bias", "actions", "citations"],
    },
}

SYSTEM_PROMPT = """\
You are Deltatrade’s Chief VALUE Investor. Your horizon is ~12 months. Your goal is to beat the S&P 500 with\nthoughtful capital allocation, low churn, and company-level fundamentals. Behave like an expert PM running a\nconcentrated book with a baseline invested band of 60–70% unless evidence is strong.\n\nDO IN ONE CALL:\n1) Use the web_search tool to gather *today’s* world state:\n   - Macro: inflation (CPI/PCE), jobs (NFP/unemployment), growth (GDP/ISM/PMI), fiscal impulses.\n   - Central banks: FOMC/ECB/BOJ communication; 2y/10y yields; balance of risks.\n   - Geopolitics/energy/shipping disruptions that change earnings or multiples.\n   - Company fundamentals: revenue trajectory, margins, unit economics, capital intensity, balance sheet,\n     guidance revisions, structural tailwinds/headwinds; prefer SEC filings, earnings call summaries,\n     and tier-1 outlets (Reuters/Bloomberg/WSJ/FT). Favor recency within {recency} day(s). Cap to ~{max_pages} pages.\n\n2) Synthesize 3–7 durable theses for the next 12 months and propose BUY/HOLD/SELL actions across S&P 500\n   and liquid ETFs (SPY/sectors/factors/commodities) that express those theses. For EACH action:\n   - Give a *plain-English* rationale focused on the next 12 months.\n   - Include valuation color (e.g., forward P/E, EV/EBITDA, FCF yield), expected revenue/CAGR or margin direction,\n     key catalysts and risks. Use the 'valuation' object when possible.\n   - If the position is currently held and < {min_hold} days, prefer HOLD unless the thesis *changed materially*;\n     then justify the change explicitly (“thesis changed because …”).\n   - Provide 1–3 source URLs used.\n\n3) Return VALID JSON per schema. No extra prose outside JSON.\n"""

USER_PROMPT_TMPL = """\
You are advising a value-investor PM with a 12-month horizon.\nPortfolio snapshot:\n- Cash %: {cash_pct:.1f}\n- Current positions: {positions_line}\n\nLatest rationales (continuity):\n{memory_lines}\n\nConstraints:\n- Universe: S&P 500 + liquid ETFs (SPY/sectors/factors/commodities).\n- Max actions per run: {max_actions}\n- Prefer these domains: {allowlist}\n- If evidence is insufficient for a trade, choose HOLD/NO_CHANGE with a clear reason.\n\nReturn STRICT JSON per schema with citations.\n"""



def _build_user_prompt(portfolio: Dict[str, Any], memory_map: Dict[str, Dict[str, Any]], cfg) -> str:
    positions = portfolio.get("positions", [])
    snippets = []
    for pos in positions[:30]:
        sym = str(pos.get("symbol", "")).upper()
        weight = float(pos.get("weight", 0.0))
        snippets.append(f"{sym}:{weight:.2%}")
    positions_line = ", ".join(snippets) if snippets else "(none)"

    mem_lines = []
    for sym, info in list(memory_map.items())[:30]:
        mem_lines.append(f"- {sym}: {info.get('rationale', '')} (as_of {info.get('as_of')})")
    memory_text = "\n".join(mem_lines) if mem_lines else "(no prior rationales available)"

    return USER_PROMPT_TMPL.format(
        cash_pct=float(portfolio.get("cash_pct", 0.0)) * 100.0,
        positions_line=positions_line,
        memory_lines=memory_text,
        max_actions=int(getattr(cfg, "ADVISOR_MAX_TRADES_PER_RUN", 6)),
        allowlist=", ".join(getattr(cfg, "WEB_ADVISOR_DOMAIN_ALLOWLIST", [])),
    )


def _portfolio_stub(tc) -> Dict[str, Any]:
    """Return cash percentage and active positions from the trading client."""

    try:
        acct = tc.get_account() if tc else None
        equity = float(getattr(acct, "equity", 0.0) or 0.0)
        cash = float(getattr(acct, "cash", 0.0) or 0.0)
        positions_raw = tc.get_all_positions() if (tc and hasattr(tc, "get_all_positions")) else []
        positions: List[Dict[str, Any]] = []
        for pos in positions_raw:
            sym = str(getattr(pos, "symbol", "")).upper()
            mv = float(getattr(pos, "market_value", 0.0) or 0.0)
            weight = (mv / equity) if equity > 0 else 0.0
            positions.append({"symbol": sym, "weight": weight})
        cash_pct = (cash / equity) if equity > 0 else 0.0
        return {"cash_pct": cash_pct, "positions": positions}
    except Exception:
        return {"cash_pct": 0.0, "positions": []}


def run_advisor(tc=None, cfg=None) -> Dict[str, Any]:
    """Execute the expert advisor and return a structured report."""

    cfg = cfg or get_cfg()
    if not getattr(cfg, "ENABLE_WEB_ADVISOR", True):
        return {"ok": False, "disabled": True, "reason": "disabled_by_setting"}
    if OpenAI is None:
        return {"ok": False, "error": "openai SDK not available"}

    client = OpenAI()
    recency = int(getattr(cfg, "WEB_ADVISOR_RECENCY_DAYS", 7))
    max_pages = int(getattr(cfg, "WEB_ADVISOR_MAX_PAGES", 12))
    model = str(getattr(cfg, "WEB_ADVISOR_MODEL", "gpt-5"))
    min_hold = int(getattr(cfg, "MIN_HOLD_DAYS_BEFORE_SELL", 90))

    from .config import EPISODES_MEMORY_LOOKBACK, EPISODES_PATH

    memory = build_position_memory(EPISODES_PATH, int(EPISODES_MEMORY_LOOKBACK))
    portfolio = _portfolio_stub(tc)

    system_prompt = SYSTEM_PROMPT.format(recency=recency, max_pages=max_pages, min_hold=min_hold)
    user_prompt = _build_user_prompt(portfolio=portfolio, memory_map=memory, cfg=cfg)

    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": SCHEMA},
        )
        payload = getattr(response, "output_text", None)
        data = json.loads(payload) if payload else {}
    except Exception as exc:  # pragma: no cover - network heavy
        return {"ok": False, "error": f"advisor_call_failed: {exc}"}

    if not isinstance(data, dict) or not data:
        data = {
            "as_of": datetime.now(ET).isoformat(),
            "world_state_summary": "Advisor returned no data.",
            "risk_bias": 0.0,
            "investment_ratio_target": 0.65,
            "theses": [],
            "actions": [],
            "citations": [],
            "notes": "fallback",
        }
    raw = payload[:50000] if isinstance(payload, str) else ""
    return {"ok": True, "report": data, "memory_used": bool(memory), "raw": raw}


def attach_advisor_to_episode(ep: Dict[str, Any], tc=None, cfg=None) -> Dict[str, Any]:
    """Attach the advisor output to an episode dict in-place."""

    cfg = cfg or get_cfg()
    episode = dict(ep or {})
    try:
        result = run_advisor(tc=tc, cfg=cfg)
    except Exception as exc:  # pragma: no cover - network heavy
        episode["advisor_ok"] = False
        episode["advisor_report"] = {"ok": False, "error": str(exc)}
        return episode

    episode["advisor_ok"] = bool(result.get("ok"))
    if result.get("ok"):
        report = result.get("report", {})
    else:
        report = {"ok": False, "error": result.get("error"), "disabled": result.get("disabled")}
    episode["advisor_report"] = report
    if isinstance(result, dict) and result.get("raw"):
        episode["advisor_raw_json"] = result["raw"]

    summary = ""
    if isinstance(report, dict):
        summary = str(report.get("world_state_summary") or "")[:300]
    episode["advisor_summary_1p"] = summary

    rationale_map: Dict[str, str] = {}
    if isinstance(report, dict):
        for action in report.get("actions", []) or []:
            ticker = str(action.get("ticker", "")).upper()
            rationale = action.get("rationale") or ""
            sources = action.get("sources") or []
            if ticker and rationale:
                suffix = f" Sources: {', '.join(sources[:3])}." if sources else ""
                rationale_map[ticker] = (rationale + suffix).strip()[:600]

    existing = episode.get("order_rationales") if isinstance(episode.get("order_rationales"), dict) else {}
    combined = dict(existing)
    combined.update(rationale_map)
    episode["order_rationales"] = combined

    return episode
