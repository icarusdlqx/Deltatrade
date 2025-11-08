from __future__ import annotations

"""Long-term portfolio construction utilities driven by LLM assessments."""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class PortfolioPlan:
    weights: Dict[str, float]
    notes: str
    meta: Dict[str, object]


def _client() -> Optional[OpenAI]:  # type: ignore[name-defined]
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)  # type: ignore[call-arg]
    except Exception:
        return None


def _normalise(weights: Mapping[str, float], cap: float) -> Dict[str, float]:
    capped = {sym: float(max(-cap, min(cap, w))) for sym, w in weights.items()}
    total = sum(max(0.0, w) for w in capped.values())
    if total <= 0:
        return {sym: 0.0 for sym in capped}
    scale = min(1.0, 1.0 / total)
    return {sym: round(max(0.0, w) * scale, 6) for sym, w in capped.items()}


def _compose_symbol_block(panel: pd.DataFrame, symbol: str) -> str:
    if symbol not in panel.index:
        return f"- {symbol}: insufficient data"
    row = panel.loc[symbol]
    def _fmt(col: str, pct: bool = True, digits: int = 1) -> str:
        val = row.get(col)
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            return "n/a"
        if pct:
            return f"{float(val) * 100:.{digits}f}%"
        return f"{float(val):.{digits}f}"

    parts = [
        f"- {symbol}: score_z {_fmt('score_z', pct=False)}, value_gap {_fmt('value_gap')},",
        f"  ret252 {_fmt('ret252')}, vol63 {_fmt('vol63')}, beta {_fmt('beta_to_market', pct=False, digits=2)}",
    ]
    return "\n".join(parts)


def _compose_prompt(
    panel: pd.DataFrame,
    candidates: Iterable[str],
    event_details: Mapping[str, Mapping[str, object]] | None,
    macro_stance: str,
) -> str:
    lines: List[str] = [
        "You are a super-expert long-term equity investor with a 12-month horizon.",
        "Goal: build a concentrated portfolio that can outperform (generate positive delta) over the next year.",
        "Adjust weights sparingly; prefer holding existing winners unless the thesis deteriorates.",
        f"Macro stance: {macro_stance}.",
        "For each candidate below you will see key metrics (composite score, value gap, trailing return, volatility, beta).",
        "Return JSON with `targets`: an array of objects {symbol, weight} where weights are percentages (0-1) that sum to <= 1.",
        "You may allocate zero weight to any name that lacks edge. Focus on 5-10 tickers total.",
        "Explain any major tilts in the optional notes field.",
        "Candidates:",
    ]
    seen = set()
    for sym in candidates:
        sym_u = str(sym).upper()
        if sym_u in seen:
            continue
        seen.add(sym_u)
        lines.append(_compose_symbol_block(panel, sym_u))
        if event_details and sym_u in event_details:
            summary = (event_details.get(sym_u) or {}).get("summary")
            if summary:
                lines.append(f"  Event insight: {summary}")
    return "\n".join(lines)


def _llm_portfolio(
    panel: pd.DataFrame,
    candidates: Iterable[str],
    event_details: Mapping[str, Mapping[str, object]] | None,
    macro_stance: str,
    model: str,
    reasoning_effort: str,
    cap: float,
) -> PortfolioPlan:
    cli = _client()
    if cli is None:
        raise RuntimeError("OpenAI client unavailable")

    prompt = _compose_prompt(panel, candidates, event_details, macro_stance)
    schema = {
        "name": "LongTermPlan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["symbol", "weight"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["targets"],
        },
    }

    response = cli.responses.create(  # type: ignore[call-arg]
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {
                "role": "system",
                "content": "Act as a discretionary portfolio manager focused on 12-month alpha with low turnover.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )

    output_text = getattr(response, "output_text", None)
    if not output_text:
        # Responses API may expose parsed schema directly
        out = getattr(response, "output", None)
        if out:
            for item in out:
                content = getattr(item, "content", None) or []
                for part in content:
                    parsed = getattr(part, "parsed", None)
                    if parsed is not None:
                        output_text = json.dumps(parsed)
                        break
                if output_text:
                    break
    if not output_text:
        raise RuntimeError("LLM returned empty payload")

    payload = json.loads(output_text)
    targets = payload.get("targets") or []
    weights: Dict[str, float] = {}
    for item in targets:
        sym = str(item.get("symbol", "")).upper()
        if not sym:
            continue
        try:
            weight = float(item.get("weight", 0.0))
        except Exception:
            continue
        weights[sym] = max(0.0, weight)

    weights = _normalise(weights, cap)
    notes = str(payload.get("notes", "")).strip()
    usage = getattr(response, "usage", None)
    meta = {
        "model": model,
        "effort": reasoning_effort,
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
    }
    return PortfolioPlan(weights=weights, notes=notes, meta=meta)


def build_long_term_portfolio(
    panel: pd.DataFrame,
    candidates: Iterable[str],
    *,
    event_details: Mapping[str, Mapping[str, object]] | None = None,
    macro_stance: str = "balanced",
    model: str = "gpt-5",
    reasoning_effort: str = "medium",
    weight_cap: float = 0.25,
    fallback_top_k: int = 5,
) -> PortfolioPlan:
    """Return target weights oriented around a one-year horizon."""

    try:
        plan = _llm_portfolio(
            panel,
            candidates,
            event_details,
            macro_stance,
            model,
            reasoning_effort,
            weight_cap,
        )
        if plan.weights:
            return plan
    except Exception as exc:
        error_meta = {
            "error": str(exc),
            "model": model,
            "reasoning": reasoning_effort,
        }
        # Fallback to deterministic ranking
        ranked = (
            panel.loc[list(candidates)]["score_z"].sort_values(ascending=False)
            if len(panel.index.intersection(list(candidates)))
            else pd.Series(dtype=float)
        )
        take = ranked.head(max(1, fallback_top_k))
        fallback_weights = _normalise({sym: 1.0 for sym in take.index}, weight_cap)
        return PortfolioPlan(
            weights=fallback_weights,
            notes="Fallback equal-weight plan (LLM unavailable).",
            meta=error_meta,
        )

    return PortfolioPlan(weights={}, notes="LLM returned no targets", meta={})


def blend_with_existing(
    proposed: Mapping[str, float],
    current: Mapping[str, float],
    *,
    max_step: float,
    ignore_band: float,
) -> Dict[str, float]:
    """Move weights toward the proposed plan while capping turnover."""

    all_syms = sorted({*proposed.keys(), *current.keys()})
    blended: Dict[str, float] = {}
    for sym in all_syms:
        prev = float(current.get(sym, 0.0))
        target = float(proposed.get(sym, 0.0))
        delta = target - prev
        if abs(delta) <= ignore_band:
            blended[sym] = prev
            continue
        step = max(-max_step, min(max_step, delta))
        blended[sym] = prev + step
        if abs(blended[sym]) < 1e-6:
            blended[sym] = 0.0
    total = sum(max(0.0, w) for w in blended.values())
    if total > 1.0 and total > 0:
        blended = {sym: max(0.0, w) / total for sym, w in blended.items()}
    return blended
