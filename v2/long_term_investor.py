from __future__ import annotations

"""Helpers for long-horizon, thesis-driven portfolio decisions."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class MacroOutlook:
    stance: str
    narrative: str
    yoy_return: float | None
    vol63: float | None
    drawdown: float | None


def _safe_pct(val: float | int | None, *, pct: bool = True) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "n/a"
    if pct:
        return f"{float(val) * 100.0:.1f}%"
    return f"{float(val):.2f}"


def _macro_drawdown(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    running_max = series.cummax()
    drawdowns = (series / running_max) - 1.0
    return float(drawdowns.min()) if not drawdowns.empty else None


def _compute_macro_outlook(spy_prices: Optional[pd.Series]) -> MacroOutlook:
    if spy_prices is None or len(spy_prices.dropna()) < 40:
        return MacroOutlook(
            stance="data-limited",
            narrative="Insufficient SPY history to infer macro backdrop; defaulting to neutral risk posture.",
            yoy_return=None,
            vol63=None,
            drawdown=None,
        )

    px = spy_prices.dropna().astype(float)
    returns = px.pct_change().dropna()
    if len(px) > 252:
        yoy = float(px.iloc[-1] / px.iloc[-252] - 1.0)
    else:
        yoy = float(px.iloc[-1] / px.iloc[0] - 1.0)
    vol63 = float(returns.tail(63).std(ddof=0) * np.sqrt(252.0)) if len(returns) >= 21 else np.nan
    drawdown = _macro_drawdown(px)

    stance = "balanced"
    rationale_parts = [
        f"SPY 1y return {_safe_pct(yoy)}",
        f"63d vol {_safe_pct(vol63)}",
    ]
    if drawdown is not None:
        rationale_parts.append(f"drawdown {_safe_pct(drawdown)}")

    if yoy > 0.08 and (np.isnan(vol63) or vol63 < 0.25) and (drawdown is None or drawdown > -0.20):
        stance = "constructive"
        rationale_parts.append("supportive growth backdrop")
    elif yoy < -0.05 or (drawdown is not None and drawdown < -0.25) or (not np.isnan(vol63) and vol63 > 0.35):
        stance = "defensive"
        rationale_parts.append("elevated macro risk")

    narrative = ", ".join(rationale_parts)
    return MacroOutlook(stance=stance, narrative=narrative, yoy_return=yoy, vol63=vol63, drawdown=drawdown)


def _symbol_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    def _get(key: str) -> float:
        val = row.get(key) if isinstance(row, Mapping) else None
        try:
            return float(val)
        except Exception:
            return float("nan")

    metrics = {
        "value_gap": _get("value_gap"),
        "quality": _get("qual126"),
        "momentum_6m": _get("ret126"),
        "growth_1y": _get("ret252"),
        "volatility": _get("vol63"),
        "drawdown": _get("maxdd"),
        "macro_resilience": _get("beta_to_market"),
        "composite_score": _get("score_z"),
    }
    for key, val in list(metrics.items()):
        if isinstance(val, float) and not np.isfinite(val):
            metrics[key] = float("nan")
    return metrics


def _thesis_from_metrics(symbol: str, metrics: Mapping[str, float], macro: MacroOutlook, event_summary: str | None = None) -> str:
    sentences: list[str] = []
    value_gap = metrics.get("value_gap")
    if value_gap is not None and np.isfinite(value_gap):
        if value_gap >= 0:
            sentences.append(
                f"Trades at {_safe_pct(value_gap)} below its 52-week high, offering a valuation cushion for long-horizon investors."
            )
        else:
            sentences.append(
                f"Maintains pricing above the prior 52-week high (premium of {_safe_pct(abs(value_gap))}), reflecting strong market conviction."
            )
    growth = metrics.get("growth_1y")
    if growth is not None and np.isfinite(growth):
        sentences.append(f"Delivered {_safe_pct(growth)} total return over the past year, consistent with structural momentum.")
    quality = metrics.get("quality")
    if quality is not None and np.isfinite(quality):
        sentences.append(f"Quality/volatility ratio of {_safe_pct(quality, pct=False)} supports durable cash-flow generation.")
    volatility = metrics.get("volatility")
    if volatility is not None and np.isfinite(volatility):
        sentences.append(f"Annualized volatility around {_safe_pct(volatility)} keeps portfolio risk controlled.")
    drawdown = metrics.get("drawdown")
    if drawdown is not None and np.isfinite(drawdown):
        sentences.append(f"Recent max drawdown of {_safe_pct(drawdown)} informs position sizing under stress scenarios.")
    beta = metrics.get("macro_resilience")
    if beta is not None and np.isfinite(beta):
        sentences.append(f"Market beta near {beta:.2f} balances exposure against the current macro stance ({macro.stance}).")

    if event_summary:
        if not isinstance(event_summary, str):
            event_summary = str(event_summary)
        sentences.append(event_summary.strip())

    sentences.append(f"Macro lens: {macro.narrative} (stance: {macro.stance}).")

    return " ".join(sentences)


def build_long_term_outlook(
    panel: pd.DataFrame,
    symbols: Iterable[str],
    spy_bars: Optional[pd.DataFrame] = None,
    event_details: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    spy_close = None
    if spy_bars is not None and "close" in spy_bars.columns:
        spy_close = spy_bars["close"]
    macro = _compute_macro_outlook(spy_close)

    theses: Dict[str, Any] = {}
    for sym in symbols:
        if sym not in panel.index:
            continue
        row = panel.loc[sym]
        metrics = _symbol_metrics(row)
        event_summary = None
        if event_details and sym in event_details:
            det = event_details.get(sym) or {}
            event_summary = det.get("summary") or det.get("thesis")
        thesis_text = _thesis_from_metrics(sym, metrics, macro, event_summary)
        theses[sym] = {
            "summary": thesis_text,
            "metrics": metrics,
            "macro_stance": macro.stance,
        }

    return {
        "macro": {
            "stance": macro.stance,
            "narrative": macro.narrative,
            "yoy_return": macro.yoy_return,
            "vol63": macro.vol63,
            "drawdown": macro.drawdown,
        },
        "theses": theses,
    }

