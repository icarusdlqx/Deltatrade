from __future__ import annotations
"""
Step 2 integration: fetch news + AI score, return a compact report.
"""
from typing import Any, Dict, List
from datetime import datetime

import pytz

from .settings_bridge import get_cfg
from .news_ingest import gather_news
from .news_llm import score_news, aggregate_bias

ET = pytz.timezone("US/Eastern")


def run_news_step(cfg=None) -> Dict[str, Any]:
    cfg = cfg or get_cfg()
    if not getattr(cfg, "ENABLE_NEWS_CHECK", True):
        return {"ok": False, "disabled": True, "items": [], "as_of": datetime.now(ET).isoformat()}
    items = gather_news(
        sources=getattr(cfg, "NEWS_SOURCES", []),
        lookback_min=int(getattr(cfg, "NEWS_LOOKBACK_MIN", 240)),
        max_items=int(getattr(cfg, "NEWS_MAX_PER_RUN", 20)),
        include_keywords=getattr(cfg, "NEWS_KEYWORDS_INCLUDE", []),
        exclude_keywords=getattr(cfg, "NEWS_KEYWORDS_EXCLUDE", []),
        cache_path=getattr(cfg, "NEWS_CACHE_PATH", None),
    )
    scored = score_news(items)
    agg = aggregate_bias(scored)
    # stitch fields back to items
    stitched: List[Dict[str, Any]] = []
    for it, sc in zip(items, scored):
        d = dict(it)
        d.update(sc)
        stitched.append(d)
    return {
        "ok": True,
        "as_of": datetime.now(ET).isoformat(),
        "count": len(stitched),
        "aggregate": agg,
        "items": stitched,
    }


def attach_news_to_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    try:
        report = run_news_step()
        ep = dict(ep or {})
        ep["news_report"] = report
        # provide a top-line hint for the dashboard
        ep["news_risk_bias"] = report.get("aggregate", {}).get("risk_bias_mean", 0.0)
        return ep
    except Exception as e:
        # never fail the run due to news ingestion
        ep = dict(ep or {})
        ep["news_report"] = {"ok": False, "error": str(e), "items": []}
        return ep
