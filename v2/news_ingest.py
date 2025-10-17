from __future__ import annotations
"""
News ingestion via RSS feeds for Deltatrade Step 2.
No paid APIs required. Uses feedparser.
"""
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import feedparser  # type: ignore
import pytz

UTC = pytz.UTC


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("\xa0", " ").strip()


def _dt_from_entry(e: Any) -> Optional[datetime]:
    ts = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    if ts:
        return datetime.fromtimestamp(time.mktime(ts), UTC)
    return None


def _normalize(url: str, e: Any) -> Dict[str, Any]:
    dt = _dt_from_entry(e)
    title = (e.get("title") or "").strip()
    summary = _strip_html(e.get("summary") or e.get("description") or "")
    link = (e.get("link") or "").strip() or url
    uid = (e.get("id") or link or f"{title}-{dt}").strip()
    source = urlparse(url).netloc
    return {
        "id": uid,
        "title": title,
        "summary": summary[:600],
        "link": link,
        "published": dt.isoformat() if dt else None,
        "source": source,
    }


def _within_lookback(dt_iso: Optional[str], now_utc: datetime, lookback_min: int) -> bool:
    if not dt_iso:
        return False
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    except Exception:
        return False
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    return now_utc - dt <= timedelta(minutes=lookback_min)


def load_cache(path: str) -> Dict[str, float]:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: Dict[str, float], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def gather_news(
    sources: List[str],
    lookback_min: int = 240,
    max_items: int = 20,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch + normalize + filter recent items across RSS sources.
    """
    now_utc = datetime.utcnow().replace(tzinfo=UTC)
    include = [k.lower() for k in (include_keywords or [])]
    exclude = [k.lower() for k in (exclude_keywords or [])]
    cache = load_cache(cache_path) if cache_path else {}
    seen = set(cache.keys())

    items: List[Dict[str, Any]] = []
    for url in sources or []:
        try:
            fp = feedparser.parse(url)  # robust to failures
            for e in getattr(fp, "entries", [])[:50]:
                d = _normalize(url, e)
                if not d.get("published"):
                    continue
                if not _within_lookback(d["published"], now_utc, lookback_min):
                    continue
                key = (d.get("link") or d.get("id") or "").strip()
                if not key or key in seen:
                    continue
                text = f"{d.get('title','')} {d.get('summary','')}".lower()
                if include and not any(k in text for k in include):
                    continue
                if exclude and any(k in text for k in exclude):
                    continue
                items.append(d)
                seen.add(key)
        except Exception:
            # ignore individual source failure; continue
            continue

    # sort newest first, cap
    items.sort(key=lambda x: x.get("published") or "", reverse=True)
    items = items[:max_items]

    # update cache with TTL ~ 2 days
    if cache_path:
        now_ts = now_utc.timestamp()
        for d in items:
            k = (d.get("link") or d.get("id") or "").strip()
            if k:
                cache[k] = now_ts
        # prune old
        cutoff = now_ts - 2 * 86400
        cache = {k: v for k, v in cache.items() if v >= cutoff}
        save_cache(cache, cache_path)

    return items
