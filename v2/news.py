from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List

# Robust import across alpaca-py versions
try:
    from alpaca.data.historical import NewsClient
except Exception:
    try:
        from alpaca.data.historical.news import NewsClient
    except Exception:
        NewsClient = None

def fetch_news_map(symbols: List[str], days: int, api_key: str, api_secret: str) -> Dict[str, List[dict]]:
    if NewsClient is None:
        return {s: [] for s in symbols}
    client = NewsClient(api_key=api_key, secret_key=api_secret)
    start = datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=days)
    # Convert to UTC for API calls
    start = start.astimezone(timezone.utc)
    out: Dict[str, List[dict]] = {s: [] for s in symbols}
    for sym in symbols:
        try:
            # Try dict-style (newer), fallback to request object (older)
            items = client.get_news({"symbols":[sym], "start":start, "limit":50})
            if not items:
                from alpaca.data.requests import NewsRequest
                items = client.get_news(NewsRequest(symbols=[sym], start=start, limit=50))
            out[sym] = [
                {"id": getattr(n, "id", None), "headline": getattr(n, "headline", ""), "summary": getattr(n, "summary", "") or "", "created_at": str(getattr(n, "created_at", ""))}
                for n in items
            ]
        except Exception:
            out[sym] = []
    return out
