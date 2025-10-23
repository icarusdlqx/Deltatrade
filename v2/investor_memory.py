from __future__ import annotations
"""Utilities for recalling prior investment rationales from episode logs."""

import json
from pathlib import Path
from typing import Any, Dict, List


def _tail_lines(path: str, n: int) -> List[str]:
    try:
        p = Path(path)
        if not p.exists():
            return []
        data = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        return data[-abs(int(n)) :]
    except Exception:
        return []


def _load_recent_episodes(path: str, n: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in _tail_lines(path, n):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _extract_rationales(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return {SYMBOL: {as_of, rationale}} for the most recent mentions."""

    latest: Dict[str, Dict[str, Any]] = {}
    for ep in reversed(episodes):
        order_notes = ep.get("order_rationales") or {}
        if isinstance(order_notes, dict):
            for sym, why in order_notes.items():
                ticker = str(sym).upper()
                if ticker and ticker not in latest and why:
                    latest[ticker] = {
                        "as_of": ep.get("as_of"),
                        "rationale": str(why)[:600],
                    }
        advisor = ep.get("advisor_report") if isinstance(ep.get("advisor_report"), dict) else None
        actions = advisor.get("actions") if isinstance(advisor, dict) else None
        if isinstance(actions, list):
            for action in actions:
                ticker = str(action.get("ticker", "")).upper()
                if not ticker or ticker in latest:
                    continue
                why = action.get("rationale") or ""
                if why:
                    latest[ticker] = {
                        "as_of": ep.get("as_of"),
                        "rationale": str(why)[:600],
                    }
    return latest


def build_position_memory(episodes_path: str, lookback_lines: int = 120) -> Dict[str, Dict[str, Any]]:
    """Produce a compact rationale map for advisor prompts."""

    episodes = _load_recent_episodes(episodes_path, lookback_lines)
    return _extract_rationales(episodes)
