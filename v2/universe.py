from __future__ import annotations

"""Universe helpers backed by the 2025-10-28 snapshot."""

from typing import List, Set

from .consolidated import S_AND_P_500_TICKERS, TOP_50_ETF_TICKERS


def load_sp500(refresh_days: int = 0) -> List[str]:  # noqa: ARG001 (kept for API compatibility)
    """Return the hard-coded S&P 500 constituent list."""
    return list(S_AND_P_500_TICKERS)


def load_top50_etfs() -> List[str]:
    """Return the hard-coded Top-50 ETF list."""
    return list(TOP_50_ETF_TICKERS)


def build_universe() -> List[str]:
    sp = set(load_sp500())
    etf = set(load_top50_etfs())
    uni: Set[str] = (sp | etf) - {None, "", "N/A"}
    return sorted(uni)


if __name__ == "__main__":
    u = build_universe()
    print("Universe size:", len(u))
    print(u[:30])
