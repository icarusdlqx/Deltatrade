from __future__ import annotations

"""Optimizer wrappers that delegate to the consolidated CVXPY implementation."""

from typing import List, Optional

import numpy as np

from .consolidated import OPT, OnlineRidgeBlender, OptParams, optimize_weights as _optimize_weights

__all__ = ["OPT", "OptParams", "optimize_weights", "OnlineRidgeBlender"]


def optimize_weights(
    alpha_bps: np.ndarray,
    Sigma_daily: np.ndarray,
    w_prev: np.ndarray,
    prices: np.ndarray,
    adv_dollars: np.ndarray,
    spread_bps: np.ndarray,
    kappa: np.ndarray,
    psi: float,
    tickers: List[str],
    *,
    sector_expo: Optional[np.ndarray] = None,
    sector_names: Optional[List[str]] = None,
    etf_mask: Optional[np.ndarray] = None,
    equity: float = 1.0,
) -> np.ndarray:
    return _optimize_weights(
        alpha_bps,
        Sigma_daily,
        w_prev,
        prices,
        adv_dollars,
        spread_bps,
        kappa,
        psi,
        tickers,
        sector_expo=sector_expo,
        sector_names=sector_names,
        etf_mask=etf_mask,
        equity=equity,
    )
