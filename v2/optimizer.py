from __future__ import annotations
import numpy as np
from typing import List
from scipy.optimize import minimize

def solve_weights(alpha: np.ndarray, Sigma: np.ndarray, w_prev: np.ndarray,
                  name_max: float, sector_ids: List[int], sector_max: float,
                  lam_risk: float, tau_turnover: float) -> np.ndarray:
    n = len(alpha)
    sec_ids = np.array(sector_ids, dtype=int)
    unique_secs = np.unique(sec_ids)

    def objective(w):
        risk = w @ Sigma @ w
        turn = np.abs(w - w_prev).sum()
        return -(alpha @ w - lam_risk * risk - tau_turnover * turn)

    cons = [{"type":"ineq","fun": lambda w: 1.0 - np.sum(w)}]  # sum w <= 1
    for sid in unique_secs:
        idx = (sec_ids == sid).astype(float)
        cons.append({"type":"ineq","fun": (lambda w, idx=idx: sector_max - idx @ w)})

    bounds = [(0.0, name_max) for _ in range(n)]
    w0 = np.clip(w_prev, 0.0, name_max)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter":300})
    w = res.x if res.success else w0
    w = np.clip(w, 0.0, name_max)
    s = w.sum()
    if s > 1.0:
        w = w / s
    return w

def scale_to_target_vol(w: np.ndarray, Sigma: np.ndarray, target_vol_daily: float) -> np.ndarray:
    vol = float(np.sqrt(w @ Sigma @ w))
    if vol <= 1e-9:
        return w
    k = target_vol_daily / vol
    return w * min(1.0, k)
