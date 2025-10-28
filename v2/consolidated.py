from __future__ import annotations

"""Consolidated trading utilities for the S&P 500 + Top-50 ETF program.

This module centralises configuration constants and helper utilities that
implement the 2025-10-28 policy drop-in.  It does not wire itself into the
orchestrator directly, but exposes reusable primitives (universe snapshots,
blending, optimizer, execution helpers) that higher layers can adopt.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import datetime as dt
import json

try:  # pragma: no cover - optional dependency for environments without cvxpy
    import cvxpy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
import numpy as np

# ---------------------------------------------------------------------------
# 0) Schedule & global policy
# ---------------------------------------------------------------------------
SCHEDULE_ET = ["10:05", "14:35", "15:45"]

POLICY = {
    "policy_version": "2025-10-28.a",
    "killswitch_bench_only": False,
}

# ---------------------------------------------------------------------------
# 1) Universe snapshot (hard-coded for determinism)
# ---------------------------------------------------------------------------
UNIVERSE_SNAPSHOT_DATE = "2025-10-28"

S_AND_P_500_TICKERS: List[str] = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES",
    "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY",
    "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF.B", "BG", "BIIB", "BK",
    "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR",
    "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COP", "COR",
    "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA",
    "CVS", "CVX", "CZR", "D", "DAL", "DASH", "DAY", "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI",
    "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
    "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR",
    "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FOX", "FOXA", "FRT", "FSLR", "FTNT",
    "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY",
    "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
    "JCI", "JKHY", "JNJ", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW",
    "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO",
    "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS",
    "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN",
    "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS",
    "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR",
    "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR",
    "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG",
    "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW",
    "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC",
    "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT",
    "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS",
]

TOP_50_ETF_TICKERS: List[str] = [
    "SPY", "IVV", "VOO", "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "SOXL", "SOXS",
    "UVXY", "SVXY", "XLF", "XLE", "XLK", "XLY", "XLV", "XLI", "XLP", "XLU",
    "XLC", "XLB", "XLRE", "SMH", "SOXX", "XBI", "IEMG", "IEFA", "EEM", "EFA",
    "VEA", "VWO", "IWB", "IWD", "IWF", "HYG", "LQD", "TLT", "SHY", "TIP",
    "GLD", "IAU", "SLV", "GDX", "USO", "UNG", "UUP", "KRE", "KBE", "ARKK",
]

ALL_TICKERS: List[str] = sorted(set(S_AND_P_500_TICKERS) | set(TOP_50_ETF_TICKERS))

# ---------------------------------------------------------------------------
# 2) Factors & news mapping
# ---------------------------------------------------------------------------


@dataclass
class FactorParams:
    residual_mom_lookback: int = 63
    trend_fast: int = 20
    trend_slow: int = 50
    reversal_lookback: int = 3
    quality_ret_days: int = 252
    quality_vol_days: int = 63
    combine: str = "rank_equal_weight"


@dataclass
class NewsParams:
    min_confidence: float = 0.60
    default_half_life_days: float = 3.0
    bucket_betas_bps: Dict[str, float] = field(
        default_factory=lambda: {
            "earnings": 9.5,
            "guidance": 7.0,
            "mna": 12.0,
            "regulatory": 4.0,
            "macro": 3.5,
            "default": 5.0,
        }
    )
    max_abs_news_bps_per_name: float = 20.0
    dedupe_by_event_id: bool = True


FACTOR = FactorParams()
NEWS = NewsParams()


def news_alpha(
    direction_score: float,
    bucket: str,
    confidence: float,
    half_life_days: Optional[float],
    age_days: float,
) -> float:
    if confidence < NEWS.min_confidence:
        return 0.0
    tau = max(half_life_days or NEWS.default_half_life_days, 0.5)
    s = np.tanh(direction_score / 2.0)
    beta = NEWS.bucket_betas_bps.get(bucket, NEWS.bucket_betas_bps["default"])
    val = s * beta * confidence * np.exp(-age_days / tau)
    return float(val)


@dataclass
class BlendParams:
    ridge_lambda: float = 10.0
    decay: float = 0.97
    min_weight_news: float = 0.0
    max_weight_news: float = 0.6
    init_weights: Tuple[float, float] = (0.6, 0.4)


BLEND = BlendParams()


class OnlineRidgeBlender:
    """Online ridge regression for combining cross-sectional and news alpha sleeves."""

    def __init__(
        self,
        params: BlendParams = BLEND,
        *,
        theta: Optional[Iterable[float]] = None,
        S: Optional[Iterable[Iterable[float]]] = None,
        b: Optional[Iterable[float]] = None,
    ) -> None:
        self.p = params
        self.theta = np.array(theta if theta is not None else self.p.init_weights, dtype=float)
        self.S = np.array(S if S is not None else np.eye(2) * self.p.ridge_lambda, dtype=float)
        self.b = np.array(b if b is not None else self.S @ self.theta, dtype=float)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, list]:
        return {
            "theta": self.theta.tolist(),
            "S": self.S.tolist(),
            "b": self.b.tolist(),
        }

    def update_with_realized(
        self,
        alpha_cs_hist: np.ndarray,
        alpha_news_hist: np.ndarray,
        realized_excess_ret_hist: np.ndarray,
    ) -> None:
        if len(realized_excess_ret_hist) == 0:
            return
        X = np.vstack([alpha_cs_hist, alpha_news_hist]).T
        y = realized_excess_ret_hist
        self.S *= self.p.decay
        self.b *= self.p.decay
        self.S += X.T @ X
        self.b += X.T @ y
        M = self.S + np.eye(2) * self.p.ridge_lambda
        self.theta = np.linalg.solve(M, self.b)
        self.theta[1] = float(np.clip(self.theta[1], self.p.min_weight_news, self.p.max_weight_news))
        self.theta[0] = max(self.theta[0], 0.0)
        s = self.theta.sum()
        if s > 1.0:
            self.theta /= s

    def combine(self, alpha_cs_vec: np.ndarray, alpha_news_vec: np.ndarray) -> np.ndarray:
        w_cs, w_news = self.theta
        return w_cs * alpha_cs_vec + w_news * alpha_news_vec


# ---------------------------------------------------------------------------
# 3) Optimizer
# ---------------------------------------------------------------------------


@dataclass
class OptParams:
    lambda_var: float = 8.0
    lambda_turnover: float = 0.0025
    position_cap: float = 0.20
    sector_cap: float = 0.30
    long_only: bool = True
    include_txn_costs_in_objective: bool = True
    bench_ticker: str = "SPY"
    bench_max_weight: float = 0.60
    vol_target_annual: float = 0.22
    etf_dominance_threshold: float = 0.50


OPT = OptParams()


def _optimize_weights_fallback(
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
    from scipy.optimize import minimize  # pragma: no cover

    n = len(tickers)
    if n == 0:
        return np.array([])

    alpha_dec = alpha_bps / 10000.0

    def objective(w: np.ndarray) -> float:
        delta_w = w - w_prev
        var_term = OPT.lambda_var * (w @ Sigma_daily @ w)
        to_term = OPT.lambda_turnover * np.sum(np.abs(delta_w))
        txn_pen = 0.0
        if OPT.include_txn_costs_in_objective:
            spread_pen = spread_bps @ np.abs(delta_w)
            adv_safe = np.maximum(adv_dollars, 1.0)
            dollar_change = equity * np.abs(delta_w)
            impact_pen = kappa @ np.power(dollar_change / adv_safe, psi) * 10000.0
            txn_pen = (spread_pen + impact_pen) / 10000.0
        return -(alpha_dec @ w) + var_term + to_term + txn_pen

    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - float(np.sum(w))},
    ]

    if OPT.long_only:
        bounds = [(0.0, OPT.position_cap) for _ in range(n)]
    else:
        bounds = [(-OPT.position_cap, OPT.position_cap) for _ in range(n)]

    if OPT.bench_ticker in tickers:
        idx = tickers.index(OPT.bench_ticker)
        constraints.append({"type": "ineq", "fun": lambda w, i=idx: OPT.bench_max_weight - w[i]})

    enable_sector_caps = False
    if (
        sector_expo is not None
        and sector_names is not None
        and etf_mask is not None
        and len(sector_names) == sector_expo.shape[0]
    ):
        etf_share_prev = float(np.sum(w_prev[etf_mask.astype(bool)])) if etf_mask.size else 0.0
        enable_sector_caps = etf_share_prev < OPT.etf_dominance_threshold
    if enable_sector_caps:
        for row in sector_expo:
            constraints.append({"type": "ineq", "fun": lambda w, r=row: OPT.sector_cap - float(r @ w)})

    w0 = np.clip(w_prev, 0.0, OPT.position_cap)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500})
    w = res.x if res.success and isinstance(res.x, np.ndarray) else w0
    w = np.clip(w, 0.0 if OPT.long_only else -OPT.position_cap, OPT.position_cap)
    if np.sum(w) > 1.0:
        w = w / max(np.sum(w), 1e-9)

    target_daily_vol = OPT.vol_target_annual / np.sqrt(252.0)
    cur_vol = float(np.sqrt(w @ Sigma_daily @ w)) if Sigma_daily.size else 0.0
    if cur_vol > 1e-8:
        w *= min(1.0, target_daily_vol / cur_vol)
    return w


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
    if cp is None:
        return _optimize_weights_fallback(
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

    n = len(tickers)
    if alpha_bps.shape != (n,):
        raise ValueError("alpha_bps length must match tickers length")

    w = cp.Variable(n)
    delta_w = w - w_prev

    alpha_dec = alpha_bps / 10000.0
    var_term = cp.quad_form(w, Sigma_daily)
    to_term = cp.norm1(delta_w)

    if OPT.include_txn_costs_in_objective:
        spread_term_bps = spread_bps @ cp.abs(delta_w)
        adv_safe = np.maximum(adv_dollars, 1.0)
        dollar_change = equity * cp.abs(delta_w)
        impact_piece = kappa @ cp.power(dollar_change / adv_safe, psi)
        txn_bps = spread_term_bps + 10000.0 * impact_piece
    else:
        txn_bps = 0.0

    objective = cp.Maximize(
        alpha_dec @ w
        - OPT.lambda_var * var_term
        - OPT.lambda_turnover * to_term
        - (txn_bps / 10000.0)
    )

    constraints = []
    if OPT.long_only:
        constraints.append(w >= 0)
    constraints.append(w <= OPT.position_cap)
    constraints.append(cp.sum(w) <= 1.0)

    if OPT.bench_ticker in tickers:
        idx = tickers.index(OPT.bench_ticker)
        constraints.append(w[idx] <= OPT.bench_max_weight)

    if (
        sector_expo is not None
        and sector_names is not None
        and etf_mask is not None
        and len(sector_names) == sector_expo.shape[0]
    ):
        etf_share_prev = float(np.sum(w_prev[etf_mask.astype(bool)])) if etf_mask.size else 0.0
        if etf_share_prev < OPT.etf_dominance_threshold:
            for s_idx, _ in enumerate(sector_names):
                constraints.append(sector_expo[s_idx] @ w <= OPT.sector_cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, verbose=False)

    if w.value is None:
        raise RuntimeError("Optimizer failed: infeasible or solver error")

    w_opt = np.clip(w.value, 0.0, OPT.position_cap)
    target_daily_vol = OPT.vol_target_annual / np.sqrt(252.0)
    cur_vol = float(np.sqrt(w_opt @ Sigma_daily @ w_opt))
    if cur_vol > 1e-8:
        scale = min(1.0, target_daily_vol / cur_vol)
        w_opt *= scale
    return w_opt


# ---------------------------------------------------------------------------
# 4) Execution scaffolding
# ---------------------------------------------------------------------------


@dataclass
class ExecParams:
    price_anchor: str = "nbbo_mid"
    max_spread_frac: float = 0.004
    pov: float = 0.08
    child_min_notional: float = 2_000.0
    child_refresh_seconds: int = 5
    start_offset_spread_share: float = 0.25
    max_offset_spread_share: float = 0.50
    max_slippage_spread_share: float = 0.40
    time_budget_seconds: int = 300


EXEC = ExecParams()


def plan_child_orders(
    symbol: str,
    target_qty: int,
    side: str,
    nbbo_bid: float,
    nbbo_ask: float,
    est_vol_shares_per_sec: float,
) -> List[Dict[str, object]]:
    mid = 0.5 * (nbbo_bid + nbbo_ask)
    spread = max(nbbo_ask - nbbo_bid, 0.01 * mid * EXEC.max_spread_frac)
    if mid <= 0 or spread / mid > EXEC.max_spread_frac:
        return []

    pov_qty = max(int(EXEC.pov * est_vol_shares_per_sec * EXEC.child_refresh_seconds), 1)
    child_qty = max(pov_qty, int(EXEC.child_min_notional / max(mid, 1e-6)))
    plan: List[Dict[str, object]] = []
    remaining = abs(int(target_qty))
    k = 0
    while remaining > 0:
        k += 1
        this_qty = min(child_qty, remaining)
        offset_share = min(
            EXEC.start_offset_spread_share + 0.05 * (k - 1),
            EXEC.max_offset_spread_share,
        )
        limit = mid + (1 if side.upper() == "BUY" else -1) * offset_share * spread
        plan.append(
            {
                "symbol": symbol,
                "side": side.upper(),
                "qty": this_qty,
                "limit": round(limit, 4),
                "tif": "DAY",
            }
        )
        remaining -= this_qty
    return plan


# ---------------------------------------------------------------------------
# 5) Risk knobs & logging defaults
# ---------------------------------------------------------------------------


@dataclass
class RiskParams:
    stop_atr_mult_sl: float = 2.5
    stop_atr_mult_tp: float = 2.0
    time_stop_days: int = 10
    intraday_drawdown_floor_bps: int = 150
    event_blackout_days: int = 1
    block_within_minutes_of_close: int = 10


RISK = RiskParams()

LOGGING = {
    "episode_path": "data/episodes_v2.jsonl",
    "persist_inputs": True,
    "idempotent_run_lock": "data/run.lock",
    "metrics": [
        "alpha_hit_rate",
        "pnl_cs",
        "pnl_news",
        "slippage_bps",
        "fill_ratio",
        "ttf_seconds",
        "turnover",
        "constraint_utilization",
        "var_vs_target",
    ],
}


def new_episode_id() -> str:
    return f"eps_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


# ---------------------------------------------------------------------------
# 6) Alpha helpers
# ---------------------------------------------------------------------------

def build_alpha_vector(
    alpha_cs_bps: np.ndarray,
    news_items_per_name: List[List[Dict[str, object]]],
    blender: OnlineRidgeBlender,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(alpha_cs_bps)
    alpha_news = np.zeros(n, dtype=float)
    for i, items in enumerate(news_items_per_name):
        seen = set()
        total = 0.0
        for item in items:
            eid = item.get("event_id")
            if NEWS.dedupe_by_event_id and eid:
                if eid in seen:
                    continue
                seen.add(eid)
            total += news_alpha(
                float(item.get("direction_score", 0.0) or 0.0),
                str(item.get("bucket", "default")),
                float(item.get("confidence", 0.0) or 0.0),
                item.get("half_life_days"),
                float(item.get("age_days", 0.0) or 0.0),
            )
        alpha_news[i] = float(
            np.clip(total, -NEWS.max_abs_news_bps_per_name, NEWS.max_abs_news_bps_per_name)
        )
    alpha_final = blender.combine(alpha_cs_bps, alpha_news)
    return alpha_news, alpha_final


def combine_factors_as_bps(
    factors: Dict[str, np.ndarray],
    *,
    annualize_scale: float = 0.05,
) -> np.ndarray:
    if not factors:
        return np.array([])
    names = sorted(factors.keys())
    X = np.column_stack([np.asarray(factors[k], dtype=float) for k in names])
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) / max(1, X.shape[0] - 1)
    combo = ranks.mean(axis=1)
    combo = combo - combo.mean()
    return 10_000 * annualize_scale * combo


# ---------------------------------------------------------------------------
# 7) Persistence helpers for the blender
# ---------------------------------------------------------------------------

BLENDER_STATE_PATH = Path("data/blender_state.json")


def load_blender(params: BlendParams = BLEND) -> OnlineRidgeBlender:
    if BLENDER_STATE_PATH.exists():
        try:
            data = json.loads(BLENDER_STATE_PATH.read_text())
            return OnlineRidgeBlender(
                params,
                theta=data.get("theta"),
                S=data.get("S"),
                b=data.get("b"),
            )
        except Exception:
            pass
    return OnlineRidgeBlender(params)


def save_blender(blender: OnlineRidgeBlender) -> None:
    BLENDER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLENDER_STATE_PATH.write_text(json.dumps(blender.to_dict()))


__all__ = [
    "ALL_TICKERS",
    "BLEND",
    "BLENDER_STATE_PATH",
    "EXEC",
    "FACTOR",
    "LOGGING",
    "NEWS",
    "OPT",
    "OnlineRidgeBlender",
    "OptParams",
    "POLICY",
    "RISK",
    "S_AND_P_500_TICKERS",
    "SCHEDULE_ET",
    "TOP_50_ETF_TICKERS",
    "UNIVERSE_SNAPSHOT_DATE",
    "build_alpha_vector",
    "combine_factors_as_bps",
    "load_blender",
    "new_episode_id",
    "news_alpha",
    "optimize_weights",
    "plan_child_orders",
    "save_blender",
]
