from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import zscore, winsorize, ann_vol_from_daily, max_drawdown
from .datafeed import get_daily_bars


def bars_from_multiindex(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in raw.groupby(level=0):
        dfg = df.reset_index(level=0, drop=True).sort_index()
        dfg.columns = [str(c).lower() for c in dfg.columns]
        out[str(sym)] = dfg
    return out


def fetch_bars(symbols: List[str], lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
    raw = get_daily_bars(symbols, lookback_days=lookback_days)
    return bars_from_multiindex(raw)

def compute_daily_returns(bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rets = {}
    for s, df in bars.items():
        if df is None or len(df) <= 1:
            continue
        dfg = df.rename(columns=str.lower)
        if "close" in dfg.columns:
            rets[s] = dfg["close"].pct_change()
    return pd.DataFrame(rets).dropna(how="all")

def moving_average(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n//2)).mean()

def compute_panel(
    bars: Dict[str, pd.DataFrame], spy: str = "SPY",
    fast: int = 20, slow: int = 50,
    resid_lookback: int = 63, reversal_days: int = 3,
    winsor_pct: float = 0.02
) -> pd.DataFrame:
    normed: Dict[str, pd.DataFrame] = {}
    for s, df in bars.items():
        if df is None or len(df) == 0:
            continue
        dfg = df.rename(columns=str.lower)
        normed[s] = dfg
    bars = normed
    rows = []
    rets_df = compute_daily_returns(bars)
    last_prices = {s: df["close"].iloc[-1] for s, df in bars.items() if "close" in df.columns and len(df) > 0}

    # Market proxy
    if spy in rets_df.columns:
        mkt = rets_df[spy].fillna(0).to_frame("mkt")
    else:
        mkt = pd.DataFrame({"mkt": rets_df.mean(axis=1)})

    for s, df in bars.items():
        if len(df) < 60 or "close" not in df.columns:
            continue
        px = df["close"]
        ret = px.pct_change()
        ret21 = (px.iloc[-1] / px.iloc[-22] - 1.0) if len(px) > 22 else np.nan
        ret63 = (px.iloc[-1] / px.iloc[-64] - 1.0) if len(px) > 64 else np.nan
        ret126 = (px.iloc[-1] / px.iloc[-127] - 1.0) if len(px) > 127 else np.nan
        ret252 = (px.iloc[-1] / px.iloc[-253] - 1.0) if len(px) > 253 else np.nan
        slow_ma = moving_average(px, slow)
        trend = (moving_average(px, fast).iloc[-1] - slow_ma.iloc[-1]) / (slow_ma.iloc[-1] + 1e-9)
        rev = -ret.rolling(reversal_days).sum().iloc[-1]
        vol20 = ann_vol_from_daily(ret.tail(21))
        vol63 = ann_vol_from_daily(ret.tail(63))
        mdd = max_drawdown(ret.tail(252))
        qual = (
            (px.iloc[-1] / px.iloc[-127] - 1.0)
            / (ann_vol_from_daily(ret.tail(63)) + 1e-9)
        ) if len(px) > 127 else np.nan

        rolling_high = px.rolling(252, min_periods=40).max()
        value_gap = (
            (rolling_high.iloc[-1] - px.iloc[-1]) / rolling_high.iloc[-1]
            if len(rolling_high.dropna()) > 0 and rolling_high.iloc[-1] > 0
            else np.nan
        )

        # Residual momentum vs market and beta to the macro proxy
        df_ret = pd.concat([ret, mkt["mkt"]], axis=1).dropna()
        resid_mom = np.nan
        beta_to_market = np.nan
        if len(df_ret) >= resid_lookback:
            X = df_ret["mkt"].tail(resid_lookback).values.reshape(-1, 1)
            y = df_ret.iloc[-resid_lookback:, 0].values
            model = LinearRegression().fit(X, y)
            resid = y - model.predict(X)
            resid_mom = float(np.nanmean(resid[-21:]) * 21)
            try:
                beta_to_market = float(model.coef_[0])
            except Exception:
                beta_to_market = np.nan

        rows.append(
            {
                "symbol": s,
                "last_price": last_prices.get(s, np.nan),
                "ret21": ret21,
                "ret63": ret63,
                "ret126": ret126,
                "ret252": ret252,
                "trend": trend,
                "reversal": rev,
                "vol20": vol20,
                "vol63": vol63,
                "maxdd": mdd,
                "qual126": qual,
                "resid_mom": resid_mom,
                "beta_to_market": beta_to_market,
                "value_gap": value_gap,
            }
        )

    panel = pd.DataFrame(rows).set_index("symbol")
    value_component = winsorize(
        panel["value_gap"].astype(float).fillna(panel["value_gap"].median()).fillna(0.0),
        winsor_pct,
    )
    quality_component = winsorize(
        panel["qual126"].astype(float).fillna(panel["qual126"].median()).fillna(0.0),
        winsor_pct,
    )
    momentum_component = winsorize(
        (panel["ret126"].fillna(0) + 0.5 * panel["resid_mom"].fillna(0)),
        winsor_pct,
    )
    growth_component = winsorize(
        panel["ret252"].astype(float).fillna(panel["ret252"].median()).fillna(0.0),
        winsor_pct,
    )
    stability_component = winsorize(
        -panel["vol63"].astype(float).fillna(panel["vol63"].median()).fillna(0.0),
        winsor_pct,
    )
    drawdown_component = winsorize(
        -panel["maxdd"].abs().astype(float).fillna(panel["maxdd"].abs().median()).fillna(0.0),
        winsor_pct,
    )
    macro_resilience_component = winsorize(
        -panel["beta_to_market"].fillna(1.0).sub(1.0).abs(),
        winsor_pct,
    )

    panel["value_component"] = value_component
    panel["quality_component"] = quality_component
    panel["momentum_component"] = momentum_component
    panel["growth_component"] = growth_component
    panel["stability_component"] = stability_component
    panel["macro_resilience_component"] = macro_resilience_component

    score_raw = (
        0.28 * value_component
        + 0.22 * quality_component
        + 0.18 * momentum_component
        + 0.12 * growth_component
        + 0.10 * stability_component
        + 0.06 * drawdown_component
        + 0.04 * macro_resilience_component
    )
    panel["score_raw"] = score_raw
    panel["score_z"]  = zscore(score_raw.fillna(score_raw.median()))
    # ATR proxy
    atrs = {}
    for s, df in bars.items():
        if set(["high","low","close"]).issubset(df.columns) and len(df) >= 15:
            hi, lo, cl = df["high"], df["low"], df["close"]
            prev = cl.shift(1)
            tr = pd.concat([(hi-lo), (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
            atrs[s] = float(tr.rolling(14).mean().iloc[-1])
    panel["atr14"] = pd.Series(atrs)
    return panel
