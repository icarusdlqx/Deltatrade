from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import zscore, winsorize, ann_vol_from_daily, max_drawdown

def compute_daily_returns(bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rets = {}
    for s, df in bars.items():
        if "close" in df.columns and len(df) > 1:
            rets[s] = df["close"].pct_change()
    return pd.DataFrame(rets).dropna(how="all")

def moving_average(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n//2)).mean()

def compute_panel(
    bars: Dict[str, pd.DataFrame], spy: str = "SPY",
    fast: int = 20, slow: int = 50,
    resid_lookback: int = 63, reversal_days: int = 3,
    winsor_pct: float = 0.02
) -> pd.DataFrame:
    rows = []
    rets_df = compute_daily_returns(bars)
    last_prices = {s: df["close"].iloc[-1] for s, df in bars.items() if "close" in df.columns and len(df) > 0}

    # Market proxy
    if spy in rets_df.columns:
        mkt = rets_df[spy].fillna(0).to_frame("mkt")
    else:
        mkt = pd.DataFrame({"mkt": rets_df.mean(axis=1)})

    for s, df in bars.items():
        if len(df) < 60 or "close" not in df.columns: continue
        px = df["close"]; ret = px.pct_change()
        ret21 = (px.iloc[-1] / px.iloc[-22] - 1.0) if len(px) > 22 else np.nan
        ret63 = (px.iloc[-1] / px.iloc[-64] - 1.0) if len(px) > 64 else np.nan
        trend = (moving_average(px, fast).iloc[-1] - moving_average(px, slow).iloc[-1]) / (moving_average(px, slow).iloc[-1] + 1e-9)
        rev = -ret.rolling(reversal_days).sum().iloc[-1]
        vol20 = ann_vol_from_daily(ret.tail(21))
        mdd = max_drawdown(ret.tail(126))
        qual = ((px.iloc[-1] / px.iloc[-127] - 1.0) / (ann_vol_from_daily(ret.tail(63)) + 1e-9)) if len(px) > 127 else np.nan

        # Residual momentum vs market
        df_ret = pd.concat([ret, mkt["mkt"]], axis=1).dropna()
        resid_mom = np.nan
        if len(df_ret) >= resid_lookback:
            X = df_ret["mkt"].tail(resid_lookback).values.reshape(-1,1)
            y = df_ret.iloc[-resid_lookback:,0].values
            model = LinearRegression().fit(X, y)
            resid = y - model.predict(X)
            resid_mom = float(np.nanmean(resid[-21:]) * 21)

        rows.append({
            "symbol": s, "last_price": last_prices.get(s, np.nan),
            "ret21": ret21, "ret63": ret63, "trend": trend, "reversal": rev,
            "vol20": vol20, "maxdd": mdd, "qual126": qual, "resid_mom": resid_mom
        })

    panel = pd.DataFrame(rows).set_index("symbol")
    score_raw = (
        0.30 * winsorize(panel["resid_mom"].fillna(0), winsor_pct) +
        0.22 * winsorize(panel["ret63"].fillna(0), winsor_pct) +
        0.18 * winsorize(panel["ret21"].fillna(0), winsor_pct) +
        0.18 * winsorize(panel["trend"].fillna(0), winsor_pct) +
        0.06 * winsorize(panel["reversal"].fillna(0), winsor_pct) +
        0.12 * winsorize(panel["qual126"].fillna(0), winsor_pct) -
        0.10 * winsorize(panel["vol20"].fillna(0), winsor_pct) -
        0.08 * winsorize(panel["maxdd"].fillna(0), winsor_pct).abs()
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
