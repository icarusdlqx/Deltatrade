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
    bars: Dict[str, pd.DataFrame],
    spy: str = "SPY",
    fast: int = 20,
    slow: int = 50,
    resid_lookback: int = 63,
    reversal_days: int = 3,
    winsor_pct: float = 0.02,
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

        ma_fast = moving_average(px, fast)
        ma_slow = moving_average(px, slow)
        trend = (ma_fast.iloc[-1] - ma_slow.iloc[-1]) / (ma_slow.iloc[-1] + 1e-9)

        ma200 = moving_average(px, 200) if len(px) >= 200 else ma_slow
        ma200_val = ma200.iloc[-1] if len(ma200) else np.nan
        ma200_prior = ma200.iloc[-21] if len(ma200) > 21 else ma200_val
        value_gap = (px.iloc[-1] / ma200_val - 1.0) if ma200_val and ma200_val > 0 else np.nan
        long_slope = ((ma200.iloc[-1] / ma200_prior) - 1.0) if ma200_val and ma200_prior and ma200_prior > 0 else np.nan

        rev = -ret.rolling(reversal_days).sum().iloc[-1]
        vol20 = ann_vol_from_daily(ret.tail(21))
        vol63 = ann_vol_from_daily(ret.tail(63))
        vol126 = ann_vol_from_daily(ret.tail(127))
        drawdown126 = max_drawdown(ret.tail(126))
        drawdown252 = max_drawdown(ret.tail(252)) if len(ret) >= 252 else np.nan

        qual = (ret252 / (vol126 + 1e-9)) if np.isfinite(ret252) and np.isfinite(vol126) else np.nan

        df_pair = pd.concat([ret.rename("asset"), mkt["mkt"]], axis=1).dropna()
        resid_mom = np.nan
        beta = np.nan
        if len(df_pair) >= resid_lookback:
            X = df_pair["mkt"].tail(resid_lookback).values.reshape(-1, 1)
            y = df_pair["asset"].tail(resid_lookback).values
            model = LinearRegression().fit(X, y)
            resid = y - model.predict(X)
            resid_mom = float(np.nanmean(resid[-21:]) * 21)

        if len(df_pair) >= 63:
            tail = df_pair.tail(min(len(df_pair), 126))
            var_mkt = float(np.var(tail["mkt"]))
            if var_mkt > 1e-9:
                beta = float(np.cov(tail["mkt"], tail["asset"])[0, 1] / var_mkt)

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
                "vol126": vol126,
                "maxdd": drawdown126,
                "maxdd252": drawdown252,
                "qual126": qual,
                "resid_mom": resid_mom,
                "value_gap": value_gap,
                "trend_slope_long": long_slope,
                "beta": beta,
            }
        )

    panel = pd.DataFrame(rows).set_index("symbol")
    if panel.empty:
        panel["score_raw"] = pd.Series(dtype=float)
        panel["score_z"] = pd.Series(dtype=float)
        panel["long_term_thesis"] = pd.Series(dtype=object)
        return panel

    panel["quality_ratio"] = panel["qual126"]
    panel["defensiveness"] = -panel["maxdd252"].fillna(panel["maxdd"].fillna(0))
    panel["stability"] = -panel["vol126"].fillna(panel["vol63"].fillna(0))
    panel["growth_long"] = panel["ret252"].fillna(panel["ret126"].fillna(0))
    panel["value"] = -panel["value_gap"].fillna(0)
    panel["long_trend"] = panel["trend_slope_long"].fillna(panel["trend"].fillna(0))

    score_raw = (
        0.28 * winsorize(panel["growth_long"].fillna(0), winsor_pct)
        + 0.20 * winsorize(panel["quality_ratio"].fillna(0), winsor_pct)
        + 0.18 * winsorize(panel["value"].fillna(0), winsor_pct)
        + 0.14 * winsorize(panel["long_trend"].fillna(0), winsor_pct)
        + 0.12 * winsorize(panel["defensiveness"].fillna(0), winsor_pct)
        + 0.08 * winsorize(panel["stability"].fillna(0), winsor_pct)
    )
    panel["score_raw"] = score_raw
    panel["score_long_term"] = score_raw
    panel["score_z"] = zscore(score_raw.fillna(score_raw.median()))

    def _describe_row(row: pd.Series) -> str:
        parts: List[str] = []
        if np.isfinite(row.get("ret252", np.nan)):
            vol = row.get("vol126")
            if np.isfinite(vol) and vol > 0:
                parts.append(
                    f"12-month return {row['ret252'] * 100:.1f}% with {vol * 100:.1f}% annual volatility"
                )
            else:
                parts.append(f"12-month return {row['ret252'] * 100:.1f}%")

        value_gap = row.get("value_gap")
        if np.isfinite(value_gap):
            if value_gap < -0.02:
                parts.append(f"trading {abs(value_gap) * 100:.1f}% below its 200-day average")
            elif value_gap > 0.02:
                parts.append(f"trading {value_gap * 100:.1f}% above its 200-day average")

        beta_val = row.get("beta")
        if np.isfinite(beta_val):
            parts.append(f"market beta ≈ {beta_val:.2f}")

        dd = row.get("maxdd252")
        if not np.isfinite(dd):
            dd = row.get("maxdd")
        if np.isfinite(dd):
            parts.append(f"max drawdown over year {dd * 100:.1f}%")

        if not parts:
            return "No robust long-term signals available yet."
        return ". ".join(parts) + "."

    panel["long_term_thesis"] = panel.apply(_describe_row, axis=1)
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
