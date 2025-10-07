from __future__ import annotations
import os
import datetime as dt
from typing import List
import pandas as pd


def _alpaca_daily(symbols: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY"),
            os.getenv("ALPACA_SECRET_KEY"),
        )
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start, dt.time.min),
            end=dt.datetime.combine(end, dt.time.max),
            adjustment="split",
            feed=os.getenv("ALPACA_FEED", "iex"),
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame()
        bars = bars.rename(
            columns={
                "close": "Close",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
            }
        )
        return bars.reset_index().set_index(["symbol", "timestamp"]).sort_index()
    except Exception:
        return pd.DataFrame()


def _yf_daily(symbols: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        import yfinance as yf

        data = yf.download(
            tickers=" ".join(symbols),
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            interval="1d",
        )
        if isinstance(data.columns, pd.MultiIndex):
            out = []
            for sym in symbols:
                if sym not in data:
                    continue
                df = data[sym][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                df = df.assign(symbol=sym).reset_index().rename(columns={"Date": "timestamp"})
                out.append(df)
            if not out:
                return pd.DataFrame()
            bars = pd.concat(out, ignore_index=True)
        else:
            bars = (
                data[["Open", "High", "Low", "Close", "Volume"]]
                .dropna(how="all")
                .reset_index()
                .rename(columns={"Date": "timestamp"})
            )
            bars["symbol"] = symbols[0]
        return bars.set_index(["symbol", "timestamp"]).sort_index()
    except Exception:
        return pd.DataFrame()


def get_daily_bars(symbols: List[str], lookback_days: int = 252) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days * 1.4))
    bars = _alpaca_daily(symbols, start, end)
    if not symbols:
        return pd.DataFrame()
    have = set(bars.index.get_level_values(0)) if not bars.empty else set()
    missing = [s for s in symbols if s not in have]
    if missing:
        yfb = _yf_daily(missing, start, end)
        if not yfb.empty:
            bars = pd.concat([bars, yfb]) if not bars.empty else yfb
    return bars
