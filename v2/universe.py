from __future__ import annotations
import os, io, time, pathlib
from typing import List, Set
import pandas as pd
import requests

DATA_DIR = pathlib.Path("data")
SP500_PATH = DATA_DIR / "sp500.csv"            # optional local cache
SP500_STATIC = DATA_DIR / "sp500_static.csv"   # optional fallback
ETFS_PATH = DATA_DIR / "etfs_top50.csv"        # required; we will add in step B

# A reasonably-stable public S&P500 constituents CSV (Symbol column).
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

def _read_csv_symbols(path: pathlib.Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = [c for c in df.columns if c.lower().strip() in ("symbol", "ticker")]
    if not col:
        return []
    return (df[col[0]].astype(str).str.upper().str.strip()).dropna().unique().tolist()

def _download_sp500() -> List[str]:
    try:
        r = requests.get(SP500_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        syms = (df["Symbol"].astype(str).str.upper().str.strip()).dropna().unique().tolist()
        # cache to data/sp500.csv
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Symbol": syms}).to_csv(SP500_PATH, index=False)
        return syms
    except Exception:
        # fallback to local static if present
        return _read_csv_symbols(SP500_STATIC)

def load_sp500_symbols(refresh_days: int = 7) -> List[str]:
    """
    Use cached data/sp500.csv if newer than refresh_days; otherwise fetch from SP500_URL.
    Falls back to data/sp500_static.csv when offline.
    """
    try:
        if SP500_PATH.exists():
            age_days = (time.time() - SP500_PATH.stat().st_mtime) / 86400.0
            if age_days <= refresh_days:
                return _read_csv_symbols(SP500_PATH)
        return _download_sp500()
    except Exception:
        return _read_csv_symbols(SP500_PATH) or _read_csv_symbols(SP500_STATIC)

def load_top50_etfs() -> List[str]:
    syms = _read_csv_symbols(ETFS_PATH)
    # hard fallback list (should not be hit because we add the CSV in step B)
    if syms:
        return syms
    return [
        "SPY","IVV","VOO","QQQ","IWM","EEM","EFA","HYG","LQD","TLT",
        "IEF","BND","BIL","SHY","GLD","SLV","GDX","XLF","XLK","XLE",
        "XLV","XLY","XLP","XLI","XLB","XLRE","XLU","XLC","DIA","SMH",
        "SOXX","XOP","XME","XHB","XRT","KRE","KBE","OIH","KWEB","FXI",
        "EWT","EWH","EWJ","EWG","EWU","EWC","EWY","INDA","IEMG","VEA"
    ]

def build_universe() -> List[str]:
    sp = set(load_sp500_symbols())
    etf = set(load_top50_etfs())
    uni: Set[str] = (sp | etf) - {None, "", "N/A"}
    return sorted(uni)

if __name__ == "__main__":
    print("Universe size:", len(build_universe()))
    print(build_universe()[:20])
