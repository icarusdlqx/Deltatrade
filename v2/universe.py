from __future__ import annotations
import io, time, pathlib
from typing import List, Set
import pandas as pd, requests

DATA_DIR = pathlib.Path("data")
SP500_PATH = DATA_DIR / "sp500.csv"          # cache (auto-refreshed weekly)
SP500_STATIC = DATA_DIR / "sp500_static.csv" # optional fallback
ETFS_PATH = DATA_DIR / "etfs_top50.csv"

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

def _read_syms(path: pathlib.Path) -> List[str]:
    if not path.exists(): return []
    df = pd.read_csv(path)
    col = [c for c in df.columns if c.lower().strip() in ("symbol","ticker")]
    if not col: return []
    return (df[col[0]].astype(str).str.upper().str.strip()).dropna().unique().tolist()

def _download_sp500() -> List[str]:
    try:
        r = requests.get(SP500_URL, timeout=15); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        syms = (df["Symbol"].astype(str).str.upper().str.strip()).dropna().unique().tolist()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Symbol": syms}).to_csv(SP500_PATH, index=False)
        return syms
    except Exception:
        return _read_syms(SP500_STATIC)

def load_sp500(refresh_days: int = 7) -> List[str]:
    try:
        if SP500_PATH.exists():
            age = (time.time() - SP500_PATH.stat().st_mtime)/86400.0
            if age <= refresh_days:
                return _read_syms(SP500_PATH)
        return _download_sp500()
    except Exception:
        return _read_syms(SP500_PATH) or _read_syms(SP500_STATIC)

def load_top50_etfs() -> List[str]:
    syms = _read_syms(ETFS_PATH)
    if syms: return syms
    # Hard fallback (should be replaced by CSV below)
    return ["SPY","IVV","VOO","QQQ","IWM","EEM","EFA","HYG","LQD","TLT","IEF","BND","BIL","SHY","GLD","SLV","GDX",
            "XLF","XLK","XLE","XLV","XLY","XLP","XLI","XLB","XLRE","XLU","XLC","DIA","SMH","SOXX","XOP","XME",
            "XHB","XRT","KRE","KBE","OIH","KWEB","FXI","EWT","EWH","EWJ","EWG","EWU","EWC","EWY","INDA","IEMG","VEA"]

def build_universe() -> List[str]:
    sp = set(load_sp500()); etf = set(load_top50_etfs())
    uni: Set[str] = (sp | etf) - {None,"","N/A"}
    return sorted(uni)

if __name__ == "__main__":
    u = build_universe()
    print("Universe size:", len(u)); print(u[:30])
