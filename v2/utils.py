from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_dir(path: str | Path) -> None:
    p = Path(path)
    (p if p.is_dir() else p.parent).mkdir(parents=True, exist_ok=True)

def write_jsonl(path: str | Path, record: dict) -> None:
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

def read_json(path: str | Path, default: dict | None = None) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default or {}

def write_json(path: str | Path, obj: dict) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def ann_vol_from_daily(ret: pd.Series) -> float:
    if ret is None or ret.dropna().empty: return 0.0
    return float(ret.std(ddof=0) * np.sqrt(252))

def max_drawdown(returns: pd.Series) -> float:
    x = returns.dropna()
    if x.empty: return 0.0
    cum = (1 + x).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd and sd > 0 else s*0

def winsorize(s: pd.Series, p: float) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)
