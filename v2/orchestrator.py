from __future__ import annotations
import os, time
from datetime import datetime, timedelta, timezone
import pytz
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from . import config as C
from .settings_bridge import get_cfg
from .utils import write_jsonl, read_json, write_json
from .features import compute_panel
from .news import fetch_news_map
from .agents import score_events_for_symbols, risk_officer_check
from .optimizer import solve_weights, scale_to_target_vol
from .execution import build_order_plans, place_orders_with_limits, estimate_cost_bps

def _alpaca_clients():
    api_key = os.environ.get("ALPACA_API_KEY")
    api_sec = os.environ.get("ALPACA_SECRET_KEY")
    force_sim = os.environ.get("SIM_MODE", "false").lower() in ("true", "1", "yes", "y")
    if force_sim or not api_key or not api_sec:
        from .simulated_clients import SimStockHistoricalDataClient, SimTradingClient

        data_client = SimStockHistoricalDataClient()
        trade_client = SimTradingClient(data_client)
        setattr(data_client, "is_simulated", True)
        setattr(trade_client, "is_simulated", True)
        return data_client, trade_client, "SIMULATED", "SIMULATED"

    paper = os.environ.get("ALPACA_PAPER","true").lower() in ("true","1","yes","y")
    return StockHistoricalDataClient(api_key, api_sec), TradingClient(api_key, api_sec, paper=paper), api_key, api_sec

def _universe(cfg) -> List[str]:
    if cfg.UNIVERSE_MODE == "etfs_only":
        return ["SPY","QQQ","IWM","DIA","XLK","XLF","XLY","XLP","XLE","XLV","XLI","XLB","XLU","XLRE","SMH","SOXX","HYG","TLT","IEF"]
    # If you drop a data/sp500.csv with header Symbol, use it
    try:
        df = pd.read_csv("data/sp500.csv")
        syms = df["Symbol"].astype(str).str.upper().tolist()
        syms = list(dict.fromkeys(syms + ["SPY","QQQ","IWM","DIA","SMH","XLK","XLF","XLY"]))
    except Exception:
        syms = ["SPY","QQQ","IWM","DIA","SMH","XLK","XLF","XLY"]
    if cfg.UNIVERSE_MAX and len(syms) > cfg.UNIVERSE_MAX:
        return syms[:cfg.UNIVERSE_MAX]
    return syms

def _fetch_bars(client: StockHistoricalDataClient, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
    try:
        et_tz = pytz.timezone("US/Eastern")
        end = datetime.now(et_tz); start = end - timedelta(days=days + 10)
        # Convert to UTC for API calls
        end = end.astimezone(timezone.utc)
        start = start.astimezone(timezone.utc)
        out: Dict[str, pd.DataFrame] = {}
        B = 150
        for i in range(0, len(symbols), B):
            chunk = symbols[i:i+B]
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start, end=end)
            bars = client.get_stock_bars(req)
            df = getattr(bars, "df", None)
            if df is None or df.empty: 
                continue
            for sym, g in df.groupby(level=0):
                dfg = g.reset_index(level=0, drop=True).sort_index()
                dfg = dfg.rename(columns={"trade_count":"trades"})
                out[str(sym)] = dfg[["open","high","low","close","volume"]]
            time.sleep(0.2)
        return out
    except Exception as e:
        # Handle subscription errors (e.g., "subscription does not permit querying recent SIP data")
        error_msg = str(e).lower()
        if "subscription" in error_msg or "permit" in error_msg or "forbidden" in error_msg:
            print(f"Alpaca subscription error: {e}")
            print("Falling back to simulated data client...")
            # Use simulated client as fallback
            if hasattr(client, 'is_simulated') and client.is_simulated:
                return {}  # Already simulated, return empty
            from .simulated_clients import SimStockHistoricalDataClient
            sim_client = SimStockHistoricalDataClient()
            return _fetch_bars(sim_client, symbols, days)
        else:
            # Re-raise other errors
            raise e

def _sector_map_from_csv() -> Dict[str,str]:
    # Optional: load sectors if data/sp500_sectors.csv present with columns Symbol,GICS Sector
    try:
        df = pd.read_csv("data/sp500_sectors.csv")
        return {str(r["Symbol"]).upper(): str(r["GICS Sector"]) for _, r in df.iterrows()}
    except Exception:
        return {}

def _current_positions(tc: TradingClient) -> Tuple[Dict[str,float], float, Dict[str,float]]:
    pos = tc.get_all_positions()
    current_mv, last_prices = {}, {}
    for p in pos:
        current_mv[p.symbol] = float(p.market_value)
        last_prices[p.symbol] = float(p.current_price)
    acct = tc.get_account()
    return current_mv, float(acct.equity), last_prices or {}

def _estimate_adv(bars: Dict[str,pd.DataFrame]) -> Dict[str,float]:
    adv = {}
    for s, df in bars.items():
        if "volume" in df.columns and "close" in df.columns and len(df) > 0:
            v = float(df["volume"].tail(20).mean())
            px = float(df["close"].iloc[-1])
            adv[s] = v * px if v and px else 0.0
    return adv

def run_once() -> dict:
    data_client, trade_client, api_key, api_sec = _alpaca_clients()
    cfg = get_cfg()

    symbols = _universe(cfg)
    bars = _fetch_bars(data_client, symbols, cfg.DATA_LOOKBACK_DAYS)
    symbols = [s for s in symbols if s in bars and len(bars[s]) >= cfg.MIN_BARS]
    if "SPY" not in symbols and "SPY" in bars: symbols.append("SPY")

    panel = compute_panel(bars, spy="SPY", fast=cfg.TREND_FAST, slow=cfg.TREND_SLOW,
                          resid_lookback=cfg.RESID_MOM_LOOKBACK, reversal_days=cfg.REVERSAL_DAYS,
                          winsor_pct=cfg.WINSOR_PCT).dropna(subset=["last_price"]).sort_values("score_z", ascending=False)

    # Event-driven alpha
    if cfg.ENABLE_EVENT_SCORE and api_key not in (None, "SIMULATED"):
        top_syms = panel.index.tolist()[:cfg.EVENT_TOP_K]
        news_map = fetch_news_map(top_syms, cfg.NEWS_LOOKBACK_DAYS, api_key, api_sec)
        event_alpha_bps = score_events_for_symbols(news_map, C.OPENAI_MODEL, C.OPENAI_REASONING_EFFORT,
                                                   C.EVENT_BPS_PER_DAY, max_abs_bps=20)
        event_alpha_bps = {k: float(v) * float(cfg.EVENT_ALPHA_MULT) for k, v in event_alpha_bps.items()}
    else:
        event_alpha_bps = {}

    # Factor + event blend
    factor_alpha_bps = 8.0 * panel["score_z"].fillna(0)
    sw = cfg.SLEEVE_WEIGHTS
    alpha_bps = (sw.get("xsec",0)*factor_alpha_bps).add(sw.get("event",0) * pd.Series(event_alpha_bps))
    alpha_bps = alpha_bps.fillna(0)
    alpha = alpha_bps / 10000.0  # daily return proxy

    # Covariance
    rets = {s: df["close"].pct_change().dropna() for s, df in bars.items()}
    df_ret = pd.DataFrame(rets).dropna(axis=1, how="any").tail(126)
    common_syms = [s for s in panel.index if s in df_ret.columns]
    df_ret = df_ret[common_syms]
    Sigma = df_ret.cov().values

    # Candidate cut
    candidates = common_syms[:max(int(cfg.TARGET_POSITIONS)*2, int(cfg.TARGET_POSITIONS))]
    alpha_vec = alpha.reindex(candidates).fillna(0).values

    cur_mv_map, equity_prev, last_prices_live = _current_positions(trade_client)
    invested_prev = sum(abs(v) for v in cur_mv_map.values()) or 1.0
    w_prev = np.array([cur_mv_map.get(s, 0.0) for s in candidates], dtype=float) / invested_prev

    # Sector caps (optional via CSV)
    sec_map = _sector_map_from_csv()
    sec_vocab, sector_ids, sid = {}, [], 0
    for s in candidates:
        sec = sec_map.get(s, "UNKNOWN")
        if sec not in sec_vocab:
            sec_vocab[sec] = sid; sid += 1
        sector_ids.append(sec_vocab[sec])
    rev_sec = {v:k for k,v in sec_vocab.items()}

    # Optimize
    Sig_sub = Sigma[:len(candidates), :len(candidates)]
    w_opt = solve_weights(alpha_vec, Sig_sub, w_prev, float(cfg.NAME_MAX), sector_ids, float(cfg.SECTOR_MAX),
                          float(cfg.LAMBDA_RISK), float(cfg.TURNOVER_PENALTY))

    # Vol targeting
    if cfg.ENABLE_VOL_TARGETING:
        w_scaled = scale_to_target_vol(w_opt, Sig_sub, float(cfg.TARGET_PORTFOLIO_VOL)/np.sqrt(252.0))
    else:
        w_scaled = w_opt

    # Targets
    acct = trade_client.get_account()
    equity = float(acct.equity)
    investable = equity * (1 - float(cfg.CASH_BUFFER))
    targets = {s: float(w * investable) for s, w in zip(candidates, w_scaled)}

    # Rebalance bands
    cur_mv_all = {p.symbol: float(p.market_value) for p in trade_client.get_all_positions()}
    targets_banded = {}
    for s, tgt in targets.items():
        cur = cur_mv_all.get(s, 0.0)
        if cur == 0:
            targets_banded[s] = tgt
        else:
            drift = abs(tgt - cur) / max(1.0, abs(tgt))
            if drift >= float(cfg.REBALANCE_BAND):
                targets_banded[s] = tgt

    # Cost-aware gate
    prices = {s: float(bars[s]["close"].iloc[-1]) for s in candidates if s in bars}
    adv = _estimate_adv(bars)
    expected_alpha_bps = float(np.dot(alpha.reindex(candidates).fillna(0).values, w_scaled)) * 10000.0
    est_cost_bps = 0.0
    for s, tgt in targets_banded.items():
        cur = cur_mv_all.get(s, 0.0)
        delta = tgt - cur
        est_cost_bps += estimate_cost_bps(delta, prices.get(s, 0.0), adv.get(s, 1.0),
                                          float(cfg.COST_SPREAD_BPS), float(cfg.COST_IMPACT_KAPPA), float(cfg.COST_IMPACT_PSI))
    proceed = True if not cfg.ENABLE_COST_GATE else (expected_alpha_bps > est_cost_bps)

    # Risk officer
    proposed_w = {s: (targets_banded.get(s, 0.0) / max(1.0, investable)) for s in candidates}
    roc = risk_officer_check(proposed_w, {s: rev_sec.get(sector_ids[i], "UNKNOWN") for i, s in enumerate(candidates)},
                             float(cfg.SECTOR_MAX), float(cfg.NAME_MAX))

    orders = []
    if proceed and roc.get("approved") == "true" and targets_banded:
        if bool(cfg.DRY_RUN):
            orders = ["DRY_RUN_NO_ORDERS"]
        else:
            plans = build_order_plans(targets_banded, cur_mv_all, prices, adv,
                                      float(cfg.MIN_ORDER_NOTIONAL), int(cfg.MAX_SLICES),
                                      float(cfg.COST_SPREAD_BPS), float(cfg.COST_IMPACT_KAPPA), float(cfg.COST_IMPACT_PSI))
            order_ids = place_orders_with_limits(trade_client, plans, prices, int(cfg.LIMIT_SLIP_BP), fill_timeout_sec=int(cfg.FILL_TIMEOUT_SEC))
            orders = order_ids

    # Stops/time exits (stored only)
    state = read_json(C.STATE_PATH, default={"stops":{}})
    stops = state.get("stops", {})
    for s, tgt in targets.items():
        px = prices.get(s, 0.0)
        df = bars.get(s)
        if df is None or px <= 0: continue
        hi, lo, cl = df["high"], df["low"], df["close"]
        prev = cl.shift(1)
        tr = pd.concat([(hi-lo), (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.0
        if atr > 0:
            stop = px - float(cfg.ATR_STOP_MULT) * atr
            take = px + float(cfg.TAKE_PROFIT_ATR) * atr
            stops[s] = {"stop": stop, "take": take, "placed_at": str(datetime.now(pytz.timezone("US/Eastern"))), "ttl_days": int(cfg.TIME_STOP_DAYS)}
    write_json(C.STATE_PATH, {"stops": stops})

    ep = {
        "as_of": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        "investable": investable,
        "expected_alpha_bps": expected_alpha_bps,
        "est_cost_bps": est_cost_bps,
        "proceed": proceed,
        "risk_officer": roc,
        "top_symbols": candidates[: int(cfg.TARGET_POSITIONS)],
        "targets": targets_banded,
        "orders_submitted": orders,
        "simulated": bool(getattr(trade_client, "is_simulated", False))
    }
    write_jsonl(C.EPISODES_PATH, ep)
    return ep
