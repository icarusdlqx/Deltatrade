from __future__ import annotations
import os, time, logging
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
from .config import (
    COST_BPS_PER_1PCT_TURNOVER,
    MIN_NET_BPS_TO_TRADE as CONFIG_MIN_NET_BPS,
    MIN_ORDER_NOTIONAL as CONFIG_MIN_ORDER_NOTIONAL,
    VOL_TARGET_ANNUAL,
)
from .settings_bridge import get_cfg
from .utils import write_jsonl, read_json, write_json
from .features import compute_panel
from .news import fetch_news_map
from .agents import score_events_for_symbols, risk_officer_check
from .optimizer import solve_weights, scale_to_target_vol
from .execution import build_order_plans, place_orders_with_limits, estimate_cost_bps
from .universe import build_universe, load_top50_etfs

logger = logging.getLogger(__name__)
log = logger

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
    mode = str(getattr(cfg, "UNIVERSE_MODE", "") or "").lower()
    if mode == "etfs_only":
        symbols = load_top50_etfs()
    else:
        symbols = build_universe()
    if getattr(cfg, "UNIVERSE_MAX", None) and len(symbols) > cfg.UNIVERSE_MAX:
        return symbols[:cfg.UNIVERSE_MAX]
    return symbols

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
    state = read_json(C.STATE_PATH, default={"stops": {}})

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
    factor_component = sw.get("xsec", 0) * factor_alpha_bps
    event_component = sw.get("event", 0) * pd.Series(event_alpha_bps, dtype=float)
    alpha_bps = factor_component.add(event_component, fill_value=0.0).fillna(0)
    alpha = alpha_bps / 10000.0  # daily return proxy

    # Covariance
    rets = {s: df["close"].pct_change().dropna() for s, df in bars.items()}
    df_ret = pd.DataFrame(rets).dropna(axis=1, how="any").tail(126)
    common_syms = [s for s in panel.index if s in df_ret.columns]
    df_ret = df_ret[common_syms]
    Sigma = df_ret.cov().values

    # Candidate cut
    candidates = common_syms[:max(int(cfg.TARGET_POSITIONS)*2, int(cfg.TARGET_POSITIONS))]
    factor_series = factor_component.reindex(candidates).fillna(0.0)
    event_series = event_component.reindex(candidates).fillna(0.0)
    alpha_series = alpha_bps.reindex(candidates).fillna(0.0)
    alpha_breakdown = {
        "per_symbol_bps": {
            s: {
                "factor": float(factor_series.get(s, 0.0)),
                "event": float(event_series.get(s, 0.0)),
                "total": float(alpha_series.get(s, 0.0)),
            }
            for s in candidates
        },
        "sleeve_totals_bps": {
            "factor": float(factor_series.sum()),
            "event": float(event_series.sum()),
            "total": float(alpha_series.sum()),
        },
    }
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

    target_weights = {s: float(w) for s, w in zip(candidates, w_scaled)}

    # Targets
    account = trade_client.get_account()
    equity = float(account.equity)
    investable = equity * (1 - float(cfg.CASH_BUFFER))
    targets = {s: float(w * investable) for s, w in zip(candidates, w_scaled)}

    state["equity"] = equity
    state["cash"] = float(getattr(account, "cash", 0.0) or 0.0)

    # Rebalance bands (skip for initial allocation from cash)
    cur_mv_all = {p.symbol: float(p.market_value) for p in trade_client.get_all_positions()}
    targets_banded = {}
    for s, tgt in targets.items():
        cur = cur_mv_all.get(s, 0.0)
        if cur == 0:
            # Always include new positions, especially during onboarding
            if abs(tgt) > 0:
                targets_banded[s] = tgt
        else:
            drift = abs(tgt - cur) / max(1.0, abs(tgt))
            if drift >= float(cfg.REBALANCE_BAND):
                targets_banded[s] = tgt

    # Cost-aware gate
    prices = {s: float(bars[s]["close"].iloc[-1]) for s in candidates if s in bars}
    adv = _estimate_adv(bars)
    expected_alpha_bps = float(np.dot(alpha.reindex(candidates).fillna(0).values, w_scaled)) * 10000.0
    cost_breakdown = {}
    rebalance_deltas: Dict[str, Dict[str, float]] = {}
    investable_denom = max(1.0, investable)
    turnover_total = 0.0
    for s, tgt in targets_banded.items():
        cur = cur_mv_all.get(s, 0.0)
        delta = tgt - cur
        rebalance_deltas[s] = {
            "current": cur,
            "target": tgt,
            "delta": delta,
        }
        per_order_bps = estimate_cost_bps(
            delta,
            prices.get(s, 0.0),
            adv.get(s, 1.0),
            float(cfg.COST_SPREAD_BPS),
            float(cfg.COST_IMPACT_KAPPA),
            float(cfg.COST_IMPACT_PSI),
        )
        turnover_share = abs(delta) / investable_denom
        turnover_total += turnover_share
        cost_breakdown[s] = {
            "per_order_bps": per_order_bps,
            "turnover_share": turnover_share,
            "notional_delta": delta,
            "current_notional": cur,
            "target_notional": tgt,
        }
    min_order_notional = float(getattr(cfg, "MIN_ORDER_NOTIONAL", CONFIG_MIN_ORDER_NOTIONAL))
    min_net_bps_to_trade = float(getattr(cfg, "MIN_NET_BPS_TO_TRADE", CONFIG_MIN_NET_BPS))
    turnover_cap = float(getattr(cfg, "TURNOVER_CAP", 0.0) or 0.0)

    # Build order plans (always do this if we have targets, especially for onboarding)
    planned_orders = []
    if targets_banded:
        planned_orders = build_order_plans(
            targets_banded,
            cur_mv_all,
            prices,
            adv,
            min_order_notional,
            int(cfg.MAX_SLICES),
            float(cfg.COST_SPREAD_BPS),
            float(cfg.COST_IMPACT_KAPPA),
            float(cfg.COST_IMPACT_PSI),
        )

        # Log onboarding scenario
        if len(planned_orders) > 0 and (state.get("cash", 0.0) / max(state.get("equity", 1.0), 1e-9) >= 0.90):
            logger.info(
                "onboarding_detected",
                extra={
                    "cash_ratio": float(state.get("cash", 0.0) / max(state.get("equity", 1.0), 1e-9)),
                    "planned_orders": len(planned_orders),
                    "targets_count": len(targets_banded),
                    "reason": "initial_allocation_from_cash",
                },
            )

    try:
        equity_snap = float(account.portfolio_value)
        cash_snap = float(account.cash)
    except Exception:
        equity_snap = float(state.get("equity", 0.0))
        cash_snap = float(state.get("cash", 0.0))

    est_turnover = 0.0
    for od in planned_orders:
        dollars = float(od.get("notional") or 0.0)
        if (dollars <= 0.0) and ("qty" in od and "last_price" in od):
            dollars = float(od.get("qty", 0.0)) * float(od.get("last_price", 0.0))
        est_turnover += abs(dollars)
    est_turnover = est_turnover / max(equity_snap, 1e-9)
    turnover_pct = est_turnover * 100.0
    cost_bps = 0.0 if len(planned_orders) == 0 else (turnover_pct * float(COST_BPS_PER_1PCT_TURNOVER))
    expected_alpha_bps = locals().get("expected_alpha_bps", 0.0)
    net_bps = expected_alpha_bps - cost_bps
    est_cost_bps = cost_bps

    cash_frac = cash_snap / max(equity_snap, 1e-9)
    force_onboard = (cash_frac >= 0.90) and (len(planned_orders) > 0)
    passes_net_bps = True
    if len(planned_orders) > 0:
        passes_net_bps = (net_bps >= min_net_bps_to_trade)

    proceed_gate = (len(planned_orders) > 0) and (passes_net_bps or force_onboard)

    reasons: List[str] = []
    if len(planned_orders) == 0:
        reasons.append("no_orders")
    else:
        if not passes_net_bps:
            reasons.append("net_below_min")
        if force_onboard:
            reasons.append("onboarding")
        turnover_fraction = est_turnover if len(planned_orders) > 0 else turnover_total
        if turnover_cap > 0 and turnover_fraction > turnover_cap:
            proceed_gate = False
            reasons.append("turnover_cap")
        if bool(cfg.ENABLE_COST_GATE) and not force_onboard and expected_alpha_bps <= cost_bps:
            if "net_below_min" not in reasons:
                reasons.append("alpha_not_covering_cost")
            proceed_gate = False

    if force_onboard:
        log.info("onboarding", extra={"cash_frac": cash_frac, "reason": "initial allocation from cash"})

    cash_ratio = cash_frac
    onboarding_gate = force_onboard

    # Risk officer
    proposed_w = {s: (targets_banded.get(s, 0.0) / max(1.0, investable)) for s in candidates}
    roc = risk_officer_check(
        proposed_w,
        {s: rev_sec.get(sector_ids[i], "UNKNOWN") for i, s in enumerate(candidates)},
        float(cfg.SECTOR_MAX),
        float(cfg.NAME_MAX),
    )
    risk_officer_verdict = {
        "approved": str(roc.get("approved", "false")).lower() == "true",
        "message": roc.get("message"),
        "payload": roc,
    }
    risk_officer_approved = bool(risk_officer_verdict["approved"])
    if proceed_gate and not risk_officer_approved:
        reasons.append("risk_officer_blocked")

    proceed_final = proceed_gate and risk_officer_approved

    mode = "sim" if getattr(trade_client, "is_simulated", False) else (
        "live" if os.environ.get("ALPACA_PAPER", "true").lower() == "false" else "paper"
    )
    model_name = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or C.OPENAI_MODEL
    effort = os.getenv("REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or C.OPENAI_REASONING_EFFORT
    regime = locals().get("regime_summary", {})
    target_gross = sum(abs(w) for w in target_weights.values())
    exposure = {"target_gross": float(target_gross), "vol_target": float(VOL_TARGET_ANNUAL)}

    gate_reason = None
    for r in reasons:
        if r != "onboarding":
            gate_reason = r
            break

    try:
        log.info(
            "gate_check",
            extra={
                "mode": mode,
                "model": model_name,
                "effort": effort,
                "equity": float(equity_snap),
                "cash": float(cash_snap),
                "cash_frac": float(cash_frac),
                "order_count": len(planned_orders),
                "turnover_pct": float(turnover_pct),
                "cost_bps": float(cost_bps),
                "expected_alpha_bps": float(expected_alpha_bps),
                "net_bps": float(net_bps),
                "min_notional": float(min_order_notional),
                "min_net_bps": float(min_net_bps_to_trade),
                "proceed": bool(proceed_gate),
                "reasons": reasons,
                "exposure": exposure,
                "regime": regime,
            },
        )
    except Exception:
        pass

    orders = []
    if proceed_final and planned_orders:
        if bool(cfg.DRY_RUN):
            orders = ["DRY_RUN_NO_ORDERS"]
        else:
            order_ids = place_orders_with_limits(
                trade_client,
                planned_orders,
                prices,
                int(cfg.LIMIT_SLIP_BP),
                fill_timeout_sec=int(cfg.FILL_TIMEOUT_SEC),
            )
            orders = order_ids

    # Stops/time exits (stored only)
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
    state["stops"] = stops
    state["equity"] = equity
    state["cash"] = float(getattr(account, "cash", 0.0) or cash_snap)
    write_json(C.STATE_PATH, state)

    net_edge_bps = net_bps
    top_changes = sorted(rebalance_deltas.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True)[:3]
    changes_summary = ", ".join(
        f"{sym}: {vals['delta']:+.0f}" for sym, vals in top_changes
    ) if top_changes else "no meaningful target adjustments"
    logger.info(
        "Run summary (gate=%s, final=%s)\nExpected alpha: %.2f bps | Estimated cost: %.2f bps (net %.2f bps)\nTop target changes: %s\nRisk officer verdict: %s",
        proceed_gate,
        proceed_final,
        expected_alpha_bps,
        est_cost_bps,
        net_edge_bps,
        changes_summary,
        risk_officer_verdict.get("message") or roc.get("approved"),
    )

    ep = {
        "as_of": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        "investable": investable,
        "cash_ratio": cash_ratio,
        "expected_alpha_bps": expected_alpha_bps,
        "est_cost_bps": est_cost_bps,
        "net_edge_bps": net_edge_bps,
        "est_cost_breakdown": cost_breakdown,
        "alpha_breakdown": alpha_breakdown,
        "rebalance_deltas": rebalance_deltas,
        "rebalance_summary": changes_summary,
        "proceed": proceed_final,
        "risk_officer": roc,
        "risk_officer_verdict": risk_officer_verdict,
        "top_symbols": candidates[: int(cfg.TARGET_POSITIONS)],
        "targets": targets_banded,
        "orders_submitted": orders,
        "gate_reason": gate_reason,
        "onboarding_gate": onboarding_gate,
        "planned_orders_count": len(planned_orders),
        "simulated": bool(getattr(trade_client, "is_simulated", False)),
        "gate": {
            "mode": mode,
            "model": model_name,
            "effort": effort,
            "equity": float(equity_snap),
            "cash": float(cash_snap),
            "cash_frac": float(cash_frac),
            "order_count": len(planned_orders),
            "turnover_pct": float(turnover_pct),
            "cost_bps": float(cost_bps),
            "expected_alpha_bps": float(expected_alpha_bps),
            "net_bps": float(net_bps),
            "min_notional": float(min_order_notional),
            "min_net_bps": float(min_net_bps_to_trade),
            "proceed": bool(proceed_gate),
            "passes_net_bps": bool(passes_net_bps),
            "force_onboard": bool(force_onboard),
            "risk_officer_approved": bool(risk_officer_approved),
            "reasons": list(reasons),
            "reason_primary": gate_reason,
            "exposure": exposure,
            "regime": regime,
        }
    }
    write_jsonl(C.EPISODES_PATH, ep)
    return ep
