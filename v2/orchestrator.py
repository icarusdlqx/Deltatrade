from __future__ import annotations
import os, time, logging
from datetime import datetime, timedelta, timezone
import pytz
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from . import config as C
from .config import (
    COST_BPS_PER_1PCT_TURNOVER,
    GROSS_EXPOSURE_FLOOR,
    MAX_WEIGHT_PER_NAME,
    MIN_NET_BPS_TO_TRADE as CONFIG_MIN_NET_BPS,
    MIN_ORDER_NOTIONAL as CONFIG_MIN_ORDER_NOTIONAL,
    TARGET_POSITIONS,
    VOL_TARGET_ANNUAL,
)
from .settings_bridge import get_cfg, get_settings
from .utils import write_jsonl, read_json, write_json
from .features import compute_panel, bars_from_multiindex
from .news import fetch_news_map
from .agents import score_events_for_symbols, risk_officer_check
from .optimizer import solve_weights, scale_to_target_vol
from .execution import build_order_plans, place_orders_with_limits, estimate_cost_bps
from .universe import build_universe, load_top50_etfs
from .datafeed import get_daily_bars

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
    diag: Dict[str, Any] = {"stage": {}}
    diag["stage"]["universe_count"] = len(symbols)

    lookback_days = int(max(int(cfg.DATA_LOOKBACK_DAYS), 252))
    diag["stage"]["lookback_days"] = lookback_days
    diag["stage"]["fallback_used"] = False

    symbols_for_bars = list(dict.fromkeys(list(symbols) + ["SPY"]))
    raw_bars: pd.DataFrame | None
    if getattr(data_client, "is_simulated", False):
        bars_dict = _fetch_bars(data_client, symbols_for_bars, lookback_days)
        if bars_dict:
            raw_bars = pd.concat(
                {sym: df for sym, df in bars_dict.items() if df is not None and not df.empty},
                names=["symbol", "timestamp"],
            )
        else:
            raw_bars = pd.DataFrame()
        bars = dict(bars_dict)
    else:
        raw_bars = get_daily_bars(symbols_for_bars, lookback_days=lookback_days)
        bars = bars_from_multiindex(raw_bars)

    if raw_bars is None or raw_bars.empty:
        symbols_with_bars = 0
    else:
        symbols_with_bars = int(raw_bars.index.get_level_values(0).unique().size)
    diag["stage"]["symbols_with_bars"] = symbols_with_bars
    universe_count = max(1, int(diag["stage"].get("universe_count", 1)))
    coverage_ratio = symbols_with_bars / universe_count
    diag["stage"]["data_coverage_ratio"] = float(coverage_ratio)
    diag["stage"]["data_coverage"] = "low" if coverage_ratio < 0.7 else "ok"

    symbols = [s for s in symbols if s in bars and len(bars[s]) >= cfg.MIN_BARS]
    if "SPY" not in symbols and "SPY" in bars:
        symbols.append("SPY")

    panel = compute_panel(bars, spy="SPY", fast=cfg.TREND_FAST, slow=cfg.TREND_SLOW,
                          resid_lookback=cfg.RESID_MOM_LOOKBACK, reversal_days=cfg.REVERSAL_DAYS,
                          winsor_pct=cfg.WINSOR_PCT).dropna(subset=["last_price"]).sort_values("score_z", ascending=False)
    valid_df = panel
    diag["stage"]["valid_features"] = int(len(valid_df))

    # Event-driven alpha
    openai_api_key = os.getenv("OPENAI_API_KEY")
    event_alpha_bps: Dict[str, float] = {}
    llm_meta: Dict[str, object] = {
        "called": False,
        "model": os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or C.OPENAI_MODEL,
        "effort": os.getenv("REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or C.OPENAI_REASONING_EFFORT,
        "tokens": 0,
    }
    if cfg.ENABLE_EVENT_SCORE and openai_api_key and len(panel) > 0:
        top_syms = panel.index.tolist()[:cfg.EVENT_TOP_K]
        if api_key in (None, "SIMULATED") or api_sec in (None, "SIMULATED"):
            news_map = {sym: [] for sym in top_syms}
        else:
            news_map = fetch_news_map(top_syms, cfg.NEWS_LOOKBACK_DAYS, api_key, api_sec)
        event_scores, llm_meta = score_events_for_symbols(
            news_map,
            C.OPENAI_MODEL,
            C.OPENAI_REASONING_EFFORT,
            C.EVENT_BPS_PER_DAY,
            max_abs_bps=20,
        )
        event_alpha_bps = {k: float(v) * float(cfg.EVENT_ALPHA_MULT) for k, v in event_scores.items()}

    diag["stage"]["event_scored"] = int(len(event_alpha_bps or {}))
    diag["stage"]["llm_called"] = int(bool(llm_meta.get("called")))
    diag["stage"]["llm_tokens"] = int(llm_meta.get("tokens") or 0)
    diag["stage"]["llm_model"] = llm_meta.get("model")
    diag["stage"]["llm_effort"] = llm_meta.get("effort")

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
    diag["stage"]["candidates_topN"] = int(len(candidates))
    factor_series = factor_component.reindex(candidates).fillna(0.0)
    event_series = event_component.reindex(candidates).fillna(0.0)
    alpha_series = alpha_bps.reindex(candidates).fillna(0.0)
    ranked_symbols = list(alpha_series.sort_values(ascending=False).index)
    top_scores = alpha_series.sort_values(ascending=False).head(10)
    diag["top_candidates"] = [
        {"symbol": str(sym), "score": float(score)}
        for sym, score in top_scores.items()
    ]
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

    fallback_used = False
    nonzero_sum = float(np.sum(np.abs(w_scaled))) if len(w_scaled) else 0.0
    if nonzero_sum <= 1e-9 and len(candidates) >= 5 and float(cfg.TARGET_PORTFOLIO_VOL) > 0:
        k = min(10, len(candidates))
        w_fb = np.zeros_like(w_scaled)
        if k > 0:
            max_weight = float(cfg.NAME_MAX)
            ew = 1.0 / k
            for idx in range(k):
                w_fb[idx] = min(ew, max_weight)
            target_vol_annual = float(cfg.TARGET_PORTFOLIO_VOL)
            if target_vol_annual > 0 and Sig_sub.size > 0:
                w_fb = scale_to_target_vol(w_fb, Sig_sub, target_vol_annual / np.sqrt(252.0))
            w_fb = np.clip(w_fb, 0.0, max_weight)
            total_w = w_fb.sum()
            if total_w > 1.0:
                w_fb = w_fb / total_w
            w_scaled = w_fb
            fallback_used = True
    diag_stage = diag.setdefault("stage", {})
    diag_stage["fallback_used"] = bool(diag_stage.get("fallback_used")) or fallback_used
    diag_stage["optimizer_nonzero"] = int(np.sum(np.abs(w_scaled) > 1e-6))

    target_weights = {s: float(w) for s, w in zip(candidates, w_scaled)}

    settings_dict = get_settings()
    target_slots = int(settings_dict.get("TARGET_POSITIONS", TARGET_POSITIONS))
    target_slots = max(1, min(10, target_slots))
    diag_stage["target_positions"] = target_slots
    diag_stage["nonzero_pre"] = int(sum(1 for w in target_weights.values() if abs(w) > 1e-6))

    if len(target_weights) > target_slots:
        target_weights = dict(
            sorted(target_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:target_slots]
        )

    fill_count = 0
    nonzero_syms = [s for s, w in target_weights.items() if abs(w) > 1e-6]
    if len(nonzero_syms) < min(target_slots, len(ranked_symbols)):
        need = target_slots - len(nonzero_syms)
        fill_syms = [sym for sym in ranked_symbols if sym not in target_weights][:need] if need > 0 else []
        if fill_syms:
            gross_now = sum(abs(w) for w in target_weights.values())
            room = max(0.0, 1.0 - gross_now)
            try:
                cap_weight = float(settings_dict.get("MAX_WEIGHT_PER_NAME", MAX_WEIGHT_PER_NAME))
            except (TypeError, ValueError):
                cap_weight = float(MAX_WEIGHT_PER_NAME)
            add_w = 0.0 if len(fill_syms) == 0 else min(room / len(fill_syms), cap_weight)
            for sym in fill_syms:
                if add_w <= 0:
                    break
                target_weights[sym] = min(cap_weight, add_w)
                fill_count += 1

    try:
        cap_weight = float(settings_dict.get("MAX_WEIGHT_PER_NAME", MAX_WEIGHT_PER_NAME))
    except (TypeError, ValueError):
        cap_weight = float(MAX_WEIGHT_PER_NAME)

    target_weights = {
        s: max(-cap_weight, min(cap_weight, float(w))) for s, w in target_weights.items()
    }

    gross = sum(abs(w) for w in target_weights.values())
    if gross > 0 and gross < GROSS_EXPOSURE_FLOOR:
        scale = min(GROSS_EXPOSURE_FLOOR / gross, 1.0)
        target_weights = {s: max(-cap_weight, min(cap_weight, w * scale)) for s, w in target_weights.items()}
        gross = sum(abs(w) for w in target_weights.values())

    if gross > 1.0 and gross > 0:
        target_weights = {s: w / gross for s, w in target_weights.items()}
        gross = sum(abs(w) for w in target_weights.values())

    diag_stage["nonzero_post"] = int(sum(1 for w in target_weights.values() if abs(w) > 1e-6))
    diag_stage["fill_count"] = int(fill_count)
    diag_stage["gross_final"] = round(gross, 4)

    w_final = np.array([target_weights.get(s, 0.0) for s in candidates], dtype=float)
    diag["exposure"] = {
        "sum_abs_weights": float(sum(abs(w) for w in target_weights.values())),
        "vol_target": float(VOL_TARGET_ANNUAL),
    }

    # Targets
    account = trade_client.get_account()
    equity = float(account.equity)
    investable = equity * (1 - float(cfg.CASH_BUFFER))
    cur_mv_all = {p.symbol: float(p.market_value) for p in trade_client.get_all_positions()}
    targets = {s: float(target_weights.get(s, 0.0) * investable) for s in candidates}
    # Ensure symbols trimmed from the active book head towards zero exposure
    for s in list(cur_mv_all.keys()):
        if s not in targets:
            targets[s] = 0.0

    state["equity"] = equity
    state["cash"] = float(getattr(account, "cash", 0.0) or 0.0)

    # Rebalance bands (skip for initial allocation from cash)
    targets_banded = {}
    filter_counts = {"min_notional_removed": 0, "turnover_cap_removed": 0, "already_at_target_removed": 0}
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
            else:
                filter_counts["already_at_target_removed"] += 1

    # Cost-aware gate
    prices = {s: float(bars[s]["close"].iloc[-1]) for s in candidates if s in bars}
    adv = _estimate_adv(bars)
    expected_alpha_bps = float(np.dot(alpha.reindex(candidates).fillna(0).values, w_final)) * 10000.0
    cost_breakdown = {}
    rebalance_deltas: Dict[str, Dict[str, float]] = {}
    investable_denom = max(1.0, investable)
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
    planned_orders: List[Dict[str, Any]] = []
    order_plans = []
    order_plan_stats: Dict[str, int] = {}
    dust_removed_local = 0
    if targets_banded:
        order_plan_stats = {}
        order_plans = build_order_plans(
            targets_banded,
            cur_mv_all,
            prices,
            adv,
            min_order_notional,
            int(cfg.MAX_SLICES),
            float(cfg.COST_SPREAD_BPS),
            float(cfg.COST_IMPACT_KAPPA),
            float(cfg.COST_IMPACT_PSI),
            order_plan_stats,
        )

        filtered_plans = []
        for plan in order_plans:
            px = float(prices.get(plan.symbol, 0.0))
            notional = abs(float(plan.qty) * px)
            if notional < min_order_notional:
                dust_removed_local += 1
                continue
            filtered_plans.append(plan)
            planned_orders.append(
                {
                    "symbol": plan.symbol,
                    "side": plan.side,
                    "qty": float(plan.qty),
                    "notional": float(notional),
                    "limit_price": float(plan.limit_price) if plan.limit_price is not None else None,
                    "slices": int(plan.slices),
                }
            )
        order_plans = filtered_plans

        # Log onboarding scenario
        if len(order_plans) > 0 and (state.get("cash", 0.0) / max(state.get("equity", 1.0), 1e-9) >= 0.90):
            logger.info(
                "onboarding_detected",
                extra={
                    "cash_ratio": float(state.get("cash", 0.0) / max(state.get("equity", 1.0), 1e-9)),
                    "planned_orders": len(order_plans),
                    "targets_count": len(targets_banded),
                    "reason": "initial_allocation_from_cash",
                },
            )

    filter_counts["min_notional_removed"] = int(order_plan_stats.get("min_notional_removed", 0) + dust_removed_local)

    try:
        equity_snap = float(account.portfolio_value)
        cash_snap = float(account.cash)
    except Exception:
        equity_snap = float(state.get("equity", 0.0))
        cash_snap = float(state.get("cash", 0.0))

    cash_frac = cash_snap / max(equity_snap, 1e-9)
    removed_turnover_cap = 0
    if turnover_cap > 0 and planned_orders:
        allowed_notional = float(turnover_cap) * investable_denom
        total_notional = sum(float(o.get("notional", 0.0)) for o in planned_orders)
        if allowed_notional > 0 and total_notional > allowed_notional and cash_frac < 0.90:
            order_pairs = sorted(
                [(idx, float(planned_orders[idx].get("notional", 0.0))) for idx in range(len(planned_orders))],
                key=lambda kv: kv[1],
            )
            keep_indices = set()
            running = 0.0
            for idx, notional in order_pairs:
                if running + notional <= allowed_notional:
                    keep_indices.add(idx)
                    running += notional
            removed_turnover_cap = len(planned_orders) - len(keep_indices)
            if removed_turnover_cap > 0:
                order_plans = [plan for i, plan in enumerate(order_plans) if i in keep_indices]
                planned_orders = [planned_orders[i] for i in range(len(planned_orders)) if i in keep_indices]
    filter_counts["turnover_cap_removed"] = int(max(0, removed_turnover_cap))

    est_dollars = sum(float(o.get("notional", 0.0)) for o in planned_orders)
    turnover_pct = 100.0 * est_dollars / max(equity_snap, 1e-9)
    cost_bps = 0.0 if len(planned_orders) == 0 else (turnover_pct * float(COST_BPS_PER_1PCT_TURNOVER))
    expected_alpha_bps = float(locals().get("expected_alpha_bps", 0.0))
    net_bps = expected_alpha_bps - cost_bps
    est_cost_bps = cost_bps

    avg_ticket = (est_dollars / len(planned_orders)) if planned_orders else 0.0
    diag_stage["avg_ticket_notional"] = round(avg_ticket, 2) if planned_orders else 0.0

    passes_net = (net_bps >= min_net_bps_to_trade)
    force_onboard = (cash_frac >= 0.90) and (len(planned_orders) > 0)
    proceed = (len(planned_orders) > 0) and (passes_net or force_onboard)

    reasons: List[str] = []
    if len(planned_orders) == 0:
        reasons.append("no_orders")
    if not passes_net:
        reasons.append("net_below_min")
    if force_onboard:
        reasons.append("onboarding")
    if diag.get("stage", {}).get("fallback_used"):
        reasons.append("fallback_portfolio")

    if force_onboard:
        log.info("onboarding", extra={"cash_frac": cash_frac, "reason": "initial allocation from cash"})

    cash_ratio = cash_frac
    onboarding_gate = force_onboard

    diag["filters"] = {
        "min_notional_removed": int(filter_counts.get("min_notional_removed", 0)),
        "turnover_cap_removed": int(filter_counts.get("turnover_cap_removed", 0)),
        "already_at_target_removed": int(filter_counts.get("already_at_target_removed", 0)),
    }
    diag["planned_orders_preview"] = [
        {
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "notional": float(order.get("notional", 0.0)),
        }
        for order in planned_orders[:5]
    ]

    # Risk officer
    proposed_w = {s: target_weights.get(s, 0.0) for s in candidates}
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
    if not risk_officer_approved and len(planned_orders) > 0:
        if "risk_officer_blocked" not in reasons:
            reasons.append("risk_officer_blocked")

    proceed_final = proceed and risk_officer_approved

    mode = "sim" if getattr(trade_client, "is_simulated", False) else (
        "live" if os.environ.get("ALPACA_PAPER", "true").lower() == "false" else "paper"
    )
    model_name = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or C.OPENAI_MODEL
    effort = os.getenv("REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or C.OPENAI_REASONING_EFFORT
    regime = locals().get("regime_summary", {})
    target_gross = sum(abs(w) for w in target_weights.values())
    exposure = {
        "target_gross": float(target_gross),
        "sum_abs_weights": float(target_gross),
        "vol_target": float(VOL_TARGET_ANNUAL),
    }

    gate_reason = None
    for r in reasons:
        if r != "onboarding":
            gate_reason = r
            break

    gate = {
        "mode": mode,
        "model": model_name,
        "effort": effort,
        "equity": float(equity_snap),
        "cash": float(cash_snap),
        "cash_frac": float(cash_frac),
        "order_count": len(planned_orders),
        "turnover_pct": float(turnover_pct),
        "expected_alpha_bps": float(expected_alpha_bps),
        "cost_bps": float(cost_bps),
        "net_bps": float(net_bps),
        "min_notional": float(min_order_notional),
        "min_net_bps": float(min_net_bps_to_trade),
        "proceed": bool(proceed),
        "proceed_final": bool(proceed_final),
        "reasons": list(reasons),
        "passes_net_bps": bool(passes_net),
        "force_onboard": bool(force_onboard),
        "risk_officer_approved": bool(risk_officer_approved),
        "exposure": exposure,
        "regime": regime,
        "turnover_cap": float(turnover_cap),
        "reason_primary": gate_reason,
    }

    try:
        log.info("gate_check", extra={"gate": gate, "diag": diag})
    except Exception:
        pass

    orders = []
    if proceed_final and order_plans:
        if bool(cfg.DRY_RUN):
            orders = ["DRY_RUN_NO_ORDERS"]
        else:
            order_ids = place_orders_with_limits(
                trade_client,
                order_plans,
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
        proceed,
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
        "gate": gate,
        "diag": diag,
    }
    write_jsonl(C.EPISODES_PATH, ep)
    return ep
