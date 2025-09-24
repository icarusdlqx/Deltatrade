# Deltatrade V1 (Autonomous, Event-Aware, Risk-Targeted)

**What it does**
- Blends **cross-sectional** signals (residual momentum, trend, short reversal, quality) with **event-driven** alpha (news → GPT-5 JSON scores).
- Builds a **constrained portfolio** via a convex optimizer (name/sector caps, turnover penalty), then **targets portfolio volatility**.
- Uses **cost-aware gating** and **sliced, pegged limit execution** (fallback to small market slices).
- Fully autonomous via a **scheduler loop**; settings are configurable live via the web UI.

## Quick start

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Set environment
```bash
export ALPACA_API_KEY=...          # required
export ALPACA_SECRET_KEY=...       # required
export ALPACA_PAPER=true           # or false for live
export OPENAI_API_KEY=...          # required for event scoring
```

### 3) Run locally (dev)
```bash
# start scheduler in background + web in foreground (port 8000)
nohup python scheduler.py >/tmp/scheduler.log 2>&1 &
gunicorn -k gevent -b 0.0.0.0:8000 webapp:app
# open http://localhost:8000/dashboard
```

### Replit (one-liner)
The repo includes `.replit` that starts the scheduler and web app automatically.  
Or manual shell:
```bash
nohup python scheduler.py >/tmp/scheduler.log 2>&1 & exec gunicorn -k gevent -b 0.0.0.0:$PORT webapp:app
```

### Heroku-like (Procfile)
Single dyno launches both scheduler and web (scheduler in background).

---

## Folder tour

- `v2/config.py`: Canonical defaults (can be overridden in UI).
- `v2/settings_bridge.py`: Loads/saves live overrides to `data/settings_overrides.json`.
- `v2/features.py`: Factor engineering & residual momentum.
- `v2/news.py`: Alpaca news fetch (robust import).
- `v2/agents.py`: OpenAI EventScore + RiskOfficer checks.
- `v2/optimizer.py`: SLSQP weights + portfolio vol targeting.
- `v2/execution.py`: Cost model, order planning, sliced limit orders.
- `v2/orchestrator.py`: The single-cycle trading brain.
- `scheduler.py`: Repeatedly runs `orchestrator.run_once()` in ET windows.
- `webapp.py`: Dashboard (episodes) + Settings (live overrides).
- `data/`: Logs and state files.

**Note on universe:** By default uses an ETF list. If you want S&P500 + ETFs, drop a `data/sp500.csv` (with a header `Symbol`) and the app will use it when present.
