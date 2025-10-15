# Deltatrade V1 - Autonomous Trading System

## Overview
This is an autonomous, event-aware, risk-targeted trading system that:
- Blends cross-sectional signals (residual momentum, trend, short reversal, quality) with event-driven alpha using GPT analysis
- Builds constrained portfolios via convex optimization 
- Uses cost-aware gating and sliced, pegged limit execution
- Runs fully autonomously via scheduler with configurable settings via web UI

## Current State
✅ Successfully imported and configured for Replit environment
✅ Python 3.11 and all dependencies installed
✅ Flask web app configured on port 5000 with 0.0.0.0 host
✅ Scheduler modified to handle missing API keys gracefully (runs in simulated mode)
✅ Workflow configured to run both scheduler and web app together
✅ Deployment configured for production with VM target
✅ Application tested and working correctly

## Project Architecture
- **Frontend**: Flask web application on port 5000
- **Backend**: Python scheduler running trading logic
- **Mode**: Currently running in simulated mode (no API keys configured)
- **Data**: Stored in `/data` directory
- **Configuration**: Live settings via web UI, stored in `data/settings_overrides.json`

## Key Files
- `webapp.py`: Flask web application with dashboard, positions, settings
- `scheduler.py`: Autonomous trading scheduler (modified for Replit)
- `v2/orchestrator.py`: Core trading logic
- `v2/config.py`: Default configuration settings
- `requirements.txt`: Python dependencies

## Environment Setup
- **Python**: 3.11 with all required packages
- **Web Server**: Gunicorn with gevent workers
- **Port**: 5000 (configured for Replit proxy)
- **Host**: 0.0.0.0 (allows access through Replit's iframe proxy)

## API Keys
✅ **Configured and Working:**
- `ALPACA_API_KEY` - Connected to Alpaca paper trading account
- `ALPACA_SECRET_KEY` - Authenticated successfully
- `OPENAI_API_KEY` - GPT analysis enabled

**Current Mode:** Paper Trading (Alpaca account synced)

## Recent Changes
**2025-10-15:**
- ✅ Fixed `from __future__ import annotations` syntax errors in webapp.py and scheduler.py
- ✅ Configured Alpaca API keys (paper trading account)
- ✅ Configured OpenAI API key for GPT analysis
- ✅ Connected to live Alpaca paper account (Equity: $10,170)
- ✅ Verified automatic balance sync - system uses real Alpaca equity for position sizing
- ✅ All services running: web dashboard + automated scheduler

**2025-09-24:**
- Configured Flask app to use port 5000 instead of 8000
- Modified scheduler.py to handle missing API keys gracefully  
- Set up Replit workflow to run scheduler + web app together
- Configured deployment settings for production VM
- Added data directory structure
- Verified application startup and functionality

## User Preferences
- No specific user preferences documented yet