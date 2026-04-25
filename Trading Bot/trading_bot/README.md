# AI Swing Trading Bot v2.0
### Regime-Aware Feature-Driven Multi-Timeframe Swing Trading Engine

---

## Overview

This bot implements the corrected blueprint v2.0 exactly:
- **Instruments:** EURUSD, GBPUSD (independent models per pair)
- **Broker:** FP Markets — Razor Account
- **Platform:** MetaTrader 5
- **Strategy:** H4 directional bias + M15 precision entry
- **ML Models:** LightGBM gradient boosting (falls back to sklearn if LightGBM unavailable)
- **Target:** 20–30% annual return | ≤10% max drawdown

---

## Project Structure

```
trading_bot/
├── bot.py                    ← Main bot (run this for live trading)
├── config/
│   └── settings.py           ← All parameters from blueprint
├── data/
│   ├── data_engine.py        ← Data loading, MT5 fetching, validation
│   └── historical/           ← Place CSV/Parquet data files here
├── features/
│   └── feature_engineer.py   ← H4 + M15 features, 500-bar normalisation
├── regime/
│   └── regime_classifier.py  ← ADX/ATR-based regime detection
├── models/
│   ├── h4_bias_model.py      ← H4 directional bias (LightGBM)
│   ├── m15_entry_model.py    ← M15 entry signal (LightGBM)
│   └── saved/                ← Trained model files saved here
├── risk/
│   └── risk_engine.py        ← Deterministic risk management
├── news/
│   └── news_filter.py        ← Economic calendar + failsafe
├── execution/
│   └── mt5_engine.py         ← MT5 orders, reconnect handling
├── backtest/
│   └── walk_forward.py       ← QMRP walk-forward optimisation
├── utils/
│   └── logger.py             ← SQLite trade database + logging
├── scripts/
│   ├── train.py              ← Training pipeline
│   ├── qmrp.py               ← Quarterly model refresh
│   └── report.py             ← Performance reporting
├── logs/                     ← Trade DB, log files, backups
└── .env.example              ← Copy to .env and fill in credentials
```

---

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```
> **Note:** MetaTrader5 package is Windows-only. All other components work cross-platform.

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your MT5 credentials and news API key
```

### 3. Add historical data
Place 10+ years of OHLCV CSV data in `data/historical/`:
```
data/historical/EURUSD_H4.csv
data/historical/EURUSD_M15.csv
data/historical/GBPUSD_H4.csv
data/historical/GBPUSD_M15.csv
```
See `data/historical/README.txt` for format and data source recommendations.

### 4. Train models
```bash
# Full QMRP training (recommended — runs walk-forward optimisation)
python scripts/train.py

# Fast training (skips walk-forward, trains on full dataset directly)
python [README.md](README.md)scripts/train.py --skip-wfo

# Train single instrument
python scripts/train.py --instrument EURUSD
```

### 5. Start live trading
```bash
python bot.py
```

---

## Key Configuration (config/settings.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RISK_PER_TRADE_PCT` | 0.75% | Normal risk per trade |
| `RISK_PER_TRADE_RECOVERY_PCT` | 0.40% | Risk during drawdown recovery |
| `MAX_TOTAL_OPEN_RISK_PCT` | 4.0% | Portfolio-level exposure cap |
| `MAX_USD_DIRECTIONAL_RISK_PCT` | 3.0% | Combined USD directional cap |
| `H4_BIAS_THRESHOLD` | 0.60 | Min probability for full-weight H4 bias |
| `SL_ATR_MULTIPLIER` | 1.5× | Stop loss distance |
| `TP_ATR_MULTIPLIER` | 2.5× | Take profit distance |
| `NORMALIZATION_WINDOW` | 500 bars | Rolling normalisation window |
| `WFO_TRAIN_MONTHS` | 12 | Walk-forward training window |
| `DRAWDOWN_RECOVERY_TRIGGER` | 6% | Drawdown level that triggers recovery mode |

All parameters are in `config/settings.py`. Edit there to adjust the system.

---

## Quarterly Model Refresh (QMRP)

Run every 3 months:
```bash
python scripts/qmrp.py
```

This will:
1. Recalculate all features on extended dataset
2. Run walk-forward optimisation
3. Compare new model vs production Sortino ratio
4. Deploy only if new model is strictly better
5. Export trades CSV for analysis

**Schedule via Windows Task Scheduler or cron:**
```bash
# Cron example (1st of Jan, Apr, Jul, Oct at 2am)
0 2 1 1,4,7,10 * cd /path/to/trading_bot && python scripts/qmrp.py
```

---

## Performance Reporting

```bash
python scripts/report.py --days 30    # Last 30 days
python scripts/report.py --days 90    # Last quarter
```

---

## News API Setup

Get a free API key from [Finnhub](https://finnhub.io) and add to `.env`:
```
NEWS_API_PROVIDER=finnhub
NEWS_API_KEY=your_key_here 
```

If the API is unavailable, the bot automatically enters **News Blackout Mode** 
and suspends new trades until the API recovers (retry every 5 minutes).

---

## Risk Controls Summary

All risk rules are **deterministic** — ML cannot override them:

| Rule | Value |
|------|-------|
| Risk per trade (normal) | 0.75% equity |
| Risk per trade (recovery) | 0.40% equity |
| Max trades per pair | 3 |
| Max pair exposure | 2.25% |
| Max portfolio exposure | 4.0% |
| Max USD directional exposure | 3.0% |
| Daily loss limit | 2% |
| Monthly loss limit | 6% |
| Drawdown recovery trigger | 6% |

---

## Blueprint Compliance

All 12 issues from the v1.0 review have been implemented:

| # | Issue | Implementation |
|---|-------|----------------|
| 1 | Partial profit removed | `bot.py` — no partial close calls |
| 2 | 10-year data requirement | `data_engine.py` → `validate_minimum_history()` |
| 3 | QMRP unified | `backtest/walk_forward.py` → `WalkForwardOptimiser.run_qmrp()` |
| 4 | USD correlation hard cap | `risk_engine.py` → `_check_usd_correlation()` |
| 5 | Portfolio exposure cap | `risk_engine.py` → `MAX_TOTAL_OPEN_RISK_PCT` |
| 6 | H4 threshold defined | `h4_bias_model.py` → `H4_BIAS_THRESHOLD = 0.60` |
| 7 | News API failsafe | `news_filter.py` → blackout mode on API failure |
| 8 | Norm window specified | `settings.py` → `NORMALIZATION_WINDOW = 500` |
| 9 | M30 fallback removed | Not present in codebase |
| 10 | SQLite storage defined | `utils/logger.py` → `TradeDatabase` |
| 11 | MT5 disconnect handling | `execution/mt5_engine.py` → `reconnect_loop()` |
| 12 | Scaling vs recovery | `risk_engine.py` → `_pending_lot_update` |

---

## Disclaimer

This bot is provided for educational and research purposes.
Algorithmic trading involves significant financial risk.
Past backtest performance does not guarantee future results.
Always test on a demo account before deploying real capital. 
Also, if you seek to see prior proto types to gain a better understanding
of the changes and improvements implemented in v2.0, do contact me via my
portfolio site at www.lethasky.co.za.
