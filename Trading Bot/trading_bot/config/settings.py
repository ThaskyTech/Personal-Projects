"""
config/settings.py
Central configuration for the AI Swing Trading Bot v2.0
All parameters derived from the corrected blueprint.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENTS
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS = ["EURUSD", "GBPUSD"]

# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAMES  (MT5 constants mapped by name for portability)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME_H4  = "H4"
TIMEFRAME_M15 = "M15"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZATION_WINDOW = 500        # bars — fixed, not subject to optimisation
ATR_PERIOD          = 14
RSI_PERIOD          = 14
EMA_FAST            = 50
EMA_SLOW            = 200
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
STOCH_K             = 14
STOCH_D             = 3

# ─────────────────────────────────────────────────────────────────────────────
# H4 DIRECTIONAL BIAS MODEL
# ─────────────────────────────────────────────────────────────────────────────
H4_BIAS_THRESHOLD          = 0.60   # minimum P(direction) for full-weight signal
H4_BIAS_NEUTRAL_PENALTY    = 0.05   # added to M15 threshold when H4 is neutral

# ─────────────────────────────────────────────────────────────────────────────
# M15 ENTRY SIGNAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
M15_BASE_ENTRY_THRESHOLD   = 0.60   # minimum entry probability
M15_MIN_RR_RATIO           = 1.50   # minimum expected_return / expected_downside

# ─────────────────────────────────────────────────────────────────────────────
# EXIT STRATEGY  (deterministic — ML cannot modify)
# ─────────────────────────────────────────────────────────────────────────────
SL_ATR_MULTIPLIER          = 1.5
TP_ATR_MULTIPLIER          = 2.5
# Resulting R:R = 2.5 / 1.5 = 1.667

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT          = 0.0075   # 0.75 % of account equity
RISK_PER_TRADE_RECOVERY_PCT = 0.0040   # 0.40 % during drawdown recovery

MAX_TRADES_PER_PAIR         = 3
MAX_EXPOSURE_PER_PAIR_PCT   = 0.0225   # 2.25 %

# Portfolio-level caps (NEW — blueprint v2.0)
MAX_TOTAL_OPEN_RISK_PCT     = 0.0400   # 4.0 % across all positions
MAX_USD_DIRECTIONAL_RISK_PCT= 0.0300   # 3.0 % combined USD directional exposure

# Drawdown limits
DAILY_LOSS_LIMIT_PCT        = 0.0200   # 2 %
MONTHLY_LOSS_LIMIT_PCT      = 0.0600   # 6 %
DRAWDOWN_RECOVERY_TRIGGER   = 0.0600   # enter recovery mode at 6 % drawdown

# ─────────────────────────────────────────────────────────────────────────────
# SPREAD / SLIPPAGE FILTERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_SPREAD_PIPS: Dict[str, float] = {
    "EURUSD": 2.0,
    "GBPUSD": 2.5,
}
MAX_SLIPPAGE_PIPS           = 2.0
ORDER_RETRY_ATTEMPTS        = 3

# ─────────────────────────────────────────────────────────────────────────────
# ECONOMIC NEWS FILTER
# ─────────────────────────────────────────────────────────────────────────────
NEWS_BLACKOUT_MINUTES_BEFORE = 30
NEWS_BLACKOUT_MINUTES_AFTER  = 30
NEWS_API_TIMEOUT_SECONDS     = 10
NEWS_API_RETRY_INTERVAL_MIN  = 5
NEWS_CURRENCIES              = ["USD", "EUR", "GBP"]
NEWS_IMPACT_LEVELS           = ["HIGH"]

# ─────────────────────────────────────────────────────────────────────────────
# MT5 CONNECTIVITY
# ─────────────────────────────────────────────────────────────────────────────
MT5_RECONNECT_INTERVAL_SEC  = 30
MT5_RECONNECT_MAX_ATTEMPTS  = 20    # 20 × 30s = 10 minutes

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD OPTIMISATION (QMRP)
# ─────────────────────────────────────────────────────────────────────────────
WFO_TRAIN_MONTHS            = 12
WFO_TEST_MONTHS             = 6
WFO_VALIDATION_MONTHS       = 6
WFO_ROLL_MONTHS             = 3
WFO_MIN_TRADES              = 80
WFO_MAX_DRAWDOWN            = 0.10
MIN_HISTORY_YEARS           = 10

# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
REGIME_LABELS = ["trending", "ranging", "high_vol", "low_vol"]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING / DATABASE
# ─────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DB_TABLE_TRADES = "trades"

# ─────────────────────────────────────────────────────────────────────────────
# CAPITAL SCALING
# ─────────────────────────────────────────────────────────────────────────────
SCALING_FREQUENCY_MONTHS    = 3     # quarterly

# ─────────────────────────────────────────────────────────────────────────────
# SESSION FLAGS (UTC hours)
# ─────────────────────────────────────────────────────────────────────────────
LONDON_SESSION_START_UTC    = 8
LONDON_SESSION_END_UTC      = 16
NY_SESSION_START_UTC        = 13
NY_SESSION_END_UTC          = 21
