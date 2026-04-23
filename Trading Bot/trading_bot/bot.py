"""
bot.py
Main trading bot orchestrator.
Wires all modules together and runs the main trading loop.
Blueprint v2.0 — all seven core modules integrated.
"""

import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from config.settings import INSTRUMENTS, TIMEFRAME_H4, TIMEFRAME_M15
from data.data_engine import DataEngine
from execution.mt5_engine import MT5Engine
from features.feature_engineer import FeatureEngineer, _atr
from models.h4_bias_model import H4BiasModel
from models.m15_entry_model import M15EntryModel
from news.news_filter import NewsFilter
from regime.regime_classifier import RegimeClassifier
from risk.risk_engine import RiskEngine
from utils.logger import setup_logger, TradeDatabase

log = setup_logger("TradingBot", level=os.getenv("LOG_LEVEL", "INFO"))

# Live bar lookback for feature calculation
LIVE_BARS = 800   # enough for 500-bar norm window + buffer


class TradingBot:
    """
    Regime-Aware Feature-Driven Multi-Timeframe Swing Trading Engine.
    Operates 24/5 on EURUSD and GBPUSD via FP Markets MT5 Razor account.
    """

    def __init__(self):
        # ── Environment ───────────────────────────────────────────────────────
        self.mt5_login    = int(os.getenv("MT5_LOGIN", "0"))
        self.mt5_password = os.getenv("MT5_PASSWORD", "")
        self.mt5_server   = os.getenv("MT5_SERVER", "FPMarkets-Live")
        self.mt5_path     = os.getenv("MT5_PATH", "")
        initial_equity    = float(os.getenv("INITIAL_EQUITY", "10000"))

        # ── Core modules ──────────────────────────────────────────────────────
        self.data_engine  = DataEngine(use_mt5=True)
        self.feat_eng     = FeatureEngineer()
        self.regime_clf   = RegimeClassifier()
        self.news_filter  = NewsFilter(
            provider=os.getenv("NEWS_API_PROVIDER", "finnhub"),
            api_key=os.getenv("NEWS_API_KEY", ""),
        )
        self.risk_engine  = RiskEngine(initial_equity)
        self.mt5          = MT5Engine(
            self.mt5_login, self.mt5_password, self.mt5_server, self.mt5_path)
        self.db           = TradeDatabase(os.getenv("DB_PATH", "logs/trades.db"))

        # ── Per-instrument models ─────────────────────────────────────────────
        self.h4_models: Dict[str, H4BiasModel]    = {}
        self.m15_models: Dict[str, M15EntryModel] = {}
        for inst in INSTRUMENTS:
            self.h4_models[inst]  = H4BiasModel(inst)
            self.m15_models[inst] = M15EntryModel(inst)

        self._running = False

        # Graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    # ─────────────────────────────────────────────────────────────────────────
    # Startup
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        log.info("=" * 60)
        log.info("  AI SWING TRADING BOT v2.0 — STARTING")
        log.info("=" * 60)

        # Connect to MT5
        if not self.mt5.connect():
            log.error("Failed to connect to MT5. Exiting.")
            sys.exit(1)

        # Sync equity from live account
        acct = self.mt5.get_account_info()
        if acct:
            self.risk_engine.update_equity(acct["equity"])
            log.info(f"Account equity: {acct['equity']} {acct['currency']}")

        # Load models
        models_ok = all(
            self.h4_models[i].load() and self.m15_models[i].load()
            for i in INSTRUMENTS
        )
        if not models_ok:
            log.warning("One or more models not found. "
                        "Run 'python scripts/train.py' to train before live trading.")

        # Initial news calendar refresh
        self.news_filter.refresh()

        self._running = True
        log.info("Bot started. Entering main loop...")
        self._main_loop()

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop (runs every completed M15 candle)
    # ─────────────────────────────────────────────────────────────────────────

    def _main_loop(self):
        last_candle_time: Dict[str, datetime] = {}

        while self._running:
            try:
                # ── Connectivity check ────────────────────────────────────────
                if not self.mt5.ensure_connected():
                    log.warning("MT5 unavailable — sleeping 60s")
                    time.sleep(60)
                    continue

                # ── Equity sync ───────────────────────────────────────────────
                acct = self.mt5.get_account_info()
                if acct:
                    self.risk_engine.update_equity(acct["equity"])
                    self.db.log_equity(
                        acct["equity"], acct["balance"],
                        self.risk_engine.get_risk_pct()
                    )

                # ── Daily DB backup (once per day) ────────────────────────────
                self._daily_backup()

                # ── Per-instrument processing ─────────────────────────────────
                open_positions = self.mt5.get_open_positions()
                for instrument in INSTRUMENTS:
                    self._process_instrument(instrument, open_positions, last_candle_time)

                # Sleep until next M15 boundary (check every 30s)
                time.sleep(30)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(60)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-instrument signal processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_instrument(self, instrument: str,
                             open_positions: List[Dict],
                             last_candle_time: Dict):
        """Full signal pipeline for one instrument on one M15 bar."""

        # ── Fetch latest bars ─────────────────────────────────────────────────
        m15_df = self.data_engine.fetch_live_bars(instrument, TIMEFRAME_M15, count=LIVE_BARS)
        h4_df  = self.data_engine.fetch_live_bars(instrument, TIMEFRAME_H4,  count=LIVE_BARS)
        if m15_df is None or h4_df is None or len(m15_df) < 600:
            log.warning(f"Insufficient bars for {instrument}")
            return

        # ── Check for new candle (avoid re-processing same bar) ───────────────
        latest_ts = m15_df.index[-1]
        if last_candle_time.get(instrument) == latest_ts:
            return
        last_candle_time[instrument] = latest_ts

        # ── News filter ───────────────────────────────────────────────────────
        permitted, news_reason = self.news_filter.is_trading_permitted()
        if not permitted:
            log.info(f"{instrument}: {news_reason}")
            return

        # ── Feature engineering ───────────────────────────────────────────────
        h4_features  = self.feat_eng.compute_h4_features(h4_df)
        m15_features = self.feat_eng.compute_m15_features(m15_df)

        if h4_features.empty or m15_features.empty:
            return

        # ── Regime classification ─────────────────────────────────────────────
        regime = self.regime_clf.classify_latest(h4_df)

        # ── H4 directional bias ───────────────────────────────────────────────
        h4_model = self.h4_models.get(instrument)
        if h4_model is None or h4_model.model is None:
            log.debug(f"{instrument}: H4 model not loaded")
            return

        h4_row  = h4_features.iloc[-1]
        h4_bias = h4_model.predict(h4_row)
        log.debug(f"{instrument} H4 bias: {h4_bias}")

        # ── M15 entry signal ──────────────────────────────────────────────────
        m15_model = self.m15_models.get(instrument)
        if m15_model is None or m15_model.long_model is None:
            log.debug(f"{instrument}: M15 model not loaded")
            return

        # Merge H4 context
        m15_ctx  = FeatureEngineer.merge_h4_context_into_m15(m15_features, h4_features)
        m15_row  = m15_ctx.iloc[-1]
        atr_ser  = _atr(m15_df["high"], m15_df["low"], m15_df["close"])
        atr      = float(atr_ser.iloc[-1])
        if atr == 0 or atr != atr:   # NaN check
            return

        current_price = float(m15_df["close"].iloc[-1])

        # Determine candidate directions from H4 bias
        directions = []
        if h4_bias["bias"] in ("long", "neutral"):
            directions.append("long")
        if h4_bias["bias"] in ("short", "neutral"):
            directions.append("short")

        for direction in directions:
            signal = m15_model.predict(m15_row, direction, atr, h4_bias, regime)

            if not signal["signal_valid"]:
                continue

            log.info(f"{instrument} SIGNAL: {direction.upper()} | "
                     f"prob={signal['entry_probability']:.3f} | "
                     f"R:R={signal['rr_ratio']:.2f}")

            # ── Risk checks ───────────────────────────────────────────────────
            sl_price, tp_price = RiskEngine.calculate_sl_tp(current_price, atr, direction)
            lot_size, risk_usd = self.risk_engine.calculate_lot_size(
                instrument, current_price, sl_price)
            risk_pct = self.risk_engine.get_risk_pct()

            allowed, risk_reason = self.risk_engine.can_open_trade(
                instrument, direction, risk_pct, open_positions)

            if not allowed:
                log.info(f"{instrument} {direction}: blocked by risk engine — {risk_reason}")
                continue

            # ── Execute order ─────────────────────────────────────────────────
            result = self.mt5.place_order(
                instrument, direction, lot_size, sl_price, tp_price,
                comment=f"SwingBot_{direction[:1].upper()}"
            )

            if result and result.get("success"):
                trade_id = str(uuid.uuid4())
                entry_price = result["price"]
                self.db.insert_trade({
                    "trade_id":           trade_id,
                    "instrument":         instrument,
                    "direction":          direction,
                    "entry_time":         datetime.now(timezone.utc).isoformat(),
                    "entry_price":        entry_price,
                    "sl_price":           sl_price,
                    "tp_price":           tp_price,
                    "lot_size":           lot_size,
                    "spread_at_entry":    m15_df["spread"].iloc[-1],
                    "entry_probability":  signal["entry_probability"],
                    "expected_return":    signal["expected_return"],
                    "expected_downside":  signal["expected_downside"],
                    "regime_trending":    regime.get("trending", 0),
                    "regime_ranging":     regime.get("ranging", 0),
                    "regime_high_vol":    regime.get("high_vol", 0),
                    "regime_low_vol":     regime.get("low_vol", 0),
                    "h4_bias_long":       h4_bias["p_long"],
                    "h4_bias_short":      h4_bias["p_short"],
                    "mt5_ticket":         result.get("ticket"),
                })
                log.info(f"{instrument} {direction} order filled | "
                         f"entry={entry_price} | SL={sl_price} | TP={tp_price} | lots={lot_size}")

            break   # one signal per bar per instrument

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    _last_backup_day = None

    def _daily_backup(self):
        today = datetime.now(timezone.utc).date()
        if self._last_backup_day != today:
            backup_path = self.db.backup(os.getenv("DB_BACKUP_PATH", "logs/backups"))
            log.info(f"Daily DB backup: {backup_path}")
            self._last_backup_day = today

    def _shutdown_handler(self, signum, frame):
        log.info("Shutdown signal received — stopping bot gracefully...")
        self._running = False
        self.mt5.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
