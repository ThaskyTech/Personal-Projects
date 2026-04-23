"""
execution/mt5_engine.py
MetaTrader 5 execution engine.
Handles connection, order placement, monitoring, and reconnect logic.
Blueprint v2.0 Section 13.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config.settings import (
    MAX_SPREAD_PIPS, MAX_SLIPPAGE_PIPS, ORDER_RETRY_ATTEMPTS,
    MT5_RECONNECT_INTERVAL_SEC, MT5_RECONNECT_MAX_ATTEMPTS,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
)
from utils.logger import setup_logger

log = setup_logger("MT5Engine")

# Trade direction constants
ORDER_BUY  = 0
ORDER_SELL = 1


class MT5Engine:
    """
    Wraps MetaTrader5 Python API with connection management,
    spread filtering, slippage control, and full disconnect/reconnect handling.
    """

    def __init__(self, login: int, password: str, server: str, path: str = ""):
        self.login    = login
        self.password = password
        self.server   = server
        self.path     = path
        self._connected          = False
        self._monitoring_only    = False   # safe mode after extended disconnect
        self._last_known_positions: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise MT5 connection."""
        try:
            import MetaTrader5 as mt5
            kwargs = {"login": self.login, "password": self.password, "server": self.server}
            if self.path:
                kwargs["path"] = self.path
            if not mt5.initialize(**kwargs):
                log.error(f"MT5 initialise failed: {mt5.last_error()}")
                return False
            info = mt5.account_info()
            if info is None:
                log.error("MT5 account_info returned None")
                return False
            self._connected       = True
            self._monitoring_only = False
            log.info(f"MT5 connected — account {info.login} | balance {info.balance} "
                     f"| equity {info.equity} | server {info.server}")
            return True
        except ImportError:
            log.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False
        except Exception as e:
            log.error(f"MT5 connect error: {e}")
            return False

    def disconnect(self):
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
            self._connected = False
            log.info("MT5 disconnected")
        except Exception:
            pass

    def is_connected(self) -> bool:
        try:
            import MetaTrader5 as mt5
            return mt5.terminal_info() is not None
        except Exception:
            return False

    def reconnect_loop(self) -> bool:
        """
        Attempt reconnection every MT5_RECONNECT_INTERVAL_SEC seconds.
        After MT5_RECONNECT_MAX_ATTEMPTS failures, enter monitoring-only mode.
        Blueprint v2.0 Issue #11.
        """
        log.warning("MT5 connection lost — starting reconnect loop")
        for attempt in range(1, MT5_RECONNECT_MAX_ATTEMPTS + 1):
            log.info(f"Reconnect attempt {attempt}/{MT5_RECONNECT_MAX_ATTEMPTS}...")
            if self.connect():
                log.info("Reconnect successful")
                self._reconcile_positions()
                return True
            time.sleep(MT5_RECONNECT_INTERVAL_SEC)

        log.error("Reconnect failed after max attempts — entering MONITORING-ONLY mode. "
                  "Manual restart required.")
        self._monitoring_only = True
        self._connected       = False
        return False

    def _reconcile_positions(self):
        """
        After reconnect, reconcile live positions against last known local state.
        Logs any discrepancies for review.
        """
        live = self.get_open_positions()
        live_tickets = {p["ticket"] for p in live}
        cached_tickets = {p.get("mt5_ticket") for p in self._last_known_positions}

        opened_while_offline = live_tickets - cached_tickets
        closed_while_offline = cached_tickets - live_tickets

        if opened_while_offline:
            log.warning(f"Positions opened while offline: {opened_while_offline}")
        if closed_while_offline:
            log.warning(f"Positions closed while offline: {closed_while_offline}")

        self._last_known_positions = live

    def ensure_connected(self) -> bool:
        """Check connection; trigger reconnect loop if down."""
        if self.is_connected():
            return True
        return self.reconnect_loop()

    # ─────────────────────────────────────────────────────────────────────────
    # Order execution
    # ─────────────────────────────────────────────────────────────────────────

    def place_order(self,
                    instrument: str,
                    direction: str,
                    lot_size: float,
                    sl_price: float,
                    tp_price: float,
                    comment: str = "SwingBot",
                    ) -> Optional[Dict]:
        """
        Place a market order with SL and TP.
        Applies spread filter, slippage tolerance, and retry logic.

        Returns order result dict or None on failure.
        """
        if self._monitoring_only:
            log.warning("Monitoring-only mode — order suppressed")
            return None

        if not self.ensure_connected():
            log.error("Cannot place order — MT5 not connected")
            return None

        # Spread filter
        spread_ok, spread_reason = self._check_spread(instrument)
        if not spread_ok:
            log.info(f"Order suppressed: {spread_reason}")
            return None

        for attempt in range(1, ORDER_RETRY_ATTEMPTS + 1):
            result = self._send_market_order(instrument, direction, lot_size,
                                              sl_price, tp_price, comment)
            if result and result.get("success"):
                log.info(f"Order placed: {instrument} {direction} {lot_size} lots | "
                         f"ticket={result.get('ticket')} | entry={result.get('price')}")
                return result
            log.warning(f"Order attempt {attempt} failed: {result}")
            time.sleep(0.5)

        log.error(f"Order failed after {ORDER_RETRY_ATTEMPTS} attempts: {instrument} {direction}")
        return None

    def _send_market_order(self, instrument: str, direction: str, lot_size: float,
                            sl_price: float, tp_price: float, comment: str) -> Optional[Dict]:
        try:
            import MetaTrader5 as mt5
            order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

            tick = mt5.symbol_info_tick(instrument)
            if tick is None:
                return {"success": False, "error": "No tick data"}

            price = tick.ask if direction == "long" else tick.bid
            deviation = int(MAX_SLIPPAGE_PIPS * 10)  # slippage in points

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    instrument,
                "volume":    lot_size,
                "type":      order_type,
                "price":     price,
                "sl":        sl_price,
                "tp":        tp_price,
                "deviation": deviation,
                "magic":     20260101,
                "comment":   comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                return {"success": False, "error": str(mt5.last_error())}
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"success": False, "retcode": result.retcode, "comment": result.comment}

            return {
                "success": True,
                "ticket":  result.order,
                "price":   result.price,
                "volume":  result.volume,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # Position monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def get_open_positions(self) -> List[Dict]:
        """Return all open positions as list of dicts."""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                result.append({
                    "ticket":     p.ticket,
                    "instrument": p.symbol,
                    "direction":  "long" if p.type == 0 else "short",
                    "lot_size":   p.volume,
                    "entry_price": p.price_open,
                    "sl_price":   p.sl,
                    "tp_price":   p.tp,
                    "pnl":        p.profit,
                    "swap":       p.swap,
                    "open_time":  datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                    "comment":    p.comment,
                })
            self._last_known_positions = result
            return result
        except Exception as e:
            log.error(f"get_open_positions error: {e}")
            return self._last_known_positions   # return cached on error

    def get_account_info(self) -> Optional[Dict]:
        """Return account equity, balance, margin."""
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info is None:
                return None
            return {
                "equity":        info.equity,
                "balance":       info.balance,
                "margin":        info.margin,
                "free_margin":   info.margin_free,
                "margin_level":  info.margin_level,
                "currency":      info.currency,
            }
        except Exception as e:
            log.error(f"get_account_info error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Spread filter
    # ─────────────────────────────────────────────────────────────────────────

    def _check_spread(self, instrument: str) -> Tuple[bool, str]:
        max_spread = MAX_SPREAD_PIPS.get(instrument, 3.0)
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(instrument)
            if tick is None:
                return False, f"No tick data for {instrument}"
            spread_pips = (tick.ask - tick.bid) / 0.0001
            if spread_pips > max_spread:
                return False, f"Spread {spread_pips:.1f} pips > max {max_spread} pips"
            return True, f"Spread OK: {spread_pips:.1f} pips"
        except ImportError:
            return True, "MT5 not available — spread check skipped"
        except Exception as e:
            return False, f"Spread check error: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def get_current_atr(self, instrument: str, timeframe: str = "M15",
                        period: int = 14) -> Optional[float]:
        """Compute ATR from the last N+1 bars via MT5."""
        import numpy as np
        try:
            import MetaTrader5 as mt5
            tf = getattr(mt5, f"TIMEFRAME_{timeframe}", mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(instrument, tf, 0, period + 10)
            if rates is None or len(rates) < period:
                return None
            import pandas as pd
            df = pd.DataFrame(rates)
            pc = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - pc).abs(),
                (df["low"]  - pc).abs(),
            ], axis=1).max(axis=1)
            return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])
        except Exception as e:
            log.error(f"get_current_atr error: {e}")
            return None
