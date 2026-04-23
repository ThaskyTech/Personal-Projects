"""
risk/risk_engine.py
Fully deterministic risk management engine.
ML cannot modify these rules.  Blueprint v2.0 Section 11.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from config.settings import (
    RISK_PER_TRADE_PCT, RISK_PER_TRADE_RECOVERY_PCT,
    MAX_TRADES_PER_PAIR, MAX_EXPOSURE_PER_PAIR_PCT,
    MAX_TOTAL_OPEN_RISK_PCT, MAX_USD_DIRECTIONAL_RISK_PCT,
    DAILY_LOSS_LIMIT_PCT, MONTHLY_LOSS_LIMIT_PCT,
    DRAWDOWN_RECOVERY_TRIGGER,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, INSTRUMENTS,
    SCALING_FREQUENCY_MONTHS,
)
from utils.logger import setup_logger

log = setup_logger("RiskEngine")

# Instruments that share USD directional exposure
USD_INSTRUMENTS = ["EURUSD", "GBPUSD"]


class RiskEngine:
    """
    All risk calculations are deterministic.
    Tracks equity, drawdown state, daily/monthly PnL,
    and enforces all position sizing and exposure rules.
    """

    def __init__(self, initial_equity: float):
        self.initial_equity      = initial_equity
        self.peak_equity         = initial_equity
        self.current_equity      = initial_equity
        self.day_start_equity    = initial_equity
        self.month_start_equity  = initial_equity
        self._last_day           = date.today()
        self._last_month         = date.today().month
        self.in_recovery_mode    = False
        self._pending_lot_update: Optional[float] = None  # v2 fix #12
        self._open_positions: List[Dict] = []             # tracked internally

    # ─────────────────────────────────────────────────────────────────────────
    # Equity update (called after each trade close or MT5 sync)
    # ─────────────────────────────────────────────────────────────────────────

    def update_equity(self, new_equity: float):
        today = date.today()

        # Reset daily tracking
        if today != self._last_day:
            self.day_start_equity = self.current_equity
            self._last_day = today

        # Reset monthly tracking
        if today.month != self._last_month:
            self.month_start_equity = self.current_equity
            self._last_month = today.month

        self.current_equity = new_equity
        self.peak_equity    = max(self.peak_equity, new_equity)

        # Check drawdown recovery
        drawdown = (self.peak_equity - new_equity) / self.peak_equity
        if drawdown >= DRAWDOWN_RECOVERY_TRIGGER:
            if not self.in_recovery_mode:
                log.warning(f"DRAWDOWN RECOVERY MODE ACTIVATED — "
                            f"drawdown={drawdown:.2%}, equity={new_equity:.2f}")
            self.in_recovery_mode = True
        elif self.in_recovery_mode and new_equity >= self.peak_equity * 0.99:
            log.info("Drawdown recovery complete — resuming normal risk sizing")
            self.in_recovery_mode = False
            # Apply pending lot size update if one exists (fix #12)
            if self._pending_lot_update is not None:
                log.info(f"Applying pending lot size base: {self._pending_lot_update}")
                self._pending_lot_update = None

    # ─────────────────────────────────────────────────────────────────────────
    # Trade gate — call before opening any position
    # ─────────────────────────────────────────────────────────────────────────

    def can_open_trade(self,
                       instrument: str,
                       direction: str,
                       proposed_risk_pct: float,
                       open_positions: List[Dict],
                       ) -> Tuple[bool, str]:
        """
        Check all risk rules before allowing a new trade.

        Returns (allowed: bool, reason: str)
        """

        # ── Daily loss limit ──────────────────────────────────────────────────
        daily_loss_pct = (self.day_start_equity - self.current_equity) / self.day_start_equity
        if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT:
            return False, f"Daily loss limit reached ({daily_loss_pct:.2%})"

        # ── Monthly loss limit ────────────────────────────────────────────────
        monthly_loss_pct = (self.month_start_equity - self.current_equity) / self.month_start_equity
        if monthly_loss_pct >= MONTHLY_LOSS_LIMIT_PCT:
            return False, f"Monthly loss limit reached ({monthly_loss_pct:.2%})"

        # ── Max trades per pair ───────────────────────────────────────────────
        pair_trades = [p for p in open_positions if p["instrument"] == instrument]
        if len(pair_trades) >= MAX_TRADES_PER_PAIR:
            return False, f"Max trades per pair ({MAX_TRADES_PER_PAIR}) reached for {instrument}"

        # ── Max exposure per pair ─────────────────────────────────────────────
        pair_risk_pct = sum(p.get("risk_pct", 0) for p in pair_trades)
        if pair_risk_pct + proposed_risk_pct > MAX_EXPOSURE_PER_PAIR_PCT:
            return False, (f"Pair exposure cap: current={pair_risk_pct:.2%} "
                           f"+ new={proposed_risk_pct:.2%} > {MAX_EXPOSURE_PER_PAIR_PCT:.2%}")

        # ── Total portfolio open risk cap (NEW v2.0) ──────────────────────────
        total_risk_pct = sum(p.get("risk_pct", 0) for p in open_positions)
        if total_risk_pct + proposed_risk_pct > MAX_TOTAL_OPEN_RISK_PCT:
            return False, (f"Portfolio cap: current={total_risk_pct:.2%} "
                           f"+ new={proposed_risk_pct:.2%} > {MAX_TOTAL_OPEN_RISK_PCT:.2%}")

        # ── USD directional correlation cap (NEW v2.0) ────────────────────────
        usd_blocked, usd_reason = self._check_usd_correlation(
            instrument, direction, proposed_risk_pct, open_positions)
        if usd_blocked:
            return False, usd_reason

        return True, "OK"

    def _check_usd_correlation(self,
                                instrument: str,
                                direction: str,
                                proposed_risk_pct: float,
                                open_positions: List[Dict],
                                ) -> Tuple[bool, str]:
        """
        Both EURUSD and GBPUSD are USD pairs.
        If both are being traded in the same USD direction, cap combined risk.
        EURUSD long = short USD; GBPUSD long = short USD.
        EURUSD short = long USD; GBPUSD short = long USD.
        """
        if instrument not in USD_INSTRUMENTS:
            return False, ""

        # USD direction from the trade direction
        usd_direction = "short_usd" if direction == "long" else "long_usd"

        # Sum existing USD directional exposure
        existing_usd_risk = 0.0
        for pos in open_positions:
            if pos.get("instrument") in USD_INSTRUMENTS:
                pos_usd_dir = "short_usd" if pos.get("direction") == "long" else "long_usd"
                if pos_usd_dir == usd_direction:
                    existing_usd_risk += pos.get("risk_pct", 0)

        combined = existing_usd_risk + proposed_risk_pct
        if combined > MAX_USD_DIRECTIONAL_RISK_PCT:
            return True, (f"USD directional cap: combined {combined:.2%} "
                          f"> {MAX_USD_DIRECTIONAL_RISK_PCT:.2%}")
        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_lot_size(self,
                           instrument: str,
                           entry_price: float,
                           sl_price: float,
                           pip_value_per_lot: float = 10.0,
                           ) -> Tuple[float, float]:
        """
        Calculate lot size to risk exactly RISK_PER_TRADE_PCT of equity.

        pip_value_per_lot: USD value of 1 pip for 1 standard lot.
            EURUSD/GBPUSD standard: $10/pip for 1 lot on a USD account.

        Returns (lot_size, risk_amount_usd)
        """
        risk_pct = RISK_PER_TRADE_RECOVERY_PCT if self.in_recovery_mode else RISK_PER_TRADE_PCT
        risk_usd = self.current_equity * risk_pct

        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            log.error("SL distance is 0 — cannot calculate lot size")
            return 0.0, 0.0

        # Convert price distance to pips (EURUSD/GBPUSD: 4dp pairs, 1 pip = 0.0001)
        pip_size    = 0.0001
        sl_pips     = sl_distance / pip_size
        lot_size    = risk_usd / (sl_pips * pip_value_per_lot)

        # Round to 2 decimal places (standard MT5 minimum step)
        lot_size = max(0.01, round(lot_size, 2))

        actual_risk_usd = lot_size * sl_pips * pip_value_per_lot
        actual_risk_pct = actual_risk_usd / self.current_equity

        log.debug(f"{instrument} lot_size={lot_size} | sl_pips={sl_pips:.1f} | "
                  f"risk={actual_risk_usd:.2f} USD ({actual_risk_pct:.2%})")

        return lot_size, actual_risk_usd

    def get_risk_pct(self) -> float:
        """Current risk % per trade (normal or recovery)."""
        return RISK_PER_TRADE_RECOVERY_PCT if self.in_recovery_mode else RISK_PER_TRADE_PCT

    # ─────────────────────────────────────────────────────────────────────────
    # SL / TP calculation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_sl_tp(entry_price: float,
                        atr: float,
                        direction: str,
                        ) -> Tuple[float, float]:
        """
        Fixed ATR-based SL and TP.  Blueprint v2.0 Section 10.
        SL = 1.5 × ATR(14)
        TP = 2.5 × ATR(14)
        """
        sl_dist = SL_ATR_MULTIPLIER * atr
        tp_dist = TP_ATR_MULTIPLIER * atr

        if direction == "long":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        return round(sl, 5), round(tp, 5)

    # ─────────────────────────────────────────────────────────────────────────
    # Quarterly lot-size scaling (Blueprint v2.0 Section 18)
    # ─────────────────────────────────────────────────────────────────────────

    def quarterly_scaling_event(self):
        """
        Called at start of each quarter.
        If in recovery mode: store pending update, do NOT apply yet.
        If normal: update is implicitly applied via current_equity in lot calc.
        """
        if self.in_recovery_mode:
            self._pending_lot_update = self.current_equity
            log.info(f"Quarterly scaling calculated but DEFERRED "
                     f"(recovery mode active). Equity={self.current_equity:.2f}")
        else:
            log.info(f"Quarterly scaling applied. Equity={self.current_equity:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Status summary
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        daily_loss = (self.day_start_equity - self.current_equity) / self.day_start_equity
        monthly_loss = (self.month_start_equity - self.current_equity) / self.month_start_equity
        return {
            "current_equity":    round(self.current_equity, 2),
            "peak_equity":       round(self.peak_equity, 2),
            "drawdown_pct":      round(drawdown, 4),
            "daily_loss_pct":    round(daily_loss, 4),
            "monthly_loss_pct":  round(monthly_loss, 4),
            "in_recovery_mode":  self.in_recovery_mode,
            "risk_per_trade_pct": self.get_risk_pct(),
        }
