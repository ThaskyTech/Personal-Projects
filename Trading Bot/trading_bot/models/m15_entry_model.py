"""
models/m15_entry_model.py
M15 Entry Signal Model — primary trading decision engine.
Outputs entry probability, expected return, and expected downside.
Blueprint v2.0 Section 9.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from config.settings import (
    M15_BASE_ENTRY_THRESHOLD, M15_MIN_RR_RATIO,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
)
from utils.logger import setup_logger

log = setup_logger("M15EntryModel")

MODEL_DIR = Path("models/saved")


class M15EntryModel:
    """
    Multi-output model for M15 entry timing:
      - entry_probability  : probability that a trade at this bar is profitable
      - expected_return    : expected gain as multiple of ATR
      - expected_downside  : expected loss as multiple of ATR

    Label construction (for training):
      - A bar is a 'long signal' if price rises >= TP_ATR within SL_ATR loss limit
      - A bar is a 'short signal' if price falls >= TP_ATR within SL_ATR loss limit
    """

    MODEL_FILENAME = "m15_entry_{instrument}_{direction}.pkl"
    FORWARD_BARS   = 192    # M15 bars to evaluate outcome (~16 hours)

    def __init__(self, instrument: str):
        self.instrument  = instrument
        self.long_model  = None
        self.short_model = None
        self.feature_cols: List[str] = []
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Label construction
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_labels(df_ohlcv: pd.DataFrame,
                     atr_series: pd.Series,
                     sl_mult: float = SL_ATR_MULTIPLIER,
                     tp_mult: float = TP_ATR_MULTIPLIER,
                     forward_bars: int = 64,
                     ) -> pd.DataFrame:
        close = df_ohlcv["close"].values
        high = df_ohlcv["high"].values
        low = df_ohlcv["low"].values
        atr = atr_series.values

        n = len(close)
        long_label = np.zeros(n, dtype=float)
        short_label = np.zeros(n, dtype=float)

        for i in range(n - forward_bars):
            c = close[i]
            a = atr[i]
            if np.isnan(a) or a == 0:
                continue

            sl_long = c - sl_mult * a
            tp_long = c + tp_mult * a
            sl_short = c + sl_mult * a
            tp_short = c - tp_mult * a

            long_resolved = False
            short_resolved = False

            for j in range(i + 1, min(i + 1 + forward_bars, n)):
                # Evaluate long independently
                if not long_resolved:
                    if low[j] <= sl_long:
                        long_label[i] = 0.0
                        long_resolved = True
                    elif high[j] >= tp_long:
                        long_label[i] = 1.0
                        long_resolved = True

                # Evaluate short independently
                if not short_resolved:
                    if high[j] >= sl_short:
                        short_label[i] = 0.0
                        short_resolved = True
                    elif low[j] <= tp_short:
                        short_label[i] = 1.0
                        short_resolved = True

                if long_resolved and short_resolved:
                    break

        idx = df_ohlcv.index
        return pd.DataFrame({
            "long_label": long_label,
            "short_label": short_label,
        }, index=idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, labels: pd.DataFrame,
              hyperparams: Optional[Dict] = None):
        """
        Train separate long and short entry classifiers.
        X        : combined M15 feature matrix (with H4 context merged in)
        labels   : DataFrame with 'long_label' and 'short_label' columns
        """
        self.feature_cols = X.columns.tolist()
        X_clean = X.fillna(0)

        aligned = X_clean.join(labels, how="inner").dropna()
        X_al    = aligned[self.feature_cols]

        self.long_model  = self._make_model(hyperparams)
        self.short_model = self._make_model(hyperparams)

        y_long  = aligned["long_label"]
        y_short = aligned["short_label"]

        self.long_model.fit(X_al,  y_long)
        self.short_model.fit(X_al, y_short)

        long_rate  = y_long.mean()
        short_rate = y_short.mean()
        log.info(f"M15EntryModel trained for {self.instrument} | "
                 f"long hit rate: {long_rate:.2%} | short hit rate: {short_rate:.2%} | "
                 f"samples: {len(X_al)} | features: {len(self.feature_cols)}")

    @staticmethod
    def _make_model(hyperparams: Optional[Dict] = None):
        try:
            import lightgbm as lgb
            params = {
                "objective":         "binary",
                "metric":            "binary_logloss",
                "n_estimators":      400,
                "learning_rate":     0.03,
                "max_depth":         6,
                "num_leaves":        40,
                "min_child_samples": 40,
                "subsample":         0.8,
                "colsample_bytree":  0.8,
                "reg_alpha":         0.1,
                "reg_lambda":        0.2,
                "n_jobs":            -1,
                "verbose":           -1,
                "random_state":      42,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            params = {
                "n_estimators":  300,
                "learning_rate": 0.03,
                "max_depth":     5,
                "subsample":     0.8,
                "random_state":  42,
            }
            if hyperparams:
                params.update(hyperparams)
            return GradientBoostingClassifier(**params)

        if hyperparams:
            params.update(hyperparams)
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=5,
                subsample=0.8, random_state=42
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self,
                features_row: pd.Series,
                direction: str,
                atr: float,
                h4_bias: Dict,
                regime: Dict,
                ) -> Dict:
        """
        Generate a full signal assessment for a single M15 bar.

        Returns dict with:
            entry_probability, expected_return, expected_downside,
            rr_ratio, signal_valid, threshold_used, direction
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction}")

        model = self.long_model if direction == "long" else self.short_model
        if model is None:
            raise RuntimeError("Model not trained or loaded.")

        x = pd.DataFrame(
            [features_row[self.feature_cols].fillna(0).values],
            columns=self.feature_cols
        )
        p_entry = float(model.predict_proba(x)[0][1])

        # Apply H4 neutral-zone penalty to threshold
        threshold = M15_BASE_ENTRY_THRESHOLD + h4_bias.get("threshold_penalty", 0.0)

        # Expected return/downside as ATR multiples (fixed from exit strategy)
        exp_return   = TP_ATR_MULTIPLIER * atr
        exp_downside = SL_ATR_MULTIPLIER * atr
        rr_ratio     = exp_return / exp_downside if exp_downside > 0 else 0.0

        signal_valid = (
            p_entry >= threshold and
            rr_ratio >= M15_MIN_RR_RATIO and
            self._regime_compatible(regime, direction)
        )

        return {
            "direction":         direction,
            "entry_probability": round(p_entry, 4),
            "expected_return":   round(exp_return, 6),
            "expected_downside": round(exp_downside, 6),
            "rr_ratio":          round(rr_ratio, 3),
            "threshold_used":    round(threshold, 4),
            "signal_valid":      signal_valid,
        }

    @staticmethod
    def _regime_compatible(regime: Dict, direction: str) -> bool:
        """
        Regime compatibility rules:
        - High volatility regime: avoid new entries (risk of spiking through SL)
        - Low volatility: fine for both
        - Trending: both directions OK (H4 bias handles direction filter)
        - Ranging: both OK at reduced confidence
        """
        if regime.get("high_vol", 0) > 0.60:
            return False    # suppress in high-vol spike regimes
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self):
        for direction, model in [("long", self.long_model), ("short", self.short_model)]:
            path = MODEL_DIR / self.MODEL_FILENAME.format(
                instrument=self.instrument, direction=direction)
            joblib.dump({"model": model, "feature_cols": self.feature_cols}, path)
        log.info(f"M15EntryModel saved for {self.instrument}")

    def load(self) -> bool:
        ok = True
        for direction, attr in [("long", "long_model"), ("short", "short_model")]:
            path = MODEL_DIR / self.MODEL_FILENAME.format(
                instrument=self.instrument, direction=direction)
            if not path.exists():
                log.warning(f"No saved M15EntryModel ({direction}) at {path}")
                ok = False
                continue
            data = joblib.load(path)
            setattr(self, attr, data["model"])
            self.feature_cols = data["feature_cols"]
        if ok:
            log.info(f"M15EntryModel loaded for {self.instrument}")
        return ok
