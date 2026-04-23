"""
regime/regime_classifier.py
Detects market regime: trending | ranging | high_vol | low_vol
Outputs a probability vector used to condition entry signals.
Blueprint v2.0 Section 7.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from config.settings import ATR_PERIOD, REGIME_LABELS
from utils.logger import setup_logger

log = setup_logger("RegimeClassifier")


class RegimeClassifier:
    """
    Rule-based + soft probability regime classifier.
    Returns a 4-element probability vector summing to ~1.0:
        [trending, ranging, high_vol, low_vol]

    Design: deterministic rules produce a score vector which is then
    softmax-normalised to a probability distribution.
    This avoids any look-ahead during live trading.
    """

    def __init__(self,
                 atr_period: int = ATR_PERIOD,
                 adx_period: int = 14,
                 vol_lookback: int = 50):
        self.atr_period   = atr_period
        self.adx_period   = adx_period
        self.vol_lookback = vol_lookback

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify every bar in df.
        Returns DataFrame with columns: trending, ranging, high_vol, low_vol
        """
        adx   = self._adx(df)
        atr   = self._atr(df)
        atr_z = self._atr_zscore(atr)

        results = []
        for i in range(len(df)):
            vec = self._score_at(adx.iloc[i], atr_z.iloc[i])
            results.append(vec)

        out = pd.DataFrame(results, index=df.index, columns=REGIME_LABELS)
        return out

    def classify_latest(self, df: pd.DataFrame) -> Dict[str, float]:
        """Classify only the most recent bar.  Used in live trading."""
        adx   = self._adx(df)
        atr   = self._atr(df)
        atr_z = self._atr_zscore(atr)
        vec = self._score_at(adx.iloc[-1], atr_z.iloc[-1])
        return dict(zip(REGIME_LABELS, vec))

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring logic
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_at(adx_val: float, atr_z: float) -> np.ndarray:
        """
        Convert ADX and ATR-z-score into a soft probability vector.
        ADX > 25 → trending; ADX < 20 → ranging.
        ATR z-score > 1.0 → high_vol; < -0.5 → low_vol.
        """
        scores = np.zeros(4)   # [trending, ranging, high_vol, low_vol]

        if np.isnan(adx_val) or np.isnan(atr_z):
            scores[:] = 0.25   # uniform when insufficient data
            return scores

        # Trending score
        if adx_val >= 30:
            scores[0] = 1.0
        elif adx_val >= 20:
            scores[0] = (adx_val - 20) / 10.0    # linear ramp 20→30

        # Ranging score
        if adx_val <= 20:
            scores[1] = 1.0
        elif adx_val <= 30:
            scores[1] = (30 - adx_val) / 10.0    # inverse ramp

        # High vol score
        if atr_z >= 1.5:
            scores[2] = 1.0
        elif atr_z >= 0.5:
            scores[2] = (atr_z - 0.5) / 1.0

        # Low vol score
        if atr_z <= -0.5:
            scores[3] = 1.0
        elif atr_z <= 0.5:
            scores[3] = max(0.0, (0.5 - atr_z) / 1.0)

        # Softmax normalisation
        exp_s = np.exp(scores - scores.max())
        return exp_s / exp_s.sum()

    # ─────────────────────────────────────────────────────────────────────────
    # Indicator helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=self.atr_period - 1, adjust=False).mean()

    def _atr_zscore(self, atr: pd.Series) -> pd.Series:
        mu  = atr.rolling(self.vol_lookback).mean()
        std = atr.rolling(self.vol_lookback).std()
        return (atr - mu) / std.replace(0, np.nan)

    def _adx(self, df: pd.DataFrame) -> pd.Series:
        """Wilder's ADX."""
        p  = self.adx_period
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        up_move   = high.diff()
        down_move = (-low.diff())

        plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr14    = tr.ewm(com=p - 1, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(com=p - 1, adjust=False).mean()  / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(com=p - 1, adjust=False).mean() / atr14.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=p - 1, adjust=False).mean()
        return adx
