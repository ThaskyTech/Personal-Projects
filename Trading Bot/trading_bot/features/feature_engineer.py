"""
features/feature_engineer.py
Calculates all H4 and M15 features with rolling-window normalisation
(500 bars) to prevent data leakage.  Blueprint v2.0 Section 6.
"""

from typing import Optional
import numpy as np
import pandas as pd

from config.settings import (
    NORMALIZATION_WINDOW,
    ATR_PERIOD, RSI_PERIOD, EMA_FAST, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K, STOCH_D,
    LONDON_SESSION_START_UTC, LONDON_SESSION_END_UTC,
    NY_SESSION_START_UTC, NY_SESSION_END_UTC,
)
from utils.logger import setup_logger

log = setup_logger("FeatureEngineer")


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast    = _ema(close, fast)
    ema_slow    = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3):
    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=max(window // 4, 10)).mean()
    std = series.rolling(window, min_periods=max(window // 4, 10)).std()
    return (series - mu) / std.replace(0, np.nan)


def _slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    def _ls(arr):
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    return series.rolling(lookback).apply(_ls, raw=True)


def _hh_ll(high: pd.Series, low: pd.Series, lookback: int = 20) -> pd.DataFrame:
    rolling_high = high.rolling(lookback).max()
    rolling_low  = low.rolling(lookback).min()
    hh = (high == rolling_high).astype(float)
    ll = (low  == rolling_low).astype(float)
    return pd.DataFrame({"higher_high": hh, "lower_low": ll})


class FeatureEngineer:

    def __init__(self, norm_window: int = NORMALIZATION_WINDOW):
        self.norm_window = norm_window

    def compute_h4_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        feat = pd.DataFrame(index=df.index)

        ema50  = _ema(close, EMA_FAST)
        ema200 = _ema(close, EMA_SLOW)
        feat["dist_ema50"]  = (close - ema50)  / close
        feat["dist_ema200"] = (close - ema200) / close
        feat["ema_cross"]   = (ema50 - ema200) / close

        feat["rsi14"] = _rsi(close, RSI_PERIOD) / 100.0
        _, _, macd_hist = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        feat["macd_hist"] = macd_hist / close

        atr14 = _atr(high, low, close, ATR_PERIOD)
        feat["atr14_pct"] = atr14 / close

        feat["ema50_slope"]  = _slope(ema50,  lookback=5) / close
        feat["ema200_slope"] = _slope(ema200, lookback=5) / close

        hh_ll = _hh_ll(high, low, lookback=20)
        feat["higher_high"] = hh_ll["higher_high"]
        feat["lower_low"]   = hh_ll["lower_low"]

        for p in [1, 3, 5, 10]:
            feat[f"ret_{p}b"] = close.pct_change(p)

        skip_norm = {"higher_high", "lower_low", "rsi14"}
        for col in feat.columns:
            if col not in skip_norm:
                feat[col] = _rolling_zscore(feat[col], self.norm_window)

        return feat.replace([np.inf, -np.inf], np.nan)

    def compute_m15_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"]
        spread = df["spread"]

        feat = pd.DataFrame(index=df.index)

        feat["rsi14"] = _rsi(close, RSI_PERIOD) / 100.0
        stoch_k, stoch_d = _stochastic(high, low, close, STOCH_K, STOCH_D)
        feat["stoch_k"] = stoch_k / 100.0
        feat["stoch_d"] = stoch_d / 100.0

        atr14 = _atr(high, low, close, ATR_PERIOD)
        feat["atr14_pct"]      = atr14 / close
        feat["vol_expansion"]  = atr14 / atr14.rolling(50).mean().replace(0, np.nan)
        feat["candle_range_pct"] = (high - low) / atr14.replace(0, np.nan)

        typical  = (high + low + close) / 3
        vwap     = (typical * volume).rolling(96).sum() / volume.rolling(96).sum().replace(0, np.nan)
        feat["dist_vwap"]  = (close - vwap)          / close
        feat["dist_ema20"] = (close - _ema(close, 20)) / close
        feat["dist_ema50"] = (close - _ema(close, EMA_FAST)) / close

        for lb in [12, 24, 48]:
            rh = high.rolling(lb).max().shift(1)
            rl = low.rolling(lb).min().shift(1)
            feat[f"break_high_{lb}b"] = ((close > rh) & (close.shift(1) <= rh)).astype(float)
            feat[f"break_low_{lb}b"]  = ((close < rl) & (close.shift(1) >= rl)).astype(float)

        feat["spread_pips"]    = spread
        hour = df.index.hour
        feat["london_session"] = ((hour >= LONDON_SESSION_START_UTC) & (hour < LONDON_SESSION_END_UTC)).astype(float)
        feat["ny_session"]     = ((hour >= NY_SESSION_START_UTC)     & (hour < NY_SESSION_END_UTC)).astype(float)

        for p in [1, 3, 8, 16]:
            feat[f"ret_{p}b"] = close.pct_change(p)

        feat["body_ratio"] = (close - open_) / (high - low).replace(0, np.nan)

        skip_norm = {
            "rsi14", "stoch_k", "stoch_d",
            "london_session", "ny_session", "spread_pips",
            *[c for c in feat.columns if c.startswith("break_")]
        }
        for col in feat.columns:
            if col not in skip_norm:
                feat[col] = _rolling_zscore(feat[col], self.norm_window)

        return feat.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def merge_h4_context_into_m15(m15: pd.DataFrame,
                                   h4_features: pd.DataFrame,
                                   prefix: str = "h4_ctx_") -> pd.DataFrame:
        """
        Forward-merge H4 features into each M15 bar using last-known H4 value.
        Avoids look-ahead bias via merge_asof with direction='backward'.

        Uses fixed internal keys __ts__ and __h4_ts__ so the merge is robust
        to any index name (datetime, time, None, 0, etc).
        """
        # H4 side — prefix all feature columns, pull index into __h4_ts__
        h4_prefixed = h4_features.copy()
        h4_prefixed.columns = [prefix + c for c in h4_features.columns]
        h4_reset = h4_prefixed.reset_index()
        h4_reset = h4_reset.rename(columns={h4_reset.columns[0]: "__h4_ts__"})
        h4_reset = h4_reset.sort_values("__h4_ts__")

        # M15 side — pull index into __ts__
        m15_reset = m15.reset_index()
        m15_reset = m15_reset.rename(columns={m15_reset.columns[0]: "__ts__"})
        m15_reset = m15_reset.sort_values("__ts__")

        # Merge: each M15 bar gets the most recent H4 context (no look-ahead)
        merged = pd.merge_asof(
            m15_reset,
            h4_reset,
            left_on="__ts__",
            right_on="__h4_ts__",
            direction="backward",
        )

        merged = merged.drop(columns=["__h4_ts__"], errors="ignore")
        merged = merged.set_index("__ts__")
        merged.index.name = "datetime"
        return merged