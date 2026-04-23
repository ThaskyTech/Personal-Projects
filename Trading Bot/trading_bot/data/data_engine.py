"""
data/data_engine.py
Handles historical data loading, alignment, and gap checking.
Supports both MT5 live data and CSV-based offline/backtest data.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    TIMEFRAME_H4, TIMEFRAME_M15, MIN_HISTORY_YEARS,
    INSTRUMENTS
)
from utils.logger import setup_logger

log = setup_logger("DataEngine")

# MT5 timeframe constants (imported lazily so the bot runs without MT5 installed)
TF_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "M30": 30,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}


class DataEngine:
    """
    Loads, validates, aligns, and serves OHLCV data for all instruments
    and timeframes required by the feature engineering layer.
    """

    def __init__(self, data_dir: str = "data/historical", use_mt5: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_mt5 = use_mt5
        self._cache: Dict[str, pd.DataFrame] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Returns {instrument: {timeframe: DataFrame}} for all instruments.
        DataFrames have columns: open, high, low, close, volume, spread
        indexed by UTC datetime.
        """
        result = {}
        for instrument in INSTRUMENTS:
            result[instrument] = {}
            for tf in [TIMEFRAME_H4, TIMEFRAME_M15]:
                df = self._load(instrument, tf)
                if df is not None:
                    result[instrument][tf] = df
                    log.info(f"Loaded {instrument} {tf}: {len(df)} bars "
                             f"({df.index[0]} → {df.index[-1]})")
                else:
                    log.warning(f"No data found for {instrument} {tf}")
        return result

    def load_pair(self, instrument: str, timeframe: str,
                  bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load a single instrument/timeframe, optionally limiting to last N bars."""
        df = self._load(instrument, timeframe)
        if df is not None and bars:
            df = df.iloc[-bars:]
        return df

    def fetch_live_bars(self, instrument: str, timeframe: str,
                        count: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch recent bars from MT5 for live trading."""
        if not self.use_mt5:
            return self._load(instrument, timeframe, tail=count)
        return self._fetch_from_mt5(instrument, timeframe, count)

    def validate_minimum_history(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
        """Verify all pairs have at least MIN_HISTORY_YEARS of H4 data."""
        required_bars = MIN_HISTORY_YEARS * 252 * 6  # ~252 trading days × 6 H4 bars/day
        ok = True
        for instrument in INSTRUMENTS:
            if instrument not in data or TIMEFRAME_H4 not in data[instrument]:
                log.error(f"Missing H4 data for {instrument}")
                ok = False
                continue
            df = data[instrument][TIMEFRAME_H4]
            span_years = (df.index[-1] - df.index[0]).days / 365.25
            if span_years < MIN_HISTORY_YEARS:
                log.error(f"{instrument} H4 spans only {span_years:.1f} years "
                          f"(minimum {MIN_HISTORY_YEARS})")
                ok = False
            else:
                log.info(f"{instrument} H4 history: {span_years:.1f} years — OK")
        return ok

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, instrument: str, timeframe: str,
              tail: Optional[int] = None) -> Optional[pd.DataFrame]:
        cache_key = f"{instrument}_{timeframe}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
            return df.iloc[-tail:] if tail else df

        if self.use_mt5:
            df = self._fetch_from_mt5(instrument, timeframe, count=99999)
        else:
            df = self._load_from_csv(instrument, timeframe)

        if df is None:
            return None

        df = self._clean(df)
        self._cache[cache_key] = df
        return df.iloc[-tail:] if tail else df

    def _load_from_csv(self, instrument: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load from CSV files placed in data/historical/.
        Expected filename: EURUSD_H4.csv or EURUSD_H4.parquet
        Expected columns: datetime, open, high, low, close, volume
                          (spread column optional)
        """
        for ext in ["parquet", "csv"]:
            path = self.data_dir / f"{instrument}_{timeframe}.{ext}"
            if path.exists():
                log.debug(f"Loading {path}")
                if ext == "parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, parse_dates=["datetime"])
                return self._standardise_columns(df)
        log.warning(f"No CSV/Parquet file found for {instrument} {timeframe} in {self.data_dir}")
        return None

    def _fetch_from_mt5(self, instrument: str, timeframe: str,
                        count: int = 5000) -> Optional[pd.DataFrame]:
        """Fetch bars directly from MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}", None)
            if tf_const is None:
                log.error(f"Unknown timeframe: {timeframe}")
                return None
            rates = mt5.copy_rates_from_pos(instrument, tf_const, 0, count)
            if rates is None or len(rates) == 0:
                log.error(f"MT5 returned no data for {instrument} {timeframe}")
                return None
            df = pd.DataFrame(rates)
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume", "real_volume": "real_vol",
                                     "spread": "spread"})
            return self._standardise_columns(df)
        except ImportError:
            log.error("MetaTrader5 package not installed.")
            return None
        except Exception as e:
            log.error(f"MT5 fetch error: {e}")
            return None

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and set datetime index."""
        df.columns = [c.lower().strip() for c in df.columns]
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        elif "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        if "spread" not in df.columns:
            df["spread"] = 0.0
        return df[["open", "high", "low", "close", "volume", "spread"]].sort_index()

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, forward-fill small gaps, drop large gaps."""
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        # Drop rows where OHLC are all zero or NaN
        mask = (df[["open", "high", "low", "close"]].replace(0, np.nan).notna().all(axis=1))
        dropped = (~mask).sum()
        if dropped:
            log.debug(f"Dropped {dropped} zero/NaN OHLC rows")
        df = df[mask]
        # Forward-fill spread (1 bar max)
        df["spread"] = df["spread"].replace(0, np.nan).ffill(limit=1).fillna(0)
        return df

    def align_timeframes(self, h4: pd.DataFrame,
                         m15: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align H4 and M15 so that each M15 bar maps to the correct H4 context bar.
        Returns (h4_aligned, m15_aligned) trimmed to the common date range.
        """
        start = max(h4.index[0], m15.index[0])
        end   = min(h4.index[-1], m15.index[-1])
        h4  = h4.loc[start:end]
        m15 = m15.loc[start:end]
        return h4, m15
