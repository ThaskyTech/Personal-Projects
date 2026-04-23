#!/usr/bin/env python3
"""
EURUSD 1-Minute CSV Resampler
Converts a 1-minute OHLCV CSV to 15-minute and 4-hour timeframes.

Supported input formats:
  - Separate date & time columns:  Date, Time, Open, High, Low, Close, Volume
  - Combined datetime column:       DateTime, Open, High, Low, Close, Volume
  - MetaTrader / Dukascopy style formats are auto-detected

Usage:
  python resample_forex.py --input EURUSD_1m.csv
  python resample_forex.py --input EURUSD_1m.csv --output-dir ./output
  python resample_forex.py --input EURUSD_1m.csv --delimiter ";" --datetime-col "Gmt time"
"""

import argparse
import os
import sys
import pandas as pd


# ── Column name aliases (case-insensitive) ───────────────────────────────────
ALIASES = {
    "open":   ["open", "o"],
    "high":   ["high", "h"],
    "low":    ["low",  "l"],
    "close":  ["close","c"],
    "volume": ["volume","vol","v","tickvol","tick volume"],
    "date":   ["date","<date>"],
    "time":   ["time","<time>"],
    "datetime": ["datetime","<datetime>","timestamp","gmt time","time (utc)","date/time"],
}

TIMEFRAMES = {
    "15min": ("15min", "15T"),
    "4h":    ("4h",    "4h"),   # pandas ≥ 2.2 uses "4h"; older uses "4H"
}


def find_col(columns: list[str], key: str) -> str | None:
    """Return the first column matching any alias for `key`, else None."""
    lower = {c.lower().strip(): c for c in columns}
    for alias in ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def load_csv(path: str, delimiter: str, datetime_col_hint: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=delimiter, engine="python", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # ── Resolve datetime ──────────────────────────────────────────────────────
    dt_col = None
    if datetime_col_hint:
        # User-supplied column name
        matches = [c for c in df.columns if c.lower() == datetime_col_hint.lower()]
        if matches:
            dt_col = matches[0]
        else:
            sys.exit(f"[ERROR] --datetime-col '{datetime_col_hint}' not found in CSV. "
                     f"Columns: {list(df.columns)}")
    else:
        dt_col = find_col(df.columns, "datetime")

    if dt_col:
        df[dt_col] = df[dt_col].astype(str).str.strip().str.strip('"').str.strip("'")
        df["_dt"] = pd.to_datetime(df[dt_col], utc=False)
    else:
        date_col = find_col(df.columns, "date")
        time_col = find_col(df.columns, "time")
        if not date_col or not time_col:
            sys.exit(
                "[ERROR] Could not find datetime columns.\n"
                f"  Detected columns: {list(df.columns)}\n"
                "  Use --datetime-col to specify the datetime column name, or\n"
                "  ensure your CSV has Date/Time or DateTime columns."
            )
        df["_dt"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))

    df = df.set_index("_dt").sort_index()

    # ── Resolve OHLCV ─────────────────────────────────────────────────────────
    mapping = {}
    for key in ("open", "high", "low", "close", "volume"):
        col = find_col(df.columns, key)
        if col:
            mapping[col] = key
        elif key != "volume":
            sys.exit(f"[ERROR] Could not find '{key}' column. Columns: {list(df.columns)}")

    df = df.rename(columns=mapping)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].apply(pd.to_numeric, errors="coerce")


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg: dict = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    # Tick volume = count of 1-minute bars that formed each candle
    df = df.copy()
    df["_tick"] = 1
    agg["_tick"] = "sum"

    # Try the rule as-is; fall back to uppercase variant for older pandas
    try:
        resampled = df.resample(rule).agg(agg).dropna(subset=["open"])
    except ValueError:
        resampled = df.resample(rule.upper()).agg(agg).dropna(subset=["open"])

    resampled = resampled.rename(columns={"_tick": "tick_volume"})
    resampled["tick_volume"] = resampled["tick_volume"].astype(int)

    return resampled


def save(df: pd.DataFrame, path: str) -> None:
    df.index.name = "datetime"
    df.to_csv(path, float_format="%.6f")
    rows = len(df)
    print(f"  ✓  Saved {rows:,} bars  →  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample a 1-minute forex CSV to 15-min and 4-hour timeframes."
    )
    parser.add_argument("--input",        required=True,  help="Path to the 1-minute CSV file")
    parser.add_argument("--output-dir",   default=None,   help="Output directory (default: same as input)")
    parser.add_argument("--delimiter",    default=",",    help="CSV delimiter (default: ',')")
    parser.add_argument("--datetime-col", default=None,   help="Name of the combined datetime column if auto-detect fails")
    parser.add_argument("--no-4h",        action="store_true", help="Skip 4-hour output")
    parser.add_argument("--no-15m",       action="store_true", help="Skip 15-minute output")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.input))[0]

    print(f"\n{'='*55}")
    print(f"  Forex CSV Resampler")
    print(f"{'='*55}")
    print(f"  Input : {args.input}")
    print(f"  Output: {out_dir}\n")

    print("  Loading CSV …", end=" ", flush=True)
    df = load_csv(args.input, args.delimiter, args.datetime_col)
    print(f"{len(df):,} 1-minute bars loaded.")
    print(f"  Date range: {df.index[0]}  →  {df.index[-1]}\n")

    if not args.no_15m:
        print("  Resampling to 15-minute …")
        df_15 = resample(df, "15min")
        save(df_15, os.path.join(out_dir, f"{base}_15min.csv"))

    if not args.no_4h:
        print("  Resampling to 4-hour …")
        df_4h = resample(df, "4h")
        save(df_4h, os.path.join(out_dir, f"{base}_4h.csv"))

    print(f"\n  Done!\n{'='*55}\n")


if __name__ == "__main__":
    main()