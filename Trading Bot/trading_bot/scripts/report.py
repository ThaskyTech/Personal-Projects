"""
scripts/report.py
Generates a performance summary report from the trade database.

Usage:
    python scripts/report.py [--days 30]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import pandas as pd

from utils.logger import setup_logger, TradeDatabase

log = setup_logger("Report")


def generate_report(days: int = 30):
    db_path = os.getenv("DB_PATH", "logs/trades.db")
    if not Path(db_path).exists():
        print("No trade database found. Run the bot first.")
        return

    conn  = sqlite3.connect(db_path)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df    = pd.read_sql(
        f"SELECT * FROM trades WHERE entry_time >= '{since}' AND exit_time IS NOT NULL",
        conn
    )
    conn.close()

    if df.empty:
        print(f"No closed trades in the last {days} days.")
        return

    print(f"\n{'='*55}")
    print(f"  PERFORMANCE REPORT — Last {days} days")
    print(f"{'='*55}")
    print(f"  Period:      {since[:10]} → {datetime.utcnow().date()}")
    print(f"  Instruments: {', '.join(df['instrument'].unique())}")
    print(f"\n  TRADE SUMMARY")
    print(f"  Total trades:    {len(df)}")
    print(f"  Winning trades:  {(df['pnl'] > 0).sum()}")
    print(f"  Losing trades:   {(df['pnl'] <= 0).sum()}")
    print(f"  Win rate:        {(df['pnl'] > 0).mean():.1%}")

    print(f"\n  P&L")
    print(f"  Total PnL:       ${df['pnl'].sum():.2f}")
    print(f"  Avg win:         ${df.loc[df['pnl']>0, 'pnl'].mean():.2f}" if (df['pnl'] > 0).any() else "  Avg win:         n/a")
    print(f"  Avg loss:        ${df.loc[df['pnl']<=0, 'pnl'].mean():.2f}" if (df['pnl'] <= 0).any() else "  Avg loss:        n/a")

    # Per instrument breakdown
    print(f"\n  BY INSTRUMENT")
    for inst in df["instrument"].unique():
        sub = df[df["instrument"] == inst]
        wr  = (sub["pnl"] > 0).mean()
        pnl = sub["pnl"].sum()
        print(f"  {inst}: {len(sub)} trades | WR={wr:.1%} | PnL=${pnl:.2f}")

    # Exit reason breakdown
    if "exit_reason" in df.columns:
        print(f"\n  EXIT REASONS")
        for reason, count in df["exit_reason"].value_counts().items():
            print(f"  {reason}: {count}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    generate_report(args.days)
