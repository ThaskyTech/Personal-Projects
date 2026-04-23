"""
scripts/qmrp.py
Runs the Quarterly Model Refresh Process (QMRP) for all instruments.
This is the script to schedule every 3 months via cron/Task Scheduler.

Usage:
    python scripts/qmrp.py [--instrument EURUSD]

Steps executed:
  1. Append latest market data (fetched from MT5 if connected)
  2. Recalculate features
  3. Run walk-forward optimisation
  4. Compare against production model
  5. Deploy if criteria met
  6. Export trades CSV
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import json
from datetime import datetime

from config.settings import INSTRUMENTS, TIMEFRAME_H4, TIMEFRAME_M15
from data.data_engine import DataEngine
from backtest.walk_forward import WalkForwardOptimiser
from utils.logger import setup_logger, TradeDatabase

log = setup_logger("QMRP")
PRODUCTION_SORTINO_FILE = Path("models/saved/production_sortino.json")


def load_production_sortino(instrument: str) -> float:
    if PRODUCTION_SORTINO_FILE.exists():
        with open(PRODUCTION_SORTINO_FILE) as f:
            data = json.load(f)
        return data.get(instrument, 0.0)
    return 0.0


def save_production_sortino(instrument: str, sortino: float):
    data = {}
    if PRODUCTION_SORTINO_FILE.exists():
        with open(PRODUCTION_SORTINO_FILE) as f:
            data = json.load(f)
    data[instrument] = sortino
    PRODUCTION_SORTINO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRODUCTION_SORTINO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run QMRP for all instruments")
    parser.add_argument("--instrument", type=str, default=None)
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else INSTRUMENTS
    data_engine = DataEngine(use_mt5=False)
    db          = TradeDatabase(os.getenv("DB_PATH", "logs/trades.db"))

    log.info(f"QMRP started at {datetime.utcnow().isoformat()} UTC")
    log.info(f"Instruments: {instruments}")

    # Export trades CSV first (step 6 of QMRP)
    csv_path = db.export_trades_csv()
    log.info(f"Trade data exported → {csv_path}")

    for instrument in instruments:
        log.info(f"\nRunning QMRP for {instrument}...")

        h4_df  = data_engine.load_pair(instrument, TIMEFRAME_H4)
        m15_df = data_engine.load_pair(instrument, TIMEFRAME_M15)

        if h4_df is None or m15_df is None:
            log.error(f"Cannot run QMRP for {instrument}: missing data")
            continue

        prod_sortino = load_production_sortino(instrument)
        log.info(f"Production Sortino for {instrument}: {prod_sortino:.4f}")

        wfo    = WalkForwardOptimiser(instrument)
        result = wfo.run_qmrp(h4_df, m15_df, production_sortino=prod_sortino)

        log.info(f"QMRP result for {instrument}:")
        log.info(f"  Deploy new model: {result.deploy_new}")
        log.info(f"  Reason: {result.deploy_reason}")
        log.info(f"  New Sortino: {result.new_sortino:.4f}")

        if result.deploy_new:
            # Re-train on full data with new model
            from scripts.train import train_instrument
            ok = train_instrument(instrument, data_engine, skip_wfo=True)
            if ok:
                save_production_sortino(instrument, result.new_sortino)
                log.info(f"New model deployed for {instrument}. "
                         f"Production Sortino updated to {result.new_sortino:.4f}")
        else:
            log.info(f"Production model retained for {instrument}")

    log.info(f"\nQMRP complete at {datetime.utcnow().isoformat()} UTC")


if __name__ == "__main__":
    main()
