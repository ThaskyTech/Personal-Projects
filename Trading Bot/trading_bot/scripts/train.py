"""
scripts/train.py
Full training pipeline — runs QMRP for all instruments and saves models.

Usage:
    python scripts/train.py [--instrument EURUSD] [--skip-wfo]

    --instrument   : train only one instrument (default: all)
    --skip-wfo     : skip walk-forward optimisation, train directly on full data
                     (useful for first-time setup / quick iteration)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import joblib
from datetime import datetime

from config.settings import INSTRUMENTS, TIMEFRAME_H4, TIMEFRAME_M15
from data.data_engine import DataEngine
from features.feature_engineer import FeatureEngineer, _atr
from models.h4_bias_model import H4BiasModel
from models.m15_entry_model import M15EntryModel
from backtest.walk_forward import WalkForwardOptimiser
from utils.logger import setup_logger, TradeDatabase

log = setup_logger("TrainScript")


def train_instrument(instrument: str, data_engine: DataEngine,
                     skip_wfo: bool = False, production_sortino: float = 0.0):
    log.info(f"\n{'='*50}")
    log.info(f"  TRAINING: {instrument}")
    log.info(f"{'='*50}")

    # Load data
    h4_df  = data_engine.load_pair(instrument, TIMEFRAME_H4)
    m15_df = data_engine.load_pair(instrument, TIMEFRAME_M15)

    if h4_df is None or m15_df is None:
        log.error(f"Cannot train {instrument}: missing data files.")
        log.error(f"Place CSV files in data/historical/ as:")
        log.error(f"  {instrument}_H4.csv  (columns: datetime,open,high,low,close,volume)")
        log.error(f"  {instrument}_M15.csv")
        return False

    log.info(f"H4:  {len(h4_df)} bars ({h4_df.index[0]} → {h4_df.index[-1]})")
    log.info(f"M15: {len(m15_df)} bars ({m15_df.index[0]} → {m15_df.index[-1]})")

    feat_eng = FeatureEngineer()

    if not skip_wfo:
        # ── Full QMRP walk-forward optimisation ───────────────────────────────
        log.info("Running QMRP walk-forward optimisation...")
        wfo = WalkForwardOptimiser(instrument)
        result = wfo.run_qmrp(h4_df, m15_df, production_sortino=production_sortino)

        log.info(f"QMRP complete | deploy={result.deploy_new} | {result.deploy_reason}")

        if not result.deploy_new and production_sortino > 0:
            log.warning("New model did not beat production — keeping existing model.")
            return False

    # ── Train final models on FULL dataset ────────────────────────────────────
    log.info("Training final models on full dataset...")

    # H4 features & model
    h4_features  = feat_eng.compute_h4_features(h4_df)
    h4_model     = H4BiasModel(instrument)
    X_h4, y_h4  = h4_model.build_training_data(h4_features.dropna())
    log.info(f"H4 training samples: {len(X_h4)}")
    h4_model.train(X_h4, y_h4)
    h4_path = h4_model.save()
    log.info(f"H4 model saved: {h4_path}")

    # M15 features & model
    m15_features = feat_eng.compute_m15_features(m15_df)
    atr_series   = _atr(m15_df["high"], m15_df["low"], m15_df["close"])
    labels       = M15EntryModel.build_labels(m15_df, atr_series)
    m15_ctx      = FeatureEngineer.merge_h4_context_into_m15(m15_features, h4_features)

    aligned_idx  = m15_ctx.index.intersection(labels.index)
    m15_model    = M15EntryModel(instrument)
    m15_model.train(m15_ctx.loc[aligned_idx], labels.loc[aligned_idx])
    m15_model.save()
    log.info(f"M15 model saved for {instrument}")

    # Log to DB
    db = TradeDatabase(os.getenv("DB_PATH", "logs/trades.db"))
    db.log_model_deployment(
        instrument=instrument,
        sortino=0.0,
        max_dd=0.0,
        trade_count=len(X_h4),
        model_path=h4_path,
    )

    log.info(f"Training complete for {instrument}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train AI Swing Trading Bot models")
    parser.add_argument("--instrument", type=str, default=None,
                        help="Instrument to train (default: all)")
    parser.add_argument("--skip-wfo", action="store_true",
                        help="Skip walk-forward optimisation (direct full-dataset training)")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else INSTRUMENTS
    data_engine = DataEngine(use_mt5=False)

    # Validate data availability
    all_data = data_engine.load_all()
    valid = data_engine.validate_minimum_history(all_data)
    if not valid:
        log.warning("Some instruments do not meet the 10-year minimum history requirement. "
                    "Training will proceed but model quality may be reduced.")

    success_count = 0
    for instrument in instruments:
        ok = train_instrument(instrument, data_engine, skip_wfo=args.skip_wfo)
        if ok:
            success_count += 1

    log.info(f"\nTraining complete: {success_count}/{len(instruments)} instruments trained")
    if success_count == len(instruments):
        log.info("All models ready. Run 'python bot.py' to start live trading.")
    else:
        log.warning("Some instruments failed. Check logs and data files.")


if __name__ == "__main__":
    main()
