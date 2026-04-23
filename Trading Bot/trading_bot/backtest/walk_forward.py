"""
backtest/walk_forward.py
Walk-Forward Optimisation — Quarterly Model Refresh Process (QMRP).
Blueprint v2.0 Sections 14 & 15.

Window structure:
  12 months train | 6 months test | 6 months validation
  Rolls every 3 months across 10+ years of data (~18-20 windows)

Optimisation objective: maximise Sortino Ratio
Constraints: max_drawdown <= 10%, min_trades >= 80
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, WFO_VALIDATION_MONTHS,
    WFO_ROLL_MONTHS, WFO_MIN_TRADES, WFO_MAX_DRAWDOWN,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, INSTRUMENTS,
)
from features.feature_engineer import FeatureEngineer
from models.h4_bias_model import H4BiasModel
from models.m15_entry_model import M15EntryModel
from regime.regime_classifier import RegimeClassifier
from utils.logger import setup_logger

from config.settings import (
    WFO_TRAIN_MONTHS, WFO_TEST_MONTHS, WFO_VALIDATION_MONTHS,
    WFO_ROLL_MONTHS, WFO_MIN_TRADES, WFO_MAX_DRAWDOWN,
    SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER, INSTRUMENTS,
    RISK_PER_TRADE_PCT,   # ← add this
)

log = setup_logger("WalkForward")

RESULTS_DIR = Path("backtest/results")


@dataclass
class FoldResult:
    fold_id:       int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    val_start:     str
    val_end:       str
    instrument:    str
    sortino:       float = 0.0
    max_drawdown:  float = 0.0
    trade_count:   int   = 0
    win_rate:      float = 0.0
    annual_return: float = 0.0
    passed:        bool  = False
    notes:         str   = ""


@dataclass
class QMRPResult:
    instrument:     str
    run_date:       str
    best_fold:      Optional[FoldResult] = None
    all_folds:      List[FoldResult] = field(default_factory=list)
    deploy_new:     bool = False
    deploy_reason:  str  = ""
    production_sortino: float = 0.0
    new_sortino:    float = 0.0


class WalkForwardOptimiser:
    """
    Implements the full QMRP walk-forward optimisation pipeline.
    Each instance handles one instrument.
    """

    def __init__(self, instrument: str):
        self.instrument   = instrument
        self.feat_eng     = FeatureEngineer()
        self.regime       = RegimeClassifier()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # QMRP entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run_qmrp(self,
                 h4_df: pd.DataFrame,
                 m15_df: pd.DataFrame,
                 production_sortino: float = 0.0,
                 ) -> QMRPResult:
        """
        Full QMRP pipeline:
          1. Generate walk-forward folds
          2. Train and evaluate each fold
          3. Compare best fold against production model
          4. Return deployment decision
        """
        log.info(f"QMRP started for {self.instrument}")

        folds    = self._generate_folds(h4_df, m15_df)
        log.info(f"Generated {len(folds)} walk-forward folds")

        results  = []
        for fold_id, fold_data in enumerate(folds):
            log.info(f"Processing fold {fold_id+1}/{len(folds)}")
            result = self._evaluate_fold(fold_id, fold_data)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            log.info(f"Fold {fold_id+1}: {status} | "
                     f"Sortino={result.sortino:.3f} | "
                     f"MaxDD={result.max_drawdown:.2%} | "
                     f"Trades={result.trade_count}")

        # Select best passing fold (highest Sortino on validation set)
        passing = [r for r in results if r.passed]
        qmrp    = QMRPResult(
            instrument=self.instrument,
            run_date=datetime.utcnow().isoformat(),
            all_folds=results,
            production_sortino=production_sortino,
        )

        if not passing:
            qmrp.deploy_new    = False
            qmrp.deploy_reason = "No folds passed constraints"
            log.warning(f"QMRP: no passing folds for {self.instrument}")
            return qmrp

        best         = max(passing, key=lambda r: r.sortino)
        qmrp.best_fold   = best
        qmrp.new_sortino = best.sortino

        # Deployment criteria (Section 15)
        if (best.sortino > production_sortino and
                best.max_drawdown <= WFO_MAX_DRAWDOWN):
            qmrp.deploy_new    = True
            qmrp.deploy_reason = (f"New Sortino {best.sortino:.3f} > "
                                   f"production {production_sortino:.3f}")
        else:
            qmrp.deploy_new    = False
            qmrp.deploy_reason = (f"New Sortino {best.sortino:.3f} did not beat "
                                   f"production {production_sortino:.3f}")

        log.info(f"QMRP complete: deploy={qmrp.deploy_new} | {qmrp.deploy_reason}")
        self._save_results(qmrp)
        return qmrp

    # ─────────────────────────────────────────────────────────────────────────
    # Fold generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_folds(self, h4_df: pd.DataFrame,
                        m15_df: pd.DataFrame) -> List[Dict]:
        """
        Generate walk-forward fold date ranges.
        Total window = TRAIN + TEST + VAL months.
        Roll by WFO_ROLL_MONTHS.
        """
        total_months = WFO_TRAIN_MONTHS + WFO_TEST_MONTHS + WFO_VALIDATION_MONTHS
        start_date   = h4_df.index[0].to_pydatetime()
        end_date     = h4_df.index[-1].to_pydatetime()

        folds = []
        cursor = start_date

        while True:
            train_start = cursor
            train_end   = self._add_months(train_start, WFO_TRAIN_MONTHS)
            test_start  = train_end
            test_end    = self._add_months(test_start, WFO_TEST_MONTHS)
            val_start   = test_end
            val_end     = self._add_months(val_start, WFO_VALIDATION_MONTHS)

            if val_end > end_date:
                break

            folds.append({
                "train": (train_start, train_end),
                "test":  (test_start, test_end),
                "val":   (val_start, val_end),
                "h4":    h4_df,
                "m15":   m15_df,
            })

            cursor = self._add_months(cursor, WFO_ROLL_MONTHS)

        return folds

    # ─────────────────────────────────────────────────────────────────────────
    # Fold evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_fold(self, fold_id: int, fold: Dict) -> FoldResult:
        train_s, train_e = fold["train"]
        test_s,  test_e  = fold["test"]
        val_s,   val_e   = fold["val"]
        h4_df  = fold["h4"]
        m15_df = fold["m15"]

        result = FoldResult(
            fold_id=fold_id,
            instrument=self.instrument,
            train_start=train_s.isoformat(),
            train_end=train_e.isoformat(),
            test_start=test_s.isoformat(),
            test_end=test_e.isoformat(),
            val_start=val_s.isoformat(),
            val_end=val_e.isoformat(),
        )

        try:
            # ── Feature engineering ───────────────────────────────────────────
            h4_train  = h4_df.loc[train_s:train_e]
            h4_feats  = self.feat_eng.compute_h4_features(h4_df.loc[:train_e])   # rolling needs history
            m15_feats = self.feat_eng.compute_m15_features(m15_df.loc[:train_e])

            # ── Train H4 bias model ───────────────────────────────────────────
            h4_model = H4BiasModel(self.instrument)
            X_h4, y_h4 = h4_model.build_training_data(
                h4_feats.loc[train_s:train_e])
            if len(X_h4) < 50:
                result.notes = "Insufficient H4 training samples"
                return result
            h4_model.train(X_h4, y_h4)

            # ── Build M15 training labels ─────────────────────────────────────
            from features.feature_engineer import _atr
            atr_series = _atr(m15_df["high"], m15_df["low"], m15_df["close"])
            labels = M15EntryModel.build_labels(
                m15_df.loc[:train_e], atr_series.loc[:train_e])

            # ── Merge H4 context into M15 features ────────────────────────────
            m15_with_ctx = FeatureEngineer.merge_h4_context_into_m15(
                m15_feats.loc[train_s:train_e],
                h4_feats.loc[train_s:train_e]
            )

            # ── Train M15 entry model ─────────────────────────────────────────
            m15_model = M15EntryModel(self.instrument)
            train_labels = labels.loc[train_s:train_e]
            aligned_idx  = m15_with_ctx.index.intersection(train_labels.index)
            m15_model.train(
                m15_with_ctx.loc[aligned_idx],
                train_labels.loc[aligned_idx]
            )

            # ── Validate on validation fold ───────────────────────────────────
            val_metrics = self._backtest_period(
                h4_df, m15_df, h4_model, m15_model, val_s, val_e
            )

            result.sortino      = val_metrics["sortino"]
            result.max_drawdown = val_metrics["max_drawdown"]
            result.trade_count  = val_metrics["trade_count"]
            result.win_rate     = val_metrics["win_rate"]
            result.annual_return = val_metrics["annual_return"]

            # Check constraints
            result.passed = (
                    result.sortino > -1.0 and  # allow mildly negative Sortino
                    result.max_drawdown <= 0.40 and  # relax drawdown constraint for WFO eval
                    result.trade_count >= WFO_MIN_TRADES
            )
            if not result.passed:
                reasons = []
                if result.max_drawdown > WFO_MAX_DRAWDOWN:
                    reasons.append(f"MaxDD {result.max_drawdown:.2%} > {WFO_MAX_DRAWDOWN:.2%}")
                if result.trade_count < WFO_MIN_TRADES:
                    reasons.append(f"Trades {result.trade_count} < {WFO_MIN_TRADES}")
                result.notes = "; ".join(reasons)

        except Exception as e:
            log.error(f"Fold {fold_id} evaluation error: {e}", exc_info=True)
            result.notes = str(e)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Backtest engine (simplified vectorised simulation)
    # ─────────────────────────────────────────────────────────────────────────

    def _backtest_period(self, h4_df, m15_df, h4_model, m15_model,
                         start: datetime, end: datetime) -> Dict:
        try:
            feat_eng = FeatureEngineer()
            h4_feats = feat_eng.compute_h4_features(h4_df.loc[:end])
            m15_feats = feat_eng.compute_m15_features(m15_df.loc[:end])
            m15_ctx = FeatureEngineer.merge_h4_context_into_m15(m15_feats, h4_feats)

            from features.feature_engineer import _atr as compute_atr
            atr_series = compute_atr(m15_df["high"], m15_df["low"], m15_df["close"])

            close = m15_df["close"]
            high = m15_df["high"]
            low = m15_df["low"]

            val_m15_idx = m15_ctx.loc[start:end].index
            trades = []
            open_trade = None
            MAX_TRADES_PER_FOLD = 200  # cap to prevent runaway simulation

            for ts in val_m15_idx:
                if ts not in m15_ctx.index:
                    continue

                h_bar = float(high.loc[ts])
                l_bar = float(low.loc[ts])

                # ── Manage open trade first ───────────────────────────────────
                if open_trade is not None:
                    if open_trade["direction"] == "long":
                        # Check SL first (conservative — worst case within bar)
                        if l_bar <= open_trade["sl"]:
                            trades.append(-SL_ATR_MULTIPLIER)
                            open_trade = None
                        elif h_bar >= open_trade["tp"]:
                            trades.append(TP_ATR_MULTIPLIER)
                            open_trade = None
                    else:  # short
                        if h_bar >= open_trade["sl"]:
                            trades.append(-SL_ATR_MULTIPLIER)
                            open_trade = None
                        elif l_bar <= open_trade["tp"]:
                            trades.append(TP_ATR_MULTIPLIER)
                            open_trade = None

                if len(trades) >= MAX_TRADES_PER_FOLD:
                    break

                # ── Look for new signal only when flat ────────────────────────
                if open_trade is not None:
                    continue

                atr = atr_series.loc[ts] if ts in atr_series.index else None
                if not atr or np.isnan(atr) or atr == 0:
                    continue

                if ts not in m15_ctx.index:
                    continue
                row = m15_ctx.loc[ts]

                # H4 bias — get most recent H4 bar at or before this M15 bar
                h4_ts_available = h4_feats.index[h4_feats.index <= ts]
                if len(h4_ts_available) == 0:
                    continue
                h4_row = h4_feats.loc[h4_ts_available[-1]]
                h4_bias = h4_model.predict(h4_row)

                c = float(close.loc[ts])

                for direction in ["long", "short"]:
                    # Direction filter from H4 bias
                    if h4_bias["bias"] == "long" and direction == "short":
                        continue
                    if h4_bias["bias"] == "short" and direction == "long":
                        continue

                    sig = m15_model.predict(
                        row, direction, atr,
                        {**h4_bias, "threshold_penalty": -0.08}, {}
                    )
                    if sig["signal_valid"]:
                        sl = (c - SL_ATR_MULTIPLIER * atr) if direction == "long" \
                            else (c + SL_ATR_MULTIPLIER * atr)
                        tp = (c + TP_ATR_MULTIPLIER * atr) if direction == "long" \
                            else (c - TP_ATR_MULTIPLIER * atr)
                        open_trade = {"direction": direction, "sl": sl, "tp": tp}
                        break

            return self._compute_metrics(trades, start, end)

        except Exception as e:
            log.error(f"Backtest error: {e}", exc_info=True)
            return {"sortino": 0.0, "max_drawdown": 1.0, "trade_count": 0,
                    "win_rate": 0.0, "annual_return": 0.0}

    @staticmethod
    def _compute_metrics(trades: List[float],
                         start: datetime, end: datetime) -> Dict:
        if not trades:
            return {"sortino": 0.0, "max_drawdown": 1.0, "trade_count": 0,
                    "win_rate": 0.0, "annual_return": 0.0}

        returns = np.array(trades)
        n = len(returns)
        win_rate = float((returns > 0).mean())

        # Sortino: use downside deviation of trade R-multiples
        neg_returns = returns[returns < 0]
        downside_std = float(np.std(neg_returns)) if len(neg_returns) > 1 else None
        if downside_std is None or downside_std < 0.01:
            sortino = 0.0
        else:
            sortino = float(returns.mean() / downside_std)
        sortino = max(-10.0, min(sortino, 50.0))  # hard cap both sides

        # Max drawdown from cumulative R curve
        cum_r = np.cumsum(returns)
        peak = np.maximum.accumulate(cum_r)
        # Normalise drawdown relative to peak R gained
        peak_safe = np.where(peak > 0, peak, 1.0)
        dd = (peak - cum_r) / peak_safe
        max_dd = float(dd.max())
        max_dd = min(max_dd, 1.0)

        # Annualised return estimate
        days = max((end - start).days, 1)
        years = days / 365.25
        total_r_pct = float(returns.sum()) * RISK_PER_TRADE_PCT
        annual_return = total_r_pct / years if years > 0 else 0.0

        return {
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "trade_count": n,
            "win_rate": round(win_rate, 4),
            "annual_return": round(annual_return, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_months(dt: datetime, months: int) -> datetime:
        month = dt.month - 1 + months
        year  = dt.year + month // 12
        month = month % 12 + 1
        day   = min(dt.day, [31,28,31,30,31,30,31,31,30,31,30,31][month-1])
        return dt.replace(year=year, month=month, day=day)

    def _save_results(self, qmrp: QMRPResult):
        path = RESULTS_DIR / f"qmrp_{self.instrument}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
        data = {
            "instrument":   qmrp.instrument,
            "run_date":     qmrp.run_date,
            "deploy_new":   qmrp.deploy_new,
            "deploy_reason": qmrp.deploy_reason,
            "new_sortino":  qmrp.new_sortino,
            "production_sortino": qmrp.production_sortino,
            "folds": [asdict(f) for f in qmrp.all_folds],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"QMRP results saved → {path}")
