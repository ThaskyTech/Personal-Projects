"""
models/h4_bias_model.py
H4 Directional Bias Model — LightGBM gradient boosting classifier.
Outputs P(Long Bias) and P(Short Bias) to weight M15 entry signals.
Blueprint v2.0 Section 8.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from config.settings import H4_BIAS_THRESHOLD, H4_BIAS_NEUTRAL_PENALTY
from utils.logger import setup_logger

log = setup_logger("H4BiasModel")

MODEL_DIR = Path("models/saved")


class H4BiasModel:
    """
    Binary classifier: predicts whether the next N bars will trend up or down.

    Label construction:
        - Look-forward window = 20 H4 bars (~3.3 days)
        - Label = 1 (long bias) if forward return > 0, else 0 (short bias)

    Threshold logic (Section 8):
        - P(long) >= 0.60  → full-weight long signals permitted
        - P(short) >= 0.60 → full-weight short signals permitted
        - Neutral zone     → M15 threshold raised by H4_BIAS_NEUTRAL_PENALTY
    """

    FORWARD_WINDOW = 20       # H4 bars for label generation
    MODEL_FILENAME = "h4_bias_{instrument}.pkl"

    def __init__(self, instrument: str):
        self.instrument = instrument
        self.model      = None
        self.feature_cols: list = []
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def build_training_data(self, h4_features: pd.DataFrame
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create (X, y) from H4 features.
        y = 1 if close N bars forward > current close, else 0.
        The last FORWARD_WINDOW rows are dropped (no future label available).
        """
        X = h4_features.copy().dropna()
        # We need the original close to build labels — pass it via feature df
        # or use a convention that 'ret_1b' exists
        if "ret_1b" not in X.columns:
            raise ValueError("h4_features must contain 'ret_1b' column for label generation")

        # Cumulative forward return over FORWARD_WINDOW bars
        # We compute on the raw (pre-normalised) ret series by summing z-scores
        # as a directional proxy (sign is preserved post-normalisation)
        fwd_ret = X["ret_1b"].shift(-self.FORWARD_WINDOW).rolling(self.FORWARD_WINDOW).sum()
        y = (fwd_ret > 0).astype(int)

        # Drop rows without future labels
        valid = y.notna()
        X = X[valid]
        y = y[valid]

        # Store feature columns for inference consistency
        self.feature_cols = X.columns.tolist()
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series,
              hyperparams: Optional[Dict] = None):
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            default_params = {
                "objective":        "binary",
                "metric":           "binary_logloss",
                "n_estimators":     300,
                "learning_rate":    0.05,
                "max_depth":        5,
                "num_leaves":       31,
                "min_child_samples": 30,
                "subsample":        0.8,
                "colsample_bytree": 0.8,
                "reg_alpha":        0.1,
                "reg_lambda":       0.1,
                "n_jobs":           -1,
                "verbose":          -1,
                "random_state":     42,
            }
            if hyperparams:
                default_params.update(hyperparams)
            self.model = lgb.LGBMClassifier(**default_params)
        except ImportError:
            log.warning("LightGBM not available — falling back to GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier
            default_params = {
                "n_estimators":  200,
                "learning_rate": 0.05,
                "max_depth":     4,
                "subsample":     0.8,
                "random_state":  42,
            }
            if hyperparams:
                default_params.update(hyperparams)
            self.model = GradientBoostingClassifier(**default_params)

        self.feature_cols = X.columns.tolist()
        X_clean = X[self.feature_cols].fillna(0)
        self.model.fit(X_clean, y)
        log.info(f"H4BiasModel trained for {self.instrument} "
                 f"on {len(X)} samples, {len(self.feature_cols)} features")

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, h4_features_row: pd.Series) -> Dict[str, float]:
        """
        Predict bias from a single H4 feature row.
        Returns: {
            'p_long': float,
            'p_short': float,
            'bias': 'long' | 'short' | 'neutral',
            'threshold_penalty': float   (0 or H4_BIAS_NEUTRAL_PENALTY)
        }
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        x = pd.DataFrame(
            [h4_features_row[self.feature_cols].fillna(0).values],
            columns=self.feature_cols
        )
        proba = self.model.predict_proba(x)[0]
        p_short, p_long = proba[0], proba[1]

        if p_long >= H4_BIAS_THRESHOLD:
            bias = "long"
            penalty = 0.0
        elif p_short >= H4_BIAS_THRESHOLD:
            bias = "short"
            penalty = 0.0
        else:
            bias = "neutral"
            penalty = H4_BIAS_NEUTRAL_PENALTY

        return {
            "p_long":           round(p_long, 4),
            "p_short":          round(p_short, 4),
            "bias":             bias,
            "threshold_penalty": penalty,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self):
        path = MODEL_DIR / self.MODEL_FILENAME.format(instrument=self.instrument)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)
        log.info(f"H4BiasModel saved → {path}")
        return str(path)

    def load(self) -> bool:
        path = MODEL_DIR / self.MODEL_FILENAME.format(instrument=self.instrument)
        if not path.exists():
            log.warning(f"No saved H4BiasModel found at {path}")
            return False
        data = joblib.load(path)
        self.model        = data["model"]
        self.feature_cols = data["feature_cols"]
        log.info(f"H4BiasModel loaded from {path}")
        return True
