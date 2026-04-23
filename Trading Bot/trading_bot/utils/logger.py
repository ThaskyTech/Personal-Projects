"""
utils/logger.py
Centralised logging setup and SQLite trade database for the trading bot.
"""

import logging
import os
import shutil
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

from config.settings import LOG_FORMAT, DB_TABLE_TRADES


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger writing to console and a rotating daily file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — one file per day
    fh = logging.FileHandler(log_dir / f"bot_{date.today().isoformat()}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# SQLite Trade Database
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TRADES_SQL = f"""
CREATE TABLE IF NOT EXISTS {DB_TABLE_TRADES} (
    trade_id            TEXT PRIMARY KEY,
    instrument          TEXT NOT NULL,
    direction           TEXT NOT NULL,        -- 'long' | 'short'
    entry_time          TEXT,
    exit_time           TEXT,
    entry_price         REAL,
    exit_price          REAL,
    sl_price            REAL,
    tp_price            REAL,
    lot_size            REAL,
    pnl                 REAL,
    spread_at_entry     REAL,
    entry_probability   REAL,
    expected_return     REAL,
    expected_downside   REAL,
    regime_trending     REAL,
    regime_ranging      REAL,
    regime_high_vol     REAL,
    regime_low_vol      REAL,
    h4_bias_long        REAL,
    h4_bias_short       REAL,
    exit_reason         TEXT,                 -- 'tp_hit' | 'sl_hit' | 'manual'
    mt5_ticket          INTEGER,
    created_at          TEXT DEFAULT (datetime('now'))
);
"""

CREATE_EQUITY_SQL = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    snapshot_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    equity         REAL NOT NULL,
    balance        REAL NOT NULL,
    open_risk_pct  REAL
);
"""

CREATE_MODEL_LOG_SQL = """
CREATE TABLE IF NOT EXISTS model_deployments (
    deploy_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument     TEXT NOT NULL,
    deployed_at    TEXT NOT NULL,
    sortino_ratio  REAL,
    max_drawdown   REAL,
    trade_count    INTEGER,
    model_path     TEXT
);
"""


class TradeDatabase:
    """Thin wrapper around SQLite for trade logging and QMRP CSV export."""

    def __init__(self, db_path: str = "logs/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(CREATE_TRADES_SQL)
            conn.execute(CREATE_EQUITY_SQL)
            conn.execute(CREATE_MODEL_LOG_SQL)
            conn.commit()

    # ── Trades ────────────────────────────────────────────────────────────────

    def insert_trade(self, trade: Dict[str, Any]):
        cols = ", ".join(trade.keys())
        placeholders = ", ".join("?" * len(trade))
        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {DB_TABLE_TRADES} ({cols}) VALUES ({placeholders})",
                list(trade.values())
            )
            conn.commit()

    def update_trade_exit(self, trade_id: str, exit_time: str, exit_price: float,
                          pnl: float, exit_reason: str):
        with self._conn() as conn:
            conn.execute(
                f"""UPDATE {DB_TABLE_TRADES}
                    SET exit_time=?, exit_price=?, pnl=?, exit_reason=?
                    WHERE trade_id=?""",
                (exit_time, exit_price, pnl, exit_reason, trade_id)
            )
            conn.commit()

    def get_open_trades(self) -> list:
        with self._conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM {DB_TABLE_TRADES} WHERE exit_time IS NULL"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_trades_since(self, since_iso: str) -> list:
        with self._conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM {DB_TABLE_TRADES} WHERE entry_time >= ?",
                (since_iso,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Equity snapshots ──────────────────────────────────────────────────────

    def log_equity(self, equity: float, balance: float, open_risk_pct: float):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_snapshots (ts, equity, balance, open_risk_pct) VALUES (?,?,?,?)",
                (datetime.utcnow().isoformat(), equity, balance, open_risk_pct)
            )
            conn.commit()

    # ── Model deployments ─────────────────────────────────────────────────────

    def log_model_deployment(self, instrument: str, sortino: float,
                              max_dd: float, trade_count: int, model_path: str):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO model_deployments
                   (instrument, deployed_at, sortino_ratio, max_drawdown, trade_count, model_path)
                   VALUES (?,?,?,?,?,?)""",
                (instrument, datetime.utcnow().isoformat(), sortino, max_dd, trade_count, model_path)
            )
            conn.commit()

    # ── QMRP CSV export ───────────────────────────────────────────────────────

    def export_trades_csv(self, output_path: Optional[str] = None) -> str:
        import csv
        if output_path is None:
            output_path = f"logs/trades_export_{date.today().isoformat()}.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            cur = conn.execute(f"SELECT * FROM {DB_TABLE_TRADES}")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        return output_path

    # ── Daily backup ──────────────────────────────────────────────────────────

    def backup(self, backup_dir: str = "logs/backups"):
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        dest = Path(backup_dir) / f"trades_{date.today().isoformat()}.db"
        shutil.copy2(self.db_path, dest)
        return str(dest)
