PLACE YOUR HISTORICAL DATA FILES HERE
======================================

Required files (CSV or Parquet format):

  EURUSD_H4.csv
  EURUSD_M15.csv
  GBPUSD_H4.csv
  GBPUSD_M15.csv

CSV column format:
  datetime, open, high, low, close, volume

  - datetime: ISO format e.g. "2015-01-02 08:00:00"
  - open/high/low/close: price as float (e.g. 1.12345)
  - volume: tick volume as integer
  - spread: optional float column (pips)

MINIMUM HISTORY REQUIRED:
  10 years (January 2014 onwards recommended)

RECOMMENDED DATA SOURCES:
  - Dukascopy (https://www.dukascopy.com/swiss/english/marketwatch/historical/)
  - FP Markets historical data request (contact your broker)
  - MetaTrader 5 built-in history (Tools > History Centre)

Parquet files are also accepted (faster loading):
  EURUSD_H4.parquet
  EURUSD_M15.parquet
  GBPUSD_H4.parquet
  GBPUSD_M15.parquet
