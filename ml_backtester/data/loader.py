"""Download and locally cache daily OHLCV price data for the configured universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from ml_backtester.config import Config

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(cache_dir: str, ticker: str) -> Path:
    """Return the parquet cache file path for a single ticker."""
    return Path(cache_dir) / f"{ticker}.parquet"


def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch raw daily OHLCV data for one ticker from Yahoo Finance.

    Kept separate from load_ticker so tests can monkeypatch this single
    function instead of hitting the network.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index.name = "date"
    return raw[_COLUMNS]


def load_ticker(
    ticker: str, start: str, end: str, cache_dir: str, force_refresh: bool = False
) -> pd.DataFrame:
    """Load one ticker's daily OHLCV series, using the local parquet cache when possible.

    Args:
        ticker: Ticker symbol.
        start: Inclusive start date, "YYYY-MM-DD".
        end: Inclusive end date, "YYYY-MM-DD".
        cache_dir: Directory holding per-ticker parquet caches.
        force_refresh: If True, re-download even if a cache file exists.

    Returns:
        DataFrame indexed by date with columns open, high, low, close,
        volume, restricted to [start, end].
    """
    path = _cache_path(cache_dir, ticker)
    cached = pd.read_parquet(path) if path.exists() else None

    needs_download = (
        force_refresh
        or cached is None
        or cached.empty
        or cached.index.min() > pd.Timestamp(start)
        or cached.index.max() < pd.Timestamp(end)
    )

    if needs_download:
        data = _download_ticker(ticker, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path)
    else:
        data = cached

    return data.loc[(data.index >= start) & (data.index <= end)]


def load_universe(config: Config, force_refresh: bool = False) -> pd.DataFrame:
    """Load daily OHLCV data for every ticker in config.universe.

    Args:
        config: Pipeline configuration providing universe, date range, and cache_dir.
        force_refresh: If True, bypass the cache and re-download every ticker.

    Returns:
        Long-format DataFrame indexed by date (ascending), with a "ticker"
        column plus open, high, low, close, volume — one row per
        (date, ticker).
    """
    frames = []
    for ticker in config.universe:
        data = load_ticker(
            ticker, config.start_date, config.end_date, config.cache_dir, force_refresh
        )
        data = data.copy()
        data["ticker"] = ticker
        frames.append(data)
    return pd.concat(frames).sort_index()
