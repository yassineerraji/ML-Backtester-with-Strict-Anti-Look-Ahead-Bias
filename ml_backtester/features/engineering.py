"""Causal technical features computed per ticker from trailing rolling windows only.

Every function here must be computable in real time at date t using only
data up to and including t. None may reference the mean/std/min/max of the
full series — that would leak future information into the feature.
"""

from __future__ import annotations

import pandas as pd

from ml_backtester.config import Config

FEATURE_COLUMNS = ["momentum", "volatility", "relative_volume", "rsi"]


def _momentum(close: pd.Series, window: int) -> pd.Series:
    """Trailing percentage price change over `window` trading days."""
    return close.pct_change(window)


def _realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Trailing rolling standard deviation of daily returns over `window` days."""
    return close.pct_change().rolling(window).std()


def _relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """Today's volume relative to its trailing rolling mean over `window` days."""
    return volume / volume.rolling(window).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's Relative Strength Index over a trailing `window`-day rolling average."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Standardize a series against its own trailing rolling mean/std, not the full series."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def compute_features_single_ticker(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute all technical features for one ticker's OHLCV history.

    Args:
        prices: Single-ticker DataFrame indexed by date with columns
            open, high, low, close, volume, sorted ascending by date.
        config: Pipeline configuration providing feature window lengths.

    Returns:
        DataFrame indexed by date with columns momentum, volatility,
        relative_volume, rsi — each rolling-normalized via
        config.normalization_window so features are on a comparable scale
        without using any full-series statistics.
    """
    raw = pd.DataFrame(
        {
            "momentum": _momentum(prices["close"], config.momentum_window),
            "volatility": _realized_volatility(prices["close"], config.volatility_window),
            "relative_volume": _relative_volume(prices["volume"], config.relative_volume_window),
            "rsi": _rsi(prices["close"], config.rsi_window),
        }
    )[FEATURE_COLUMNS]
    return raw.apply(lambda col: _rolling_zscore(col, config.normalization_window))


def compute_features(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute technical features for every ticker in a long-format price DataFrame.

    Args:
        prices: Long-format DataFrame as returned by
            ml_backtester.data.loader.load_universe — indexed by date with
            a "ticker" column plus open, high, low, close, volume.
        config: Pipeline configuration providing feature window lengths.

    Returns:
        Long-format DataFrame indexed by date with a "ticker" column plus
        the feature columns, one row per (date, ticker). Rows with any NaN
        feature (from the rolling warm-up period) are dropped.
    """
    per_ticker = []
    for ticker, group in prices.groupby("ticker", sort=False):
        features = compute_features_single_ticker(group.sort_index(), config)
        features["ticker"] = ticker
        per_ticker.append(features)
    combined = pd.concat(per_ticker).sort_index()
    return combined.dropna()
