"""Forward-return labels, each stamped with a label end-time (t1) for purging.

t1 is the timestamp at which a sample's outcome is actually known. A sample
observed at t0 with a 5-day horizon isn't resolved until t1 = t0 + 5 trading
days later; any training sample whose [t0, t1] interval overlaps a test
window has seen information from inside (or after) that test window and
must be purged. See ml_backtester.validation.splitters.
"""

from __future__ import annotations

import pandas as pd

from ml_backtester.config import Config


def _forward_return(close: pd.Series, horizon_days: int) -> pd.Series:
    """Percentage return from t0 to t0 + horizon_days, indexed at t0."""
    return close.shift(-horizon_days) / close - 1


def _label_end_time(index: pd.DatetimeIndex, horizon_days: int) -> pd.Series:
    """Map each t0 to its t1: the timestamp horizon_days trading days later.

    Returns NaT for the trailing observations where t0 + horizon_days falls
    past the end of the available series — those samples have no resolved
    label yet and must be dropped.
    """
    t1 = pd.Series(index=index, dtype="datetime64[ns]")
    if horizon_days < len(index):
        t1.iloc[: len(index) - horizon_days] = index[horizon_days:]
    return t1


def add_labels_single_ticker(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach a forward-return label and its label end-time (t1) to one ticker's prices.

    Args:
        prices: Single-ticker DataFrame indexed by date with a "close" column,
            sorted ascending by date.
        config: Pipeline configuration providing horizon_days.

    Returns:
        DataFrame indexed by date (t0) with columns "label" (forward return
        over config.horizon_days) and "t1" (label end-time). Rows whose
        label or t1 is unresolved (trailing horizon_days observations) are
        dropped.
    """
    labels = pd.DataFrame(
        {
            "label": _forward_return(prices["close"], config.horizon_days),
            "t1": _label_end_time(prices.index, config.horizon_days),
        },
        index=prices.index,
    )
    return labels.dropna()


def add_labels(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach forward-return labels and t1 to every ticker in a long-format price DataFrame.

    Args:
        prices: Long-format DataFrame as returned by
            ml_backtester.data.loader.load_universe — indexed by date with
            a "ticker" column plus a "close" column.
        config: Pipeline configuration providing horizon_days.

    Returns:
        Long-format DataFrame indexed by date (t0) with columns "ticker",
        "label", "t1" — one row per (date, ticker) with a resolved label.
    """
    per_ticker = []
    for ticker, group in prices.groupby("ticker", sort=False):
        labels = add_labels_single_ticker(group.sort_index(), config)
        labels["ticker"] = ticker
        per_ticker.append(labels)
    return pd.concat(per_ticker).sort_index()
