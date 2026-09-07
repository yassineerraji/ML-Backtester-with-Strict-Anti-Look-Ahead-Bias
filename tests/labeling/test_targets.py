"""Tests for label construction: correct forward return and correct t1 stamping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.config import Config
from ml_backtester.labeling.targets import add_labels, add_labels_single_ticker


@pytest.fixture
def config() -> Config:
    return Config(horizon_days=5)


def _flat_close_prices(n: int) -> pd.DataFrame:
    """Prices that increase by exactly 1 per day, so returns are easy to hand-check."""
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    close = 100 + np.arange(n)
    return pd.DataFrame({"close": close}, index=index)


def test_label_matches_known_forward_return(config):
    prices = _flat_close_prices(20)
    labels = add_labels_single_ticker(prices, config)

    t0 = prices.index[0]
    expected_return = prices.loc[prices.index[5], "close"] / prices.loc[t0, "close"] - 1
    assert labels.loc[t0, "label"] == pytest.approx(expected_return)


def test_t1_is_horizon_days_after_t0(config):
    prices = _flat_close_prices(20)
    labels = add_labels_single_ticker(prices, config)

    t0 = prices.index[3]
    expected_t1 = prices.index[3 + config.horizon_days]
    assert labels.loc[t0, "t1"] == expected_t1


def test_trailing_unresolved_rows_are_dropped(config):
    prices = _flat_close_prices(20)
    labels = add_labels_single_ticker(prices, config)

    assert len(labels) == 20 - config.horizon_days
    assert labels.index.max() < prices.index.max()
    assert not labels.isna().any().any()


def test_add_labels_preserves_ticker_and_horizon_across_tickers(config):
    prices = pd.concat(
        [
            _flat_close_prices(20).assign(ticker="AAA"),
            _flat_close_prices(20).assign(ticker="BBB"),
        ]
    )
    labels = add_labels(prices, config)

    assert set(labels["ticker"].unique()) == {"AAA", "BBB"}
    assert (labels["t1"] > labels.index).all()
