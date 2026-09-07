"""Tests for feature engineering: causality (no lookahead) and basic sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.config import Config
from ml_backtester.features.engineering import compute_features, compute_features_single_ticker


@pytest.fixture
def config() -> Config:
    return Config(
        momentum_window=5,
        volatility_window=5,
        relative_volume_window=5,
        rsi_window=5,
        normalization_window=10,
    )


def _synthetic_prices(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=index,
    )


def test_features_are_causal(config):
    """A feature value at date t must not change when rows after t are appended."""
    full = _synthetic_prices(80)
    truncated = full.iloc[:50]

    features_full = compute_features_single_ticker(full, config)
    features_truncated = compute_features_single_ticker(truncated, config)

    common_dates = features_truncated.index
    pd.testing.assert_frame_equal(
        features_full.loc[common_dates], features_truncated.loc[common_dates]
    )


def test_warmup_rows_are_dropped_for_multi_ticker(config):
    prices = pd.concat(
        [
            _synthetic_prices(60, seed=1).assign(ticker="AAA"),
            _synthetic_prices(60, seed=2).assign(ticker="BBB"),
        ]
    )
    features = compute_features(prices, config)

    assert not features.isna().any().any()
    assert set(features["ticker"].unique()) == {"AAA", "BBB"}


def test_single_ticker_feature_columns(config):
    prices = _synthetic_prices(60)
    features = compute_features_single_ticker(prices, config)

    assert list(features.columns) == ["momentum", "volatility", "relative_volume", "rsi"]
    assert features.dropna().shape[0] > 0
