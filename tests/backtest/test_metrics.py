"""Tests for performance metrics against hand-computed values on small fixed series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.backtest.metrics import (
    hit_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


def test_sharpe_ratio_matches_hand_computation():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])

    result = sharpe_ratio(returns, periods_per_year=252)

    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_sharpe_ratio_is_nan_for_zero_variance():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(sharpe_ratio(returns))


def test_sortino_ratio_only_penalizes_downside():
    returns = pd.Series([0.02, 0.03, -0.01, 0.04, -0.02])

    result = sortino_ratio(returns, periods_per_year=252)

    downside_std = returns[returns < 0].std()
    expected = returns.mean() / downside_std * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_sortino_ratio_is_nan_with_no_downside():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isnan(sortino_ratio(returns))


def test_max_drawdown_on_known_path():
    # Equity: 1.00 -> 1.10 -> 0.99 -> 1.05. Trough 0.99 is a 10% drop from peak 1.10.
    returns = pd.Series([0.10, -0.10, 0.0606060606])

    result = max_drawdown(returns)

    assert result == pytest.approx(-0.10, abs=1e-6)


def test_max_drawdown_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.01])
    assert max_drawdown(returns) == pytest.approx(0.0)


def test_turnover_matches_mean_absolute_position_change():
    positions = pd.Series([0.0, 0.5, 0.5, -0.5])

    result = turnover(positions)

    expected = positions.diff().abs().mean()
    assert result == pytest.approx(expected)


def test_hit_ratio_matches_fraction_of_positive_returns():
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])

    result = hit_ratio(returns)

    assert result == pytest.approx(2 / 5)
