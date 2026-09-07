"""Tests for transaction cost models against hand-computed values."""

from __future__ import annotations

import numpy as np

from ml_backtester.backtest.costs import (
    market_impact_cost,
    spread_slippage_cost,
    total_transaction_cost,
)
from ml_backtester.config import Config


def test_spread_slippage_cost_matches_hand_computation():
    config = Config(spread_bps=5.0, slippage_bps=2.0)
    trade_size = np.array([0.0, 0.5, 1.0])

    cost = spread_slippage_cost(trade_size, config)

    expected = trade_size * (7.0 / 10_000)
    np.testing.assert_allclose(cost, expected)


def test_market_impact_cost_is_zero_when_disabled():
    config = Config(use_market_impact=False, impact_coefficient=0.1)
    trade_size = np.array([0.25, 1.0])

    cost = market_impact_cost(trade_size, config)

    np.testing.assert_array_equal(cost, [0.0, 0.0])


def test_market_impact_cost_follows_square_root_law_when_enabled():
    config = Config(use_market_impact=True, impact_coefficient=0.1)
    trade_size = np.array([0.25, 1.0])

    cost = market_impact_cost(trade_size, config)

    expected = 0.1 * np.sqrt(trade_size)
    np.testing.assert_allclose(cost, expected)


def test_total_transaction_cost_sums_both_components():
    config = Config(
        spread_bps=5.0, slippage_bps=2.0, use_market_impact=True, impact_coefficient=0.1
    )
    trade_size = np.array([1.0])

    total = total_transaction_cost(trade_size, config)

    expected = spread_slippage_cost(trade_size, config) + market_impact_cost(trade_size, config)
    np.testing.assert_allclose(total, expected)
