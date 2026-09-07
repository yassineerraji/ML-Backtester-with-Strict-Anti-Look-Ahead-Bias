"""Tests for the backtest engine: position sizing, costs, and PnL aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.backtest.engine import BacktestEngine, signal_to_position
from ml_backtester.config import Config


def test_signal_to_position_is_cross_sectional_zscore_clipped():
    dates = pd.Index(["2020-01-01", "2020-01-01", "2020-01-01"])
    signal = pd.Series([1.0, 2.0, 3.0])
    config = Config(risk_cap=1.0)

    position = signal_to_position(signal, dates, config)

    # Mean signal is 2.0, so the middle row should land near zero exposure.
    assert position.iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert (position.abs() <= config.risk_cap + 1e-9).all()


def test_signal_to_position_zero_for_single_row_dates():
    dates = pd.Index(["2020-01-01", "2020-01-02"])
    signal = pd.Series([5.0, -3.0])
    config = Config()

    position = signal_to_position(signal, dates, config)

    assert (position == 0.0).all()


def test_signal_to_position_respects_risk_cap():
    dates = pd.Index(["2020-01-01"] * 4)
    signal = pd.Series([-100.0, -1.0, 1.0, 100.0])
    config = Config(risk_cap=0.5)

    position = signal_to_position(signal, dates, config)

    assert position.max() <= 0.5 + 1e-9
    assert position.min() >= -0.5 - 1e-9


def test_engine_zero_cost_zero_signal_gives_zero_return():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    test_data = pd.DataFrame({"ticker": ["AAA", "BBB"], "label": [0.05, -0.03]}, index=dates)
    signal = pd.Series([0.0, 0.0])
    config = Config(spread_bps=0.0, slippage_bps=0.0, use_market_impact=False)

    result = BacktestEngine(config).run(test_data, signal)

    assert result.returns.iloc[0] == pytest.approx(0.0)


def test_engine_return_reflects_signal_direction_and_costs():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    # AAA gets a positive signal, BBB an equal-and-opposite negative signal -> symmetric
    # long/short positions, so with an identical label the two legs' PnL cancels exactly.
    test_data = pd.DataFrame({"ticker": ["AAA", "BBB"], "label": [0.02, 0.02]}, index=dates)
    signal = pd.Series([2.0, -2.0])
    config = Config(spread_bps=0.0, slippage_bps=0.0, use_market_impact=False, risk_cap=1.0)

    result = BacktestEngine(config).run(test_data, signal)

    assert result.returns.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_engine_applies_transaction_costs():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    test_data = pd.DataFrame({"ticker": ["AAA", "BBB"], "label": [0.0, 0.0]}, index=dates)
    signal = pd.Series([1.0, -1.0])
    config = Config(spread_bps=10.0, slippage_bps=0.0, use_market_impact=False, risk_cap=1.0)

    result_with_cost = BacktestEngine(config).run(test_data, signal)
    result_no_cost = BacktestEngine(
        Config(spread_bps=0.0, slippage_bps=0.0, use_market_impact=False, risk_cap=1.0)
    ).run(test_data, signal)

    assert result_with_cost.returns.iloc[0] < result_no_cost.returns.iloc[0]


def test_engine_exposure_reflects_mean_absolute_position():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    test_data = pd.DataFrame({"ticker": ["AAA", "BBB"], "label": [0.0, 0.0]}, index=dates)
    signal = pd.Series([1.0, -1.0])
    config = Config(risk_cap=1.0)

    result = BacktestEngine(config).run(test_data, signal)

    # Two symmetric points z-score to +-1/sqrt(2) (ddof=1), well under the risk cap.
    assert result.exposure.iloc[0] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
