"""Tests for the deflated Sharpe ratio: hand-checked formula and expected monotonicity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from ml_backtester.backtest.deflation import _expected_max_sharpe_under_null, deflated_sharpe_ratio


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    assert _expected_max_sharpe_under_null(n_trials=1, sharpe_std=1.0) == 0.0


def test_expected_max_sharpe_increases_with_more_trials():
    small = _expected_max_sharpe_under_null(n_trials=5, sharpe_std=1.0)
    large = _expected_max_sharpe_under_null(n_trials=500, sharpe_std=1.0)

    assert large > small > 0


def test_deflated_sharpe_matches_hand_computation_for_single_trial():
    observed_sharpe = 1.5
    n_observations = 252

    result = deflated_sharpe_ratio(
        observed_sharpe, n_trials=1, n_observations=n_observations, skewness=0.0, kurtosis=3.0
    )

    denominator = np.sqrt(1 - 0.0 * observed_sharpe + (3.0 - 1) / 4 * observed_sharpe**2)
    expected = norm.cdf(observed_sharpe * np.sqrt(n_observations - 1) / denominator)
    assert result == pytest.approx(expected)


def test_more_trials_deflates_the_same_observed_sharpe():
    kwargs = dict(observed_sharpe=1.2, n_observations=252, skewness=0.0, kurtosis=3.0)

    few_trials = deflated_sharpe_ratio(n_trials=1, **kwargs)
    many_trials = deflated_sharpe_ratio(n_trials=1000, **kwargs)

    assert many_trials < few_trials


def test_deflated_sharpe_is_a_valid_probability():
    result = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=50, n_observations=500)
    assert 0.0 <= result <= 1.0


def test_higher_observed_sharpe_increases_deflated_sharpe():
    kwargs = dict(n_trials=20, n_observations=252, skewness=0.0, kurtosis=3.0)

    low = deflated_sharpe_ratio(observed_sharpe=0.5, **kwargs)
    high = deflated_sharpe_ratio(observed_sharpe=2.0, **kwargs)

    assert high > low
