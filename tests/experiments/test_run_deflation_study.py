"""Integration tests for the hyperparameter sweep and deflated Sharpe ratio, no network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.config import Config
from ml_backtester.experiments import run_comparison as run_comparison_module
from ml_backtester.experiments.run_deflation_study import (
    ConfigTrialResult,
    compute_deflated_sharpe,
    format_report,
    run_config_sweep,
)

# n_estimators (not num_leaves/max_depth) so the difference survives LightGBM's
# min_child_samples default on this test's tiny synthetic dataset.
SMALL_GRID = [{"n_estimators": 3}, {"n_estimators": 60}]


def _synthetic_universe_prices(n_days: int, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n_days, freq="B", name="date")
    frames = []
    for ticker in tickers:
        close = 100 + np.cumsum(rng.normal(0, 1, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": rng.integers(1000, 5000, n_days),
                    "ticker": ticker,
                },
                index=index,
            )
        )
    return pd.concat(frames).sort_index()


@pytest.fixture
def small_config() -> Config:
    return Config(
        universe=["AAA", "BBB", "CCC"],
        momentum_window=5,
        volatility_window=5,
        relative_volume_window=5,
        rsi_window=5,
        normalization_window=10,
        horizon_days=3,
        train_window_days=25,
        test_window_days=8,
        embargo_days=2,
    )


@pytest.fixture(autouse=True)
def patch_load_universe(monkeypatch, small_config):
    prices = _synthetic_universe_prices(150, small_config.universe)
    monkeypatch.setattr(run_comparison_module, "load_universe", lambda config: prices)


def test_run_config_sweep_produces_one_result_per_grid_entry(small_config):
    trials = run_config_sweep(small_config, SMALL_GRID)

    assert len(trials) == len(SMALL_GRID)
    assert all(isinstance(t, ConfigTrialResult) for t in trials)
    assert [t.hyperparameters for t in trials] == SMALL_GRID


def test_run_config_sweep_varies_returns_across_hyperparameters(small_config):
    trials = run_config_sweep(small_config, SMALL_GRID)

    assert not trials[0].returns.equals(trials[1].returns)


def test_compute_deflated_sharpe_selects_best_by_annualized_sharpe(small_config):
    trials = run_config_sweep(small_config, SMALL_GRID)

    best, deflated = compute_deflated_sharpe(trials)

    assert best.annualized_sharpe == max(t.annualized_sharpe for t in trials)
    assert np.isfinite(deflated)


def test_more_trials_does_not_increase_deflated_sharpe_for_the_same_best_trial(small_config):
    """Adding more (weaker) trials should not make the same best result look more significant."""
    two_trials = run_config_sweep(small_config, SMALL_GRID)
    _, deflated_two = compute_deflated_sharpe(two_trials)

    five_trials = run_config_sweep(small_config, SMALL_GRID * 3)  # same best trial repeated
    _, deflated_five = compute_deflated_sharpe(five_trials)

    assert deflated_five <= deflated_two + 1e-9


def test_format_report_contains_key_sections(small_config):
    trials = run_config_sweep(small_config, SMALL_GRID)
    best, deflated = compute_deflated_sharpe(trials)

    report = format_report(trials, best, deflated)

    assert "Deflated Sharpe Ratio" in report
    assert "Best observed Sharpe" in report
    assert str(len(trials)) in report
