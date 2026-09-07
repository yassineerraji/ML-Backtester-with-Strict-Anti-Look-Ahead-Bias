"""Integration tests for the end-to-end naive-vs-corrected comparison, no network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.config import Config
from ml_backtester.experiments import run_comparison as run_comparison_module
from ml_backtester.experiments.run_comparison import (
    RunResult,
    build_dataset,
    format_report,
    run_comparison,
    run_walk_forward,
)
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter


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


def test_build_dataset_joins_features_and_labels_on_date_and_ticker(small_config):
    dataset = build_dataset(small_config)

    assert set(
        ["ticker", "momentum", "volatility", "relative_volume", "rsi", "label", "t1"]
    ).issubset(dataset.columns)
    assert not dataset.isna().any().any()
    assert set(dataset["ticker"].unique()) == set(small_config.universe)


def test_run_walk_forward_produces_folds_and_metrics(small_config):
    dataset = build_dataset(small_config)
    splitter = PurgedEmbargoedSplitter(
        small_config.train_window_days, small_config.test_window_days, small_config.embargo_days
    )

    result = run_walk_forward(dataset, splitter, small_config)

    assert isinstance(result, RunResult)
    assert len(result.returns) > 0
    assert set(result.metrics.keys()) == {
        "sharpe",
        "sortino",
        "max_drawdown",
        "turnover",
        "hit_ratio",
    }


def test_run_comparison_returns_naive_and_corrected(small_config):
    results = run_comparison(small_config)

    assert set(results.keys()) == {"naive", "corrected"}
    for result in results.values():
        assert isinstance(result, RunResult)
        assert len(result.returns) > 0


def test_naive_splitter_uses_more_or_equal_training_rows_than_purged(small_config):
    """The naive splitter never purges, so its first-fold train set can't be smaller."""
    dataset = build_dataset(small_config)
    naive = WalkForwardSplitter(
        small_config.train_window_days, small_config.test_window_days, small_config.embargo_days
    )
    corrected = PurgedEmbargoedSplitter(
        small_config.train_window_days, small_config.test_window_days, small_config.embargo_days
    )

    naive_train_idx, _ = next(naive.split(dataset, dataset["t1"]))
    corrected_train_idx, _ = next(corrected.split(dataset, dataset["t1"]))

    assert len(naive_train_idx) >= len(corrected_train_idx)


def test_format_report_contains_metrics_table_and_gap(small_config):
    results = run_comparison(small_config)

    report = format_report(results)

    assert "Naive vs. Corrected" in report
    assert "Sharpe gap" in report
    assert "sharpe" in report
