"""Tests for the UI's pure logic layer — no streamlit import here, by design."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.logic import (
    build_config_from_form,
    compute_fold_timeline,
    cumulative_equity,
    format_metrics_table,
    grid_from_editor_dataframe,
    validate_config,
)
from ml_backtester.config import Config
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter


def test_build_config_from_form_converts_date_objects_to_iso_strings():
    config = build_config_from_form(
        {"start_date": date(2020, 1, 1), "end_date": date(2021, 6, 15), "horizon_days": 7}
    )

    assert config.start_date == "2020-01-01"
    assert config.end_date == "2021-06-15"
    assert config.horizon_days == 7


def test_build_config_from_form_passes_through_string_dates_unchanged():
    config = build_config_from_form({"start_date": "2019-03-01", "end_date": "2020-03-01"})

    assert config.start_date == "2019-03-01"
    assert config.end_date == "2020-03-01"


def test_build_config_from_form_passes_through_other_fields():
    config = build_config_from_form(
        {"universe": ["AAPL", "MSFT"], "risk_cap": 0.5, "model_params": {"num_leaves": 7}}
    )

    assert config.universe == ["AAPL", "MSFT"]
    assert config.risk_cap == 0.5
    assert config.model_params == {"num_leaves": 7}


def test_validate_config_accepts_the_default_config():
    assert validate_config(Config()) == []


def test_validate_config_flags_empty_universe():
    problems = validate_config(Config(universe=[]))
    assert any("universe" in p.lower() for p in problems)


def test_validate_config_flags_start_after_end():
    problems = validate_config(Config(start_date="2022-01-01", end_date="2021-01-01"))
    assert any("start date" in p.lower() for p in problems)


def test_validate_config_flags_horizon_not_smaller_than_train_window():
    problems = validate_config(Config(horizon_days=30, train_window_days=30))
    assert any("horizon" in p.lower() for p in problems)


def test_validate_config_flags_nonpositive_windows():
    problems = validate_config(Config(train_window_days=0, test_window_days=0, embargo_days=-1))
    assert len(problems) >= 3


def test_format_metrics_table_has_naive_and_corrected_columns():
    table = format_metrics_table(
        {"sharpe": 1.0, "hit_ratio": 0.5}, {"sharpe": 0.5, "hit_ratio": 0.4}
    )

    assert list(table.columns) == ["Naive", "Corrected"]
    assert table.loc["sharpe", "Naive"] == 1.0
    assert table.loc["sharpe", "Corrected"] == 0.5


def test_cumulative_equity_matches_hand_computation():
    returns = pd.Series([0.1, -0.1, 0.05])

    equity = cumulative_equity(returns)

    expected = [1.1, 1.1 * 0.9, 1.1 * 0.9 * 1.05]
    np.testing.assert_allclose(equity.to_numpy(), expected)


def test_grid_from_editor_dataframe_casts_whole_number_floats_to_int():
    df = pd.DataFrame({"num_leaves": [7.0, 31.0], "max_depth": [2.0, 5.0]})

    records = grid_from_editor_dataframe(df)

    assert records == [{"num_leaves": 7, "max_depth": 2}, {"num_leaves": 31, "max_depth": 5}]


def test_grid_from_editor_dataframe_keeps_non_integer_floats():
    df = pd.DataFrame({"learning_rate": [0.05, 0.1]})

    records = grid_from_editor_dataframe(df)

    assert records == [{"learning_rate": 0.05}, {"learning_rate": 0.1}]


def test_grid_from_editor_dataframe_drops_fully_empty_rows():
    df = pd.DataFrame({"num_leaves": [7.0, np.nan], "max_depth": [2.0, np.nan]})

    records = grid_from_editor_dataframe(df)

    assert records == [{"num_leaves": 7, "max_depth": 2}]


def test_grid_from_editor_dataframe_drops_only_nan_fields_from_partial_rows():
    df = pd.DataFrame({"num_leaves": [7.0, 31.0], "max_depth": [2.0, np.nan]})

    records = grid_from_editor_dataframe(df)

    assert records == [{"num_leaves": 7, "max_depth": 2}, {"num_leaves": 31}]


def _synthetic_dataset(n_dates: int, tickers: list[str], horizon: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B", name="date")
    t1_by_date = pd.Series(dates + pd.offsets.BDay(horizon), index=dates)
    frames = []
    for ticker in tickers:
        frames.append(pd.DataFrame({"ticker": ticker, "t1": t1_by_date}, index=dates))
    return pd.concat(frames).sort_index()


def test_compute_fold_timeline_marks_purged_dates_for_corrected_splitter():
    horizon = 3
    dataset = _synthetic_dataset(60, ["AAA", "BBB"], horizon)
    splitter = PurgedEmbargoedSplitter(train_window=10, test_window=5, embargo=2)

    timeline = compute_fold_timeline(dataset, splitter, "corrected")

    assert set(timeline["variant"]) == {"corrected"}
    assert set(timeline["segment"]).issubset({"train", "purged", "test", "embargo"})
    fold0 = timeline[timeline["fold"] == 0]
    purged_rows = fold0[fold0["segment"] == "purged"]
    assert len(purged_rows) == 1
    # horizon=3 trading dates purged out of a 10-day train block (business-day count, since
    # calendar-day span between start/end varies depending on whether it crosses a weekend).
    n_purged_dates = len(pd.bdate_range(purged_rows["start"].iloc[0], purged_rows["end"].iloc[0]))
    assert n_purged_dates == horizon


def test_compute_fold_timeline_naive_has_no_embargo_segment():
    horizon = 3
    dataset = _synthetic_dataset(60, ["AAA"], horizon)
    splitter = WalkForwardSplitter(train_window=10, test_window=5, embargo=2)

    timeline = compute_fold_timeline(dataset, splitter, "naive")

    assert "embargo" not in set(timeline["segment"])


def test_compute_fold_timeline_corrected_has_embargo_segment_when_folds_allow():
    horizon = 3
    dataset = _synthetic_dataset(60, ["AAA"], horizon)
    splitter = PurgedEmbargoedSplitter(train_window=10, test_window=5, embargo=2)

    timeline = compute_fold_timeline(dataset, splitter, "corrected")

    assert "embargo" in set(timeline["segment"])
    embargo_rows = timeline[timeline["segment"] == "embargo"]
    assert (embargo_rows["end"] >= embargo_rows["start"]).all()
