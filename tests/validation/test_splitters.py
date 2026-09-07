"""Tests asserting the temporal invariants of the splitters directly, not just that they run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter

HORIZON = 3
TRAIN_WINDOW = 10
TEST_WINDOW = 5
EMBARGO = 4


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=60, freq="B", name="date")


@pytest.fixture
def X(dates) -> pd.DataFrame:
    return pd.DataFrame({"val": range(len(dates))}, index=dates)


@pytest.fixture
def t1(dates) -> pd.Series:
    """Label end-time horizon_days after each t0 — may extrapolate past `dates`, which is fine
    since only relative ordering to test-block boundaries is checked."""
    return pd.Series(dates + pd.offsets.BDay(HORIZON), index=dates)


def test_naive_splitter_does_not_purge_overlapping_labels(X, t1):
    splitter = WalkForwardSplitter(TRAIN_WINDOW, TEST_WINDOW)
    train_idx, test_idx = next(splitter.split(X, t1))

    assert len(train_idx) == TRAIN_WINDOW
    assert len(test_idx) == TEST_WINDOW


def test_purged_splitter_removes_overlapping_labels(X, t1):
    splitter = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, embargo=0)
    train_idx, test_idx = next(splitter.split(X, t1))

    assert len(train_idx) == TRAIN_WINDOW - HORIZON
    test_start = X.index[TRAIN_WINDOW]
    assert (t1.iloc[train_idx].to_numpy() < np.datetime64(test_start)).all()


def test_purged_splitter_never_leaves_a_leaking_row(X, t1):
    """No train row in any fold may have a label resolving at/after that fold's test start."""
    splitter = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO)
    for train_idx, test_idx in splitter.split(X, t1):
        test_start = X.index[test_idx[0]]
        assert (t1.iloc[train_idx].to_numpy() < np.datetime64(test_start)).all()


def test_embargo_gap_between_test_and_next_train(dates):
    splitter = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO)
    blocks = list(splitter.blocks(dates))

    assert len(blocks) >= 2
    _, test_dates_0 = blocks[0]
    train_dates_1, _ = blocks[1]

    gap_days = dates.get_loc(train_dates_1[0]) - dates.get_loc(test_dates_0[-1])
    assert gap_days == EMBARGO + 1


def test_naive_and_purged_share_call_signature(X, t1):
    naive = WalkForwardSplitter(TRAIN_WINDOW, TEST_WINDOW, EMBARGO)
    purged = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, EMBARGO)

    for splitter in (naive, purged):
        train_idx, test_idx = next(splitter.split(X, t1))
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)


def test_splits_never_shuffle_order(X, t1):
    splitter = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, embargo=EMBARGO)
    for train_idx, test_idx in splitter.split(X, t1):
        assert (np.diff(train_idx) > 0).all()
        assert (np.diff(test_idx) > 0).all()
        assert train_idx.max() < test_idx.min()


def test_multi_ticker_rows_are_split_by_date_not_row_count(dates):
    two_tickers = pd.concat(
        [
            pd.DataFrame({"val": range(len(dates)), "ticker": "AAA"}, index=dates),
            pd.DataFrame({"val": range(len(dates)), "ticker": "BBB"}, index=dates),
        ]
    ).sort_index()
    t1_full = (two_tickers.index + pd.offsets.BDay(HORIZON)).to_series(index=two_tickers.index)

    splitter = PurgedEmbargoedSplitter(TRAIN_WINDOW, TEST_WINDOW, embargo=0)
    train_idx, test_idx = next(splitter.split(two_tickers, t1_full))

    # Both tickers' rows for each in-window date should be present (2 rows/date).
    assert len(test_idx) == TEST_WINDOW * 2
