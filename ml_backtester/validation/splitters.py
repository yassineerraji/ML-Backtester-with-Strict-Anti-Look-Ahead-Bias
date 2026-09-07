"""Sequential walk-forward splitters: naive (biased) and purged/embargoed (corrected).

Both follow the same train -> test cycle laid out in README.md's diagram:
train window -> [purge overlapping labels] -> test window -> [embargo buffer]
-> next train window. The two classes share one constructor/split()
signature so ml_backtester.experiments.run_comparison can swap between them
with a single argument while everything else in the pipeline stays fixed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """Naive sequential walk-forward splitter: no purge, no embargo.

    Deliberately biased baseline: training observations whose label window
    overlaps the following test period are left in, and the next training
    block starts immediately at the end of the test block. Used to
    demonstrate the Sharpe inflation caused by skipping purge/embargo.
    """

    def __init__(self, train_window: int, test_window: int, embargo: int = 0):
        """
        Args:
            train_window: Number of unique observation dates per training block.
            test_window: Number of unique observation dates per test block.
            embargo: Accepted for interface parity with PurgedEmbargoedSplitter;
                ignored here — the naive splitter applies no embargo gap.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.embargo = embargo

    def blocks(
        self, dates: pd.DatetimeIndex
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield successive (train_dates, test_dates) blocks with no embargo gap between them."""
        start = 0
        n = len(dates)
        while start + self.train_window + self.test_window <= n:
            train_dates = dates[start : start + self.train_window]
            test_dates = dates[
                start + self.train_window : start + self.train_window + self.test_window
            ]
            yield train_dates, test_dates
            start += self.train_window + self.test_window

    def split(self, X: pd.DataFrame, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) positional index arrays into X.

        Args:
            X: Feature matrix indexed by observation date (t0); may hold
                multiple rows per date (one per ticker).
            t1: Label end-time per row, positionally aligned to X.index.
                Unused by the naive splitter; accepted for interface parity
                with PurgedEmbargoedSplitter.

        Returns:
            Iterator of (train_idx, test_idx) positional integer arrays.
        """
        dates = X.index.unique().sort_values()
        for train_dates, test_dates in self.blocks(dates):
            train_idx = np.flatnonzero(X.index.isin(train_dates))
            test_idx = np.flatnonzero(X.index.isin(test_dates))
            yield train_idx, test_idx


class PurgedEmbargoedSplitter(WalkForwardSplitter):
    """Sequential walk-forward splitter with purge and embargo applied.

    Prevents label leakage across the train/test boundary caused by
    overlapping label windows and residual autocorrelation:
      - Purge: training rows whose label window [t0, t1] overlaps the
        following test block (t1 >= test start) are dropped from the train
        fold.
      - Embargo: after each test block, `embargo` dates are skipped before
        the next training block is allowed to begin.
    """

    def blocks(
        self, dates: pd.DatetimeIndex
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield successive (train_dates, test_dates) blocks separated by an embargo gap."""
        start = 0
        n = len(dates)
        while start + self.train_window + self.test_window <= n:
            train_dates = dates[start : start + self.train_window]
            test_dates = dates[
                start + self.train_window : start + self.train_window + self.test_window
            ]
            yield train_dates, test_dates
            start += self.train_window + self.test_window + self.embargo

    def split(self, X: pd.DataFrame, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) positional index arrays into X, purged and embargoed.

        Args:
            X: Feature matrix indexed by observation date (t0); may hold
                multiple rows per date (one per ticker).
            t1: Label end-time per row, positionally aligned to X.index —
                the purge boundary is computed from these timestamps, never
                from row position or count.

        Returns:
            Iterator of (train_idx, test_idx) positional integer arrays.
            Train indices exclude any row whose t1 falls at or after the
            test block's first date.
        """
        dates = X.index.unique().sort_values()
        t1_values = t1.to_numpy()
        for train_dates, test_dates in self.blocks(dates):
            test_start = np.datetime64(test_dates[0])
            in_train_block = X.index.isin(train_dates)
            not_purged = t1_values < test_start
            train_idx = np.flatnonzero(in_train_block & not_purged)
            test_idx = np.flatnonzero(X.index.isin(test_dates))
            yield train_idx, test_idx
