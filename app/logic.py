"""Pure, Streamlit-independent logic behind the UI: config building, validation, and the
data transformations each tab renders. Kept free of `streamlit` imports so it can be unit
tested directly, without booting the app.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ml_backtester.config import Config
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter


def build_config_from_form(overrides: dict) -> Config:
    """Build a Config from raw widget values, normalizing date widgets to ISO strings.

    Args:
        overrides: Config field values collected from UI widgets. "start_date"
            and "end_date" may be datetime.date objects (as returned by
            st.date_input) and are converted to "YYYY-MM-DD" strings; any
            already a string pass through unchanged.

    Returns:
        A Config instance ready to pass into build_dataset / run_walk_forward.
    """
    normalized = dict(overrides)
    for key in ("start_date", "end_date"):
        value = normalized.get(key)
        if isinstance(value, date):
            normalized[key] = value.isoformat()
    return Config(**normalized)


def validate_config(config: Config) -> list[str]:
    """Check a Config for values that would make the pipeline error out or be meaningless.

    Args:
        config: Candidate configuration, typically built from UI widget values.

    Returns:
        Human-readable problem descriptions; empty if the config looks runnable.
    """
    problems = []
    if not config.universe:
        problems.append("Universe must include at least one ticker.")
    if config.start_date >= config.end_date:
        problems.append("Start date must be before end date.")
    if config.horizon_days < 1:
        problems.append("Horizon must be at least 1 trading day.")
    if config.train_window_days < 1:
        problems.append("Train window must be at least 1 trading day.")
    if config.test_window_days < 1:
        problems.append("Test window must be at least 1 trading day.")
    if config.embargo_days < 0:
        problems.append("Embargo must be zero or more trading days.")
    if config.horizon_days >= config.train_window_days:
        problems.append("Horizon is >= train window: purge would remove the entire training block.")
    return problems


def format_metrics_table(naive_metrics: dict, corrected_metrics: dict) -> pd.DataFrame:
    """Combine naive and corrected metric dicts into one side-by-side display table.

    Args:
        naive_metrics: RunResult.metrics from the naive-splitter run.
        corrected_metrics: RunResult.metrics from the corrected-splitter run.

    Returns:
        DataFrame indexed by metric name with "Naive" and "Corrected" columns.
    """
    return pd.DataFrame({"Naive": naive_metrics, "Corrected": corrected_metrics})


def cumulative_equity(returns: pd.Series) -> pd.Series:
    """Convert a per-period return series into a cumulative equity curve starting at 1.0."""
    return (1 + returns).cumprod()


def compute_fold_timeline(
    dataset: pd.DataFrame,
    splitter: WalkForwardSplitter | PurgedEmbargoedSplitter,
    variant: str,
) -> pd.DataFrame:
    """Compute per-fold train/purged/test/embargo date segments for a timeline chart.

    Args:
        dataset: Output of build_dataset — indexed by date, with a "t1" column.
        splitter: A WalkForwardSplitter (naive) or PurgedEmbargoedSplitter
            (corrected) instance.
        variant: Label ("naive" or "corrected") stored in the output for display.

    Returns:
        Long-format DataFrame with columns fold, variant, segment
        ("train", "purged", "test", "embargo"), start, end — one row per
        segment present in that fold. "purged" segments mark training dates
        whose label window overlaps the test block; they're computed for
        every splitter (so a naive splitter's un-purged leak is visible
        too), even though only PurgedEmbargoedSplitter actually excludes
        them from training.
    """
    dates = dataset.index.unique().sort_values()
    date_t1 = dataset.groupby(level=0)["t1"].first()

    blocks = list(splitter.blocks(dates))
    rows = []
    for fold_idx, (train_dates, test_dates) in enumerate(blocks):
        test_start = pd.Timestamp(test_dates[0])
        purged_mask = (date_t1.loc[train_dates] >= test_start).to_numpy()
        kept_dates = train_dates[~purged_mask]
        purged_dates = train_dates[purged_mask]

        if len(kept_dates) > 0:
            rows.append(
                {
                    "fold": fold_idx,
                    "variant": variant,
                    "segment": "train",
                    "start": kept_dates.min(),
                    "end": kept_dates.max(),
                }
            )
        if len(purged_dates) > 0:
            rows.append(
                {
                    "fold": fold_idx,
                    "variant": variant,
                    "segment": "purged",
                    "start": purged_dates.min(),
                    "end": purged_dates.max(),
                }
            )
        rows.append(
            {
                "fold": fold_idx,
                "variant": variant,
                "segment": "test",
                "start": test_dates.min(),
                "end": test_dates.max(),
            }
        )

        if fold_idx + 1 < len(blocks):
            next_train_dates = blocks[fold_idx + 1][0]
            embargo_start_pos = dates.get_loc(test_dates.max()) + 1
            embargo_end_pos = dates.get_loc(next_train_dates.min()) - 1
            if embargo_end_pos >= embargo_start_pos:
                rows.append(
                    {
                        "fold": fold_idx,
                        "variant": variant,
                        "segment": "embargo",
                        "start": dates[embargo_start_pos],
                        "end": dates[embargo_end_pos],
                    }
                )

    return pd.DataFrame(rows)


def grid_from_editor_dataframe(df: pd.DataFrame) -> list[dict]:
    """Convert the hyperparameter-grid editor's DataFrame into LightGBM kwarg dicts.

    Args:
        df: Editable grid with one column per hyperparameter, one row per trial
            (as produced by st.data_editor).

    Returns:
        List of {column: value} dicts, one per non-empty row. Whole-number
        float columns are cast back to int, since st.data_editor represents
        numeric columns as floats even when every value is integral.
    """
    records = []
    for raw_row in df.to_dict("records"):
        row = {
            key: value
            for key, value in raw_row.items()
            if value is not None and not (isinstance(value, float) and pd.isna(value))
        }
        if not row:
            continue
        row = {
            key: (int(value) if isinstance(value, float) and value.is_integer() else value)
            for key, value in row.items()
        }
        records.append(row)
    return records
