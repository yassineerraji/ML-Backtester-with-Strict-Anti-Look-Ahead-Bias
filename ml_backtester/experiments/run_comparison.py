"""End-to-end experiment: run the naive and purged/embargoed walk-forward pipelines on
identical data, model, and costs, and compare the resulting Sharpe ratios — the project's
core deliverable (see README.md, "With/Without Bias Comparison").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_backtester.backtest.engine import BacktestEngine
from ml_backtester.backtest.metrics import (
    hit_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)
from ml_backtester.config import Config
from ml_backtester.data.loader import load_universe
from ml_backtester.features.engineering import compute_features
from ml_backtester.labeling.targets import add_labels
from ml_backtester.models.train import fit_predict
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter


@dataclass
class RunResult:
    """Aggregated out-of-sample results from one full walk-forward run."""

    returns: pd.Series
    """Concatenated out-of-sample portfolio returns across every fold's test block."""
    exposure: pd.Series
    """Concatenated turnover-proxy exposure series across every fold's test block."""
    metrics: dict[str, float]
    """sharpe, sortino, max_drawdown, turnover, hit_ratio computed over `returns`/`exposure`."""


def build_dataset(config: Config) -> pd.DataFrame:
    """Load prices, compute features and labels, and join them into one modeling dataset.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame indexed by date (t0) with columns "ticker", FEATURE_COLUMNS,
        "label", "t1" — one row per (date, ticker) with both a resolved
        feature vector and a resolved label. Rows present in only one of
        features/labels (rolling warm-up or label horizon truncation) are
        dropped.
    """
    prices = load_universe(config)
    features = compute_features(prices, config).reset_index()
    labels = add_labels(prices, config).reset_index()
    merged = features.merge(labels, on=["date", "ticker"], how="inner")
    return merged.set_index("date").sort_index()


def run_walk_forward(
    dataset: pd.DataFrame,
    splitter: WalkForwardSplitter | PurgedEmbargoedSplitter,
    config: Config,
) -> RunResult:
    """Run one full walk-forward backtest using the given splitter.

    Args:
        dataset: Output of build_dataset.
        splitter: A WalkForwardSplitter (naive) or PurgedEmbargoedSplitter
            (corrected) instance.
        config: Pipeline configuration.

    Returns:
        RunResult aggregating out-of-sample returns, exposure, and summary
        metrics across every fold.
    """
    engine = BacktestEngine(config)
    fold_returns = []
    fold_exposure = []

    for train_idx, test_idx in splitter.split(dataset, dataset["t1"]):
        train, test = dataset.iloc[train_idx], dataset.iloc[test_idx]
        if train.empty or test.empty:
            continue
        predictions = fit_predict(train, test, config)
        signal = pd.Series(predictions, index=test.index)
        result = engine.run(test, signal)
        fold_returns.append(result.returns)
        fold_exposure.append(result.exposure)

    returns = pd.concat(fold_returns).sort_index()
    exposure = pd.concat(fold_exposure).sort_index()
    metrics = {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover": turnover(exposure),
        "hit_ratio": hit_ratio(returns),
    }
    return RunResult(returns=returns, exposure=exposure, metrics=metrics)


def run_comparison(config: Config | None = None) -> dict[str, RunResult]:
    """Run both the naive and purged/embargoed walk-forward pipelines on identical data.

    Args:
        config: Pipeline configuration; defaults to Config(). Universe, dates,
            features, model, and costs are identical between the two runs —
            only the splitter differs (see CLAUDE.md, Temporal-Correctness
            Rules).

    Returns:
        Dict with keys "naive" and "corrected", each mapping to a RunResult.
    """
    config = config or Config()
    dataset = build_dataset(config)

    naive_splitter = WalkForwardSplitter(
        train_window=config.train_window_days,
        test_window=config.test_window_days,
        embargo=config.embargo_days,
    )
    corrected_splitter = PurgedEmbargoedSplitter(
        train_window=config.train_window_days,
        test_window=config.test_window_days,
        embargo=config.embargo_days,
    )

    return {
        "naive": run_walk_forward(dataset, naive_splitter, config),
        "corrected": run_walk_forward(dataset, corrected_splitter, config),
    }


def format_report(results: dict[str, RunResult]) -> str:
    """Render the naive-vs-corrected comparison as a Markdown report.

    Args:
        results: Output of run_comparison.

    Returns:
        Markdown text with a metrics table and the Sharpe gap between runs.
    """
    naive, corrected = results["naive"].metrics, results["corrected"].metrics
    gap = naive["sharpe"] - corrected["sharpe"]

    lines = [
        "# Naive vs. Corrected Walk-Forward Comparison",
        "",
        "| Metric | Naive (no purge/embargo) | Corrected (purged/embargoed) |",
        "| --- | --- | --- |",
    ]
    for key in ("sharpe", "sortino", "max_drawdown", "turnover", "hit_ratio"):
        lines.append(f"| {key} | {naive[key]:.4f} | {corrected[key]:.4f} |")
    lines += [
        "",
        f"**Sharpe gap (naive − corrected): {gap:.4f}**",
        "",
        "The naive Sharpe is inflated because training samples whose label "
        "window overlaps the test period leak test-period information into "
        "the model (no purge), and no embargo buffer removes residual "
        "autocorrelation across the train/test boundary.",
    ]
    return "\n".join(lines)


def main(config: Config | None = None, report_path: str = "reports/bias_comparison.md") -> None:
    """Run the naive-vs-corrected comparison end-to-end and write the Markdown report to disk.

    Args:
        config: Pipeline configuration; defaults to Config().
        report_path: Destination path for the Markdown report.
    """
    report = format_report(run_comparison(config))
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


if __name__ == "__main__":
    main()
