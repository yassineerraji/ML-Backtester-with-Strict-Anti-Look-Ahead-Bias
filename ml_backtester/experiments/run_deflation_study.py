"""Multi-configuration hyperparameter sweep with deflated Sharpe ratio correction.

Reporting the best Sharpe found across several model configurations
overstates skill: some configurations look good by chance alone, and that
chance component grows with the number of configurations tried. This
experiment runs the corrected (purged/embargoed) pipeline once per
configuration in MODEL_HYPERPARAMETER_GRID and reports both the best raw
Sharpe and its deflated counterpart — see ml_backtester.backtest.deflation
and README.md's Documentation Checklist.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from scipy.stats import kurtosis, skew

from ml_backtester.backtest.deflation import deflated_sharpe_ratio
from ml_backtester.backtest.metrics import sharpe_ratio
from ml_backtester.config import Config
from ml_backtester.experiments.run_comparison import build_dataset, run_walk_forward
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter

MODEL_HYPERPARAMETER_GRID: list[dict] = [
    {"num_leaves": 7, "max_depth": 2},
    {"num_leaves": 15, "max_depth": 3},
    {"num_leaves": 31, "max_depth": 5},
    {"num_leaves": 63, "max_depth": 7},
    {"num_leaves": 15, "max_depth": 7},
]


@dataclass
class ConfigTrialResult:
    """One hyperparameter configuration's out-of-sample performance."""

    hyperparameters: dict
    annualized_sharpe: float
    """Sharpe ratio annualized for readability; ranking-equivalent to per_period_sharpe."""
    per_period_sharpe: float
    """Non-annualized per-period Sharpe ratio — the unit deflation math is defined in."""
    returns: pd.Series


def run_config_sweep(
    base_config: Config, hyperparameter_grid: list[dict] = MODEL_HYPERPARAMETER_GRID
) -> list[ConfigTrialResult]:
    """Run the corrected (purged/embargoed) walk-forward pipeline once per hyperparameter set.

    Args:
        base_config: Pipeline configuration shared by every trial (universe,
            dates, features, splitter windows, costs) — only model_params
            varies across trials.
        hyperparameter_grid: Candidate LightGBM hyperparameter dicts to try.

    Returns:
        One ConfigTrialResult per grid entry, in grid order. All trials
        share the same dataset and splitter, computed once.
    """
    dataset = build_dataset(base_config)
    splitter = PurgedEmbargoedSplitter(
        base_config.train_window_days, base_config.test_window_days, base_config.embargo_days
    )

    results = []
    for hyperparameters in hyperparameter_grid:
        trial_config = replace(base_config, model_params=hyperparameters)
        run_result = run_walk_forward(dataset, splitter, trial_config)
        results.append(
            ConfigTrialResult(
                hyperparameters=hyperparameters,
                annualized_sharpe=run_result.metrics["sharpe"],
                per_period_sharpe=sharpe_ratio(run_result.returns, periods_per_year=1),
                returns=run_result.returns,
            )
        )
    return results


def compute_deflated_sharpe(trials: list[ConfigTrialResult]) -> tuple[ConfigTrialResult, float]:
    """Identify the best trial and compute its deflated Sharpe ratio.

    Args:
        trials: Output of run_config_sweep.

    Returns:
        (best_trial, deflated_sharpe) — best_trial by annualized_sharpe, and
        the probability its true Sharpe exceeds what chance alone would
        produce given len(trials) independent configurations were tried.
    """
    best = max(trials, key=lambda t: t.annualized_sharpe)

    per_period_sharpes = pd.Series([t.per_period_sharpe for t in trials])
    sharpe_std = per_period_sharpes.std()
    if pd.isna(sharpe_std) or sharpe_std == 0:
        sharpe_std = 1.0

    deflated = deflated_sharpe_ratio(
        observed_sharpe=best.per_period_sharpe,
        n_trials=len(trials),
        n_observations=len(best.returns),
        skewness=float(skew(best.returns)),
        kurtosis=float(kurtosis(best.returns, fisher=False)),
        sharpe_std=float(sharpe_std),
    )
    return best, deflated


def format_report(trials: list[ConfigTrialResult], best: ConfigTrialResult, deflated: float) -> str:
    """Render the sweep results and deflated Sharpe ratio as a Markdown report.

    Args:
        trials: All trial results, in grid order.
        best: The best trial by annualized Sharpe.
        deflated: Deflated Sharpe ratio of `best`.

    Returns:
        Markdown text with a per-trial table and the deflation result.
    """
    lines = [
        "# Hyperparameter Sweep — Deflated Sharpe Ratio",
        "",
        f"{len(trials)} configurations tested on identical data, splitter, and costs "
        "(purged/embargoed walk-forward); only model hyperparameters vary.",
        "",
        "| Hyperparameters | Annualized Sharpe |",
        "| --- | --- |",
    ]
    for trial in trials:
        lines.append(f"| {trial.hyperparameters} | {trial.annualized_sharpe:.4f} |")
    lines += [
        "",
        f"**Best observed Sharpe: {best.annualized_sharpe:.4f}** ({best.hyperparameters})",
        f"**Deflated Sharpe ratio: {deflated:.4f}**",
        "",
        f"The deflated Sharpe ratio is the probability the best trial's true Sharpe exceeds "
        f"the Sharpe an average skill-less strategy would achieve by chance, given "
        f"{len(trials)} configurations were tried. A value near 0.5 means the best-observed "
        "result is indistinguishable from noise; a value near 1.0 means it is unlikely to be "
        "a product of the search itself.",
    ]
    return "\n".join(lines)


def main(
    base_config: Config | None = None, report_path: str = "reports/deflation_study.md"
) -> None:
    """Run the hyperparameter sweep end-to-end and write the Markdown report to disk.

    Args:
        base_config: Pipeline configuration; defaults to Config().
        report_path: Destination path for the Markdown report.
    """
    base_config = base_config or Config()
    trials = run_config_sweep(base_config)
    best, deflated = compute_deflated_sharpe(trials)
    report = format_report(trials, best, deflated)

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


if __name__ == "__main__":
    main()
