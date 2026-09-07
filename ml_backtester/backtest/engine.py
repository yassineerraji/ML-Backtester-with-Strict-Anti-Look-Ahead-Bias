"""In-house backtest engine: predicted signal -> capped position -> costs -> portfolio PnL."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_backtester.backtest.costs import total_transaction_cost
from ml_backtester.config import Config


@dataclass
class BacktestResult:
    """Output of one backtest run over a test fold."""

    returns: pd.Series
    """Net portfolio return per date, after costs, equal-weighted across the universe."""

    exposure: pd.Series
    """Mean absolute position size per date — turnover input for backtest.metrics.turnover."""


def signal_to_position(signal: pd.Series, dates: pd.Index, config: Config) -> pd.Series:
    """Convert raw predicted-return signals into capped, cross-sectionally comparable positions.

    Args:
        signal: Predicted forward return per row.
        dates: Observation date per row, positionally aligned to `signal`,
            used to z-score the signal cross-sectionally within each date.
        config: Pipeline configuration providing risk_cap.

    Returns:
        Position size (fraction of capital) per row: the signal, z-scored
        against same-date peers (so sizing reflects relative conviction,
        not raw prediction magnitude), then clipped to
        [-risk_cap, risk_cap]. Dates with a single row (no cross-sectional
        spread) get position 0.
    """
    frame = pd.DataFrame({"signal": signal.to_numpy(), "date": dates})
    grouped = frame.groupby("date")["signal"]
    z = (frame["signal"] - grouped.transform("mean")) / grouped.transform("std")
    position = z.fillna(0.0).clip(-config.risk_cap, config.risk_cap)
    position.index = signal.index
    return position


class BacktestEngine:
    """Turns per-(date, ticker) predicted signals and realized labels into portfolio PnL."""

    def __init__(self, config: Config):
        """Store the pipeline configuration used by every run() call."""
        self.config = config

    def run(self, test_data: pd.DataFrame, signal: pd.Series) -> BacktestResult:
        """Run the backtest over one test fold.

        Args:
            test_data: Test fold rows indexed by date with columns "ticker"
                and "label" (realized forward return over
                config.horizon_days).
            signal: Predicted forward return per row, positionally aligned
                to test_data.

        Returns:
            BacktestResult with the portfolio's net per-date return series
            and a turnover-proxy exposure series.
        """
        frame = test_data.copy()
        frame["position"] = signal_to_position(signal, frame.index, self.config).to_numpy()
        frame = frame.sort_index()

        frame["trade_size"] = (
            frame.groupby("ticker")["position"].diff().abs().fillna(frame["position"].abs())
        )
        frame["cost"] = total_transaction_cost(frame["trade_size"].to_numpy(), self.config)
        frame["net_return"] = frame["position"] * frame["label"] - frame["cost"]

        by_date = frame.groupby(level=0)
        returns = by_date["net_return"].mean()
        exposure = by_date["position"].apply(lambda s: s.abs().mean())
        return BacktestResult(returns=returns, exposure=exposure)
