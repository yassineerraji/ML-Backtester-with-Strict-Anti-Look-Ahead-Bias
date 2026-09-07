"""Pure performance metrics: no I/O, no hidden state, just (returns, ...) -> float."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of a return series.

    Args:
        returns: Per-period returns.
        periods_per_year: Number of periods per year, for annualization.
        risk_free: Per-period risk-free rate, subtracted before annualizing.

    Returns:
        Annualized Sharpe ratio, or NaN if returns has zero variance.
    """
    excess = returns - risk_free
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio: like Sharpe, but penalizing only downside deviation.

    Args:
        returns: Per-period returns.
        periods_per_year: Number of periods per year, for annualization.
        risk_free: Per-period risk-free rate, subtracted before annualizing.

    Returns:
        Annualized Sortino ratio, or NaN if there is no downside deviation.
    """
    excess = returns - risk_free
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the equity curve implied by `returns`.

    Args:
        returns: Per-period returns.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.23 for a 23% drawdown).
    """
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def turnover(positions: pd.Series) -> float:
    """Average per-period absolute position change, a proxy for trading activity.

    Args:
        positions: Position size (fraction of capital) per period.

    Returns:
        Mean absolute change in position between consecutive periods.
    """
    return float(positions.diff().abs().mean())


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return.

    Args:
        returns: Per-period returns.

    Returns:
        Proportion of periods where returns > 0, in [0, 1].
    """
    return float((returns > 0).mean())
