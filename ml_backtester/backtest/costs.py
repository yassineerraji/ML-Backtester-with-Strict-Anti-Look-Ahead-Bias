"""Transaction cost models: fixed spread/slippage, plus optional square-root-law market impact."""

from __future__ import annotations

import numpy as np

from ml_backtester.config import Config


def spread_slippage_cost(trade_size: np.ndarray, config: Config) -> np.ndarray:
    """Fixed per-trade cost from bid/ask spread and slippage, proportional to trade size.

    Args:
        trade_size: Absolute change in position (as a fraction of capital)
            for each trade, e.g. |position[t] - position[t-1]|.
        config: Pipeline configuration providing spread_bps and slippage_bps.

    Returns:
        Cost per trade, in the same units as trade_size (fraction of capital).
    """
    bps = config.spread_bps + config.slippage_bps
    return trade_size * (bps / 10_000)


def market_impact_cost(trade_size: np.ndarray, config: Config) -> np.ndarray:
    """Square-root-law market impact cost: cost grows with the square root of trade size.

    Args:
        trade_size: Absolute change in position (as a fraction of capital)
            for each trade.
        config: Pipeline configuration providing impact_coefficient.

    Returns:
        Impact cost per trade, in the same units as trade_size, or an
        all-zero array if config.use_market_impact is False.
    """
    if not config.use_market_impact:
        return np.zeros_like(trade_size, dtype=float)
    return config.impact_coefficient * np.sqrt(trade_size)


def total_transaction_cost(trade_size: np.ndarray, config: Config) -> np.ndarray:
    """Combine spread/slippage and (optional) market impact into one per-trade cost.

    Args:
        trade_size: Absolute change in position (as a fraction of capital)
            for each trade.
        config: Pipeline configuration providing cost parameters.

    Returns:
        Total cost per trade, in the same units as trade_size.
    """
    return spread_slippage_cost(trade_size, config) + market_impact_cost(trade_size, config)
