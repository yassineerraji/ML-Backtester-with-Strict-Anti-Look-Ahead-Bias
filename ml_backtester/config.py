"""Central, immutable configuration for the pipeline: universe, dates, horizon, costs, seeds."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_UNIVERSE: list[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "V",
    "PG",
    "UNH",
    "HD",
    "XOM",
    "KO",
    "PEP",
    "CSCO",
]


@dataclass(frozen=True)
class Config:
    """Single source of truth for every tunable parameter in the pipeline.

    Passed explicitly through the pipeline (data -> features -> labels ->
    validation -> model -> backtest) rather than read from globals, so a run
    is fully reproducible from one object.
    """

    # Universe and date range
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # Labeling
    horizon_days: int = 5
    """Forward-looking window, in trading days, over which the return label is computed."""

    # Feature windows (all trailing/rolling, never expanding to the full series)
    momentum_window: int = 20
    volatility_window: int = 20
    relative_volume_window: int = 20
    rsi_window: int = 14
    normalization_window: int = 60
    """Rolling window for standardizing features (mean/std over this trailing window only)."""

    # Walk-forward validation
    train_window_days: int = 30
    test_window_days: int = 10
    embargo_days: int = 5
    """Buffer, in trading days, after each test window before the next training window resumes."""
    # train_window_days is deliberately short relative to a 3-year institutional retrain
    # cadence: purge removes horizon_days / train_window_days of each fold's training data
    # (5/30 ~ 17% here). With a multi-year train window that fraction is negligible and the
    # naive-vs-corrected Sharpe gap this project exists to demonstrate nearly disappears —
    # see reports/bias_comparison.md and reports/purge_embargo.md.

    # Backtest
    risk_cap: float = 1.0
    """Maximum absolute position size per asset, as a fraction of capital."""
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    use_market_impact: bool = False
    impact_coefficient: float = 0.1
    """Square-root-law market impact coefficient; only used when use_market_impact is True."""

    # Model
    model_params: dict = field(default_factory=dict)
    """Extra LightGBM constructor kwargs (e.g. num_leaves, max_depth), merged over the defaults —
    the knob varied across trials in the multi-configuration deflated-Sharpe study."""

    # Reproducibility
    random_seed: int = 42

    # Caching
    cache_dir: str = "data_cache"
