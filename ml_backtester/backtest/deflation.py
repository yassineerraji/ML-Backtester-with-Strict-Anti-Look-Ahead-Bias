"""Deflated Sharpe ratio (Bailey & López de Prado), correcting for multiple-testing bias.

Testing many feature/hyperparameter configurations and reporting the best
observed Sharpe overstates skill: some configurations look good by chance.
The deflated Sharpe ratio asks "what Sharpe would even a skill-less strategy
achieve by chance, given how many configurations were tried?" and reports
the probability the observed Sharpe exceeds that chance benchmark.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _expected_max_sharpe_under_null(n_trials: int, sharpe_std: float) -> float:
    """Expected maximum Sharpe ratio across n_trials independent skill-less strategies.

    Uses the extreme value approximation from Bailey & López de Prado
    (2014), "The Deflated Sharpe Ratio."
    """
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649
    return sharpe_std * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
) -> float:
    """Probability the observed Sharpe ratio exceeds what chance alone would produce.

    Args:
        observed_sharpe: Best (non-annualized, per-period) Sharpe ratio
            observed across all trials.
        n_trials: Number of independent feature/hyperparameter configurations tested.
        n_observations: Number of return observations the Sharpe ratio was computed over.
        skewness: Skewness of the per-period returns of the selected strategy.
        kurtosis: Kurtosis of the per-period returns of the selected strategy
            (3.0 for a normal distribution).
        sharpe_std: Cross-trial standard deviation of Sharpe ratios, used to
            scale the expected maximum under the null. Defaults to 1.0
            (unit-scale trials) when this isn't estimated from the trials.

    Returns:
        Deflated Sharpe ratio: P(true Sharpe > expected max Sharpe under the
        null of no skill), in [0, 1]. Values near 1 indicate the observed
        Sharpe is unlikely to be a product of testing many configurations;
        values near 0.5 indicate it is indistinguishable from the best of
        n_trials skill-less strategies.
    """
    benchmark_sharpe = _expected_max_sharpe_under_null(n_trials, sharpe_std)

    numerator = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1)
    denominator = np.sqrt(
        1 - skewness * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    )
    return float(norm.cdf(numerator / denominator))
