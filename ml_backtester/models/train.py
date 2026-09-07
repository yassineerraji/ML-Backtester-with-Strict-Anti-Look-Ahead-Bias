"""Model-agnostic fit/predict wrapper around the chosen ML estimator.

A fresh model is fit on each walk-forward fold's training data alone and
predicts only that fold's test data — never fit on the full dataset — so no
prediction can depend on information outside its own fold's train window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin

from ml_backtester.config import Config
from ml_backtester.features.engineering import FEATURE_COLUMNS


def make_model(config: Config) -> RegressorMixin:
    """Construct a fresh, unfit regressor predicting forward return.

    Args:
        config: Pipeline configuration providing random_seed and model_params.

    Returns:
        An unfit scikit-learn-compatible regressor. `deterministic=True` plus
        `force_row_wise=True` are required for LightGBM to be bit-reproducible
        across separate process runs, not just within one — `random_state`
        alone fixes the seed but not the (by default multi-threaded, order-
        dependent) histogram-building strategy.
    """
    return LGBMRegressor(
        random_state=config.random_seed,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
        **config.model_params,
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, config: Config) -> np.ndarray:
    """Fit a fresh model on one fold's training rows and predict on its test rows.

    Args:
        train: Training fold rows with FEATURE_COLUMNS and a "label" column.
        test: Test fold rows with FEATURE_COLUMNS.
        config: Pipeline configuration providing model hyperparameters.

    Returns:
        Predicted forward return per row of `test`, in `test`'s row order.
    """
    model = make_model(config)
    model.fit(train[FEATURE_COLUMNS], train["label"])
    return model.predict(test[FEATURE_COLUMNS])
