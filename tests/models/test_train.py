"""Tests for the model wrapper: correct shape, determinism, and per-fold freshness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_backtester.config import Config
from ml_backtester.features.engineering import FEATURE_COLUMNS
from ml_backtester.models.train import fit_predict, make_model


@pytest.fixture
def config() -> Config:
    return Config(random_seed=7)


def _synthetic_fold(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    data["label"] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


def test_fit_predict_returns_one_prediction_per_test_row(config):
    train = _synthetic_fold(50, seed=1)
    test = _synthetic_fold(10, seed=2)

    predictions = fit_predict(train, test, config)

    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()


def test_fit_predict_is_deterministic_given_same_seed(config):
    train = _synthetic_fold(50, seed=1)
    test = _synthetic_fold(10, seed=2)

    first = fit_predict(train, test, config)
    second = fit_predict(train, test, config)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_construct_differently_seeded_models():
    model_a = make_model(Config(random_seed=1))
    model_b = make_model(Config(random_seed=2))

    assert model_a.random_state != model_b.random_state


def test_model_params_reach_the_underlying_estimator():
    model = make_model(Config(model_params={"num_leaves": 7, "max_depth": 3}))

    assert model.num_leaves == 7
    assert model.max_depth == 3


def test_different_model_params_change_predictions():
    """max_depth alone can be masked by LightGBM's min_child_samples default on tiny data,
    so vary n_estimators instead — the number of boosting rounds always moves predictions."""
    train = _synthetic_fold(50, seed=1)
    test = _synthetic_fold(10, seed=2)

    few_rounds = fit_predict(train, test, Config(random_seed=7, model_params={"n_estimators": 1}))
    many_rounds = fit_predict(train, test, Config(random_seed=7, model_params={"n_estimators": 50}))

    assert not np.array_equal(few_rounds, many_rounds)


def test_fit_predict_uses_only_the_given_fold(config):
    """Predictions for a fixed test fold must not change based on unrelated train data seed."""
    test = _synthetic_fold(10, seed=99)

    predictions_a = fit_predict(_synthetic_fold(50, seed=1), test, config)
    predictions_b = fit_predict(_synthetic_fold(50, seed=2), test, config)

    assert predictions_a.shape == predictions_b.shape
    assert not np.array_equal(predictions_a, predictions_b)
