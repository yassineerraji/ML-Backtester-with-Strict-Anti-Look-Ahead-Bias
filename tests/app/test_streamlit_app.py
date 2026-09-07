"""Smoke tests for the Streamlit app via AppTest — simulates the app without a browser.

Uses the real (already-cached) default-config dataset, so these exercise the actual
pipeline end to end, not mocks. st.cache_data is process-level, so once one test warms
the cache for the default config, later tests in this file reuse it and run fast.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
TIMEOUT = 300


def test_app_boots_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    assert not at.exception


def test_sidebar_and_tabs_render_expected_widgets():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)

    assert not at.exception
    assert len(at.multiselect) == 1
    assert len(at.date_input) == 2
    assert len(at.tabs) == 3


def test_running_bias_comparison_produces_metrics_and_no_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception

    at.button(key="run_bias").click().run(timeout=TIMEOUT)

    assert not at.exception
    assert len(at.metric) >= 3


def test_running_timeline_produces_no_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception

    at.button(key="run_timeline").click().run(timeout=TIMEOUT)

    assert not at.exception


def test_invalid_config_shows_validation_errors_instead_of_running():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception

    # Sidebar number_input order: horizon, train_window, test_window, embargo, ...
    # Drive horizon_days above train_window_days to trigger validate_config's error.
    horizon_input = at.sidebar.number_input[0]
    train_window_input = at.sidebar.number_input[1]
    train_window_input.set_value(5).run(timeout=TIMEOUT)
    horizon_input.set_value(10).run(timeout=TIMEOUT)

    assert not at.exception
    assert len(at.error) > 0


def test_running_sweep_produces_metrics_and_no_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception

    at.button(key="run_sweep").click().run(timeout=TIMEOUT)

    assert not at.exception
    assert len(at.metric) >= 2
