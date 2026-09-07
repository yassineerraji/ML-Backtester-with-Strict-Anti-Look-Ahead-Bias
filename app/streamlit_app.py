"""Interactive companion to reports/bias_comparison.md, purge_embargo.md, and
deflation_study.md: configure the pipeline, run it, and see the naive-vs-corrected
Sharpe gap, the purge/embargo fold timeline, and the deflated Sharpe sweep update live.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from app.logic import (
    build_config_from_form,
    compute_fold_timeline,
    cumulative_equity,
    format_metrics_table,
    grid_from_editor_dataframe,
    validate_config,
)
from ml_backtester.config import DEFAULT_UNIVERSE, Config
from ml_backtester.experiments.run_comparison import build_dataset, run_walk_forward
from ml_backtester.experiments.run_deflation_study import (
    MODEL_HYPERPARAMETER_GRID,
    compute_deflated_sharpe,
    run_config_sweep,
)
from ml_backtester.validation.splitters import PurgedEmbargoedSplitter, WalkForwardSplitter

SEGMENT_COLORS = {
    "train": "#4C78A8",
    "purged": "#E45756",
    "test": "#54A24B",
    "embargo": "#B279A2",
}


@st.cache_data(show_spinner=False)
def cached_build_dataset(config: Config) -> pd.DataFrame:
    """Cache dataset construction by config — the most expensive step every tab shares."""
    return build_dataset(config)


@st.cache_data(show_spinner=False)
def cached_walk_forward(config: Config, variant: str):
    """Cache one full walk-forward run (naive or corrected) by (config, variant)."""
    dataset = cached_build_dataset(config)
    splitter_cls = WalkForwardSplitter if variant == "naive" else PurgedEmbargoedSplitter
    splitter = splitter_cls(config.train_window_days, config.test_window_days, config.embargo_days)
    return run_walk_forward(dataset, splitter, config)


@st.cache_data(show_spinner=False)
def cached_sweep(config: Config, grid: list[dict]):
    """Cache the hyperparameter sweep by (config, grid)."""
    return run_config_sweep(config, grid)


def render_sidebar() -> Config:
    """Render the configuration form and build a Config from its current widget values."""
    st.sidebar.header("Configuration")
    with st.sidebar.form("config_form"):
        universe = st.multiselect("Universe", options=DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)

        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start date", value=date(2015, 1, 1))
        end_date = col2.date_input("End date", value=date(2024, 12, 31))

        horizon_days = st.number_input("Horizon (trading days)", min_value=1, value=5)
        train_window_days = st.number_input("Train window (trading days)", min_value=1, value=30)
        test_window_days = st.number_input("Test window (trading days)", min_value=1, value=10)
        embargo_days = st.number_input("Embargo (trading days)", min_value=0, value=5)
        if train_window_days:
            st.caption(
                f"Purge fraction ≈ horizon / train window = "
                f"{horizon_days / train_window_days:.0%} of each training block"
            )

        with st.expander("Feature windows"):
            momentum_window = st.number_input("Momentum window", min_value=2, value=20)
            volatility_window = st.number_input("Volatility window", min_value=2, value=20)
            relative_volume_window = st.number_input(
                "Relative volume window", min_value=2, value=20
            )
            rsi_window = st.number_input("RSI window", min_value=2, value=14)
            normalization_window = st.number_input("Normalization window", min_value=2, value=60)

        with st.expander("Costs & position sizing"):
            risk_cap = st.slider("Risk cap", 0.1, 2.0, 1.0)
            spread_bps = st.number_input("Spread (bps)", min_value=0.0, value=5.0)
            slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, value=2.0)
            use_market_impact = st.checkbox("Use square-root-law market impact", value=False)
            impact_coefficient = st.number_input("Impact coefficient", min_value=0.0, value=0.1)

        random_seed = st.number_input("Random seed", min_value=0, value=42)
        st.form_submit_button("Apply configuration")

    return build_config_from_form(
        {
            "universe": universe,
            "start_date": start_date,
            "end_date": end_date,
            "horizon_days": horizon_days,
            "train_window_days": train_window_days,
            "test_window_days": test_window_days,
            "embargo_days": embargo_days,
            "momentum_window": momentum_window,
            "volatility_window": volatility_window,
            "relative_volume_window": relative_volume_window,
            "rsi_window": rsi_window,
            "normalization_window": normalization_window,
            "risk_cap": risk_cap,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "use_market_impact": use_market_impact,
            "impact_coefficient": impact_coefficient,
            "random_seed": random_seed,
        }
    )


def render_bias_tab(config: Config) -> None:
    """Render the naive-vs-corrected Sharpe comparison tab."""
    st.subheader("Naive vs. Corrected Walk-Forward Comparison")
    problems = validate_config(config)
    for problem in problems:
        st.error(problem)
    if problems:
        return

    if st.button("Run comparison", key="run_bias"):
        with st.spinner("Running naive and corrected walk-forward backtests..."):
            naive = cached_walk_forward(config, "naive")
            corrected = cached_walk_forward(config, "corrected")
        st.session_state["bias_result"] = (naive, corrected)

    result = st.session_state.get("bias_result")
    if result is None:
        st.info("Configure the sidebar and click **Run comparison**.")
        return
    naive, corrected = result

    gap = naive.metrics["sharpe"] - corrected.metrics["sharpe"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Naive Sharpe", f"{naive.metrics['sharpe']:.3f}")
    col2.metric("Corrected Sharpe", f"{corrected.metrics['sharpe']:.3f}")
    col3.metric("Sharpe gap (naive − corrected)", f"{gap:.3f}")

    st.dataframe(format_metrics_table(naive.metrics, corrected.metrics).style.format("{:.4f}"))

    equity = pd.DataFrame(
        {
            "Naive": cumulative_equity(naive.returns),
            "Corrected": cumulative_equity(corrected.returns),
        }
    )
    st.line_chart(equity)
    st.caption(
        "Naive and corrected runs don't cover identical calendar dates (the corrected "
        "splitter fits fewer folds, since it spends time on the embargo gap) — see "
        "reports/limitations.md."
    )


def render_timeline_tab(config: Config) -> None:
    """Render the purge/embargo fold timeline tab."""
    st.subheader("Purge & Embargo Fold Timeline")
    problems = validate_config(config)
    for problem in problems:
        st.error(problem)
    if problems:
        return

    if st.button("Compute timeline", key="run_timeline"):
        with st.spinner("Building dataset and computing fold boundaries..."):
            dataset = cached_build_dataset(config)
            naive_splitter = WalkForwardSplitter(
                config.train_window_days, config.test_window_days, config.embargo_days
            )
            corrected_splitter = PurgedEmbargoedSplitter(
                config.train_window_days, config.test_window_days, config.embargo_days
            )
            timeline = pd.concat(
                [
                    compute_fold_timeline(dataset, naive_splitter, "naive"),
                    compute_fold_timeline(dataset, corrected_splitter, "corrected"),
                ],
                ignore_index=True,
            )
        st.session_state["timeline_result"] = timeline

    timeline = st.session_state.get("timeline_result")
    if timeline is None:
        st.info("Configure the sidebar and click **Compute timeline**.")
        return

    max_fold = int(timeline["fold"].max())
    folds_to_show = st.slider(
        "Folds to display (per variant)", 1, max_fold + 1, min(10, max_fold + 1)
    )
    display_df = timeline[timeline["fold"] < folds_to_show]

    chart = (
        alt.Chart(display_df)
        .mark_bar(height=14)
        .encode(
            x=alt.X("start:T", title="Date"),
            x2="end:T",
            y=alt.Y("fold:O", title="Fold"),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(
                    domain=list(SEGMENT_COLORS.keys()), range=list(SEGMENT_COLORS.values())
                ),
                title="Segment",
            ),
            row=alt.Row("variant:N", title=None),
            tooltip=["variant", "fold", "segment", "start:T", "end:T"],
        )
        .properties(width=700)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Red 'purged' bands mark training dates whose label window overlaps the test block. "
        "The corrected splitter drops them from training; the naive splitter keeps them — "
        "same highlight, different consequence. Purple 'embargo' bands only appear for the "
        "corrected variant."
    )
    st.dataframe(display_df, use_container_width=True)


def render_sweep_tab(config: Config) -> None:
    """Render the deflated-Sharpe hyperparameter sweep tab."""
    st.subheader("Hyperparameter Sweep — Deflated Sharpe Ratio")
    problems = validate_config(config)
    for problem in problems:
        st.error(problem)
    if problems:
        return

    st.caption(
        "Edit, add, or remove rows to define the trial grid — each row is one LightGBM config."
    )
    grid_df = st.data_editor(
        pd.DataFrame(MODEL_HYPERPARAMETER_GRID), num_rows="dynamic", key="grid_editor"
    )

    if st.button("Run sweep", key="run_sweep"):
        grid = grid_from_editor_dataframe(grid_df)
        if not grid:
            st.error("Grid must have at least one trial row.")
        else:
            with st.spinner(f"Running {len(grid)} trial(s) on the corrected splitter..."):
                trials = cached_sweep(config, grid)
                best, deflated = compute_deflated_sharpe(trials)
            st.session_state["sweep_result"] = (trials, best, deflated)

    result = st.session_state.get("sweep_result")
    if result is None:
        st.info("Edit the grid and click **Run sweep**.")
        return
    trials, best, deflated = result

    trial_table = pd.DataFrame(
        [
            {
                "Hyperparameters": str(trial.hyperparameters),
                "Annualized Sharpe": trial.annualized_sharpe,
            }
            for trial in trials
        ]
    )
    st.dataframe(trial_table, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric(
        "Best observed Sharpe", f"{best.annualized_sharpe:.3f}", help=str(best.hyperparameters)
    )
    col2.metric("Deflated Sharpe ratio", f"{deflated:.3f}")

    if deflated >= 0.95:
        st.success("Unlikely to be a product of testing multiple configurations.")
    elif deflated <= 0.5:
        st.warning("Indistinguishable from the best of N skill-less strategies — likely noise.")
    else:
        st.info("Ambiguous — neither clearly noise nor clearly skill.")


def main() -> None:
    """Entry point: page config, sidebar, and the three tabs."""
    st.set_page_config(page_title="ML Backtester", page_icon="📈", layout="wide")
    st.title("ML Backtester — Purge & Embargo Bias Demonstration")
    st.caption(
        "Interactive companion to reports/bias_comparison.md, reports/purge_embargo.md, "
        "and reports/deflation_study.md."
    )

    config = render_sidebar()
    tab_bias, tab_timeline, tab_sweep = st.tabs(
        ["Naive vs. Corrected", "Purge & Embargo Timeline", "Deflated Sharpe Sweep"]
    )
    with tab_bias:
        render_bias_tab(config)
    with tab_timeline:
        render_timeline_tab(config)
    with tab_sweep:
        render_sweep_tab(config)


main()
