# ML Backtester — Agent Build Guide

Read `README.md` first — it defines the objective. This file pins down the
architecture, naming, and conventions so the project can be built
incrementally without ambiguity or drift.

## Core Thesis (do not lose sight of this)

The project's entire value is demonstrating, with numbers, that a naive
walk-forward split (no purge/embargo) inflates the Sharpe ratio versus a
correct purged/embargoed split. Every design decision should preserve the
ability to run both versions on identical data/features/model and diff the
result. If a shortcut would make the two versions harder to compare
apples-to-apples, don't take it.

## Directory Layout

Build exactly this structure. Do not invent alternate top-level dirs.

```
ml_backtester/
    __init__.py
    config.py              # Config dataclass: universe, dates, horizon, costs, seeds
    data/
        __init__.py
        loader.py           # yfinance download + local parquet cache
    features/
        __init__.py
        engineering.py       # momentum, realized vol, relative volume, RSI — rolling-only
    labeling/
        __init__.py
        targets.py            # forward return label + label end-time (t1) per sample
    validation/
        __init__.py
        splitters.py            # WalkForwardSplitter, PurgedEmbargoedSplitter
    models/
        __init__.py
        train.py                  # model factory + fit/predict wrapper
    backtest/
        __init__.py
        engine.py                  # BacktestEngine: signal -> position -> PnL
        costs.py                    # spread/slippage (+ optional sqrt-law impact)
        metrics.py                   # sharpe, sortino, max_drawdown, turnover, hit_ratio
        deflation.py                  # deflated Sharpe ratio (López de Prado)
    experiments/
        __init__.py
        run_comparison.py              # naive vs corrected end-to-end, produces the report
        run_deflation_study.py          # hyperparameter sweep + deflated Sharpe, produces the report

app/
    __init__.py
    logic.py                 # pure, streamlit-free: config building, validation, chart data prep
    streamlit_app.py          # thin rendering layer (widgets, st.cache_data) over app/logic.py
                               # and ml_backtester.experiments — see "UI (Streamlit)" below

scripts/
    run_pipeline.py          # thin CLI entry point calling ml_backtester.experiments

tests/
    test_<mirrors ml_backtester/ subpackage and module names>
    app/
        test_logic.py          # plain pytest, no streamlit import
        test_streamlit_app.py  # streamlit.testing.v1.AppTest smoke tests

reports/
    bias_comparison.md        # the naive-vs-corrected numbers + explanation
    purge_embargo.md            # mechanism + diagram (mirror README mermaid)
    limitations.md                # universe size, no financing costs, etc.
    deflation_study.md              # hyperparameter sweep + deflated Sharpe result

pyproject.toml
requirements.txt
```

One module = one responsibility. `engine.py` never calls `yfinance`;
`loader.py` never computes indicators; `splitters.py` never touches the
model. If a function needs something from another layer, import it — don't
inline it.

## Documentation Convention (mandatory, non-negotiable)

- **Every file** starts with a module docstring, 1–3 lines: what this module
  does and why it exists. No essay.
- **Every function/method/class** gets a Google-style docstring: one-line
  summary, then `Args:`/`Returns:` only when the signature isn't
  self-explanatory from type hints. Skip `Raises:` unless the function
  deliberately raises for a caller-relevant reason.
- Docstrings explain *intent*, not mechanics the code already shows. For the
  bias-critical functions (purge, embargo, label horizon) the docstring must
  state the temporal invariant being enforced — this is the pedagogical
  point of the project.
- No inline comments restating the next line. A comment is only for a
  non-obvious *why* (e.g., "using t1 not t0 here because the label horizon
  extends past the observation date").

Example:

```python
"""Purged, embargoed walk-forward cross-validation splitter.

Prevents label leakage across the train/test boundary caused by
overlapping label windows and residual autocorrelation.
"""

class PurgedEmbargoedSplitter:
    """Yields (train_idx, test_idx) folds with purge and embargo applied."""

    def split(self, X: pd.DataFrame, t1: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index folds.

        Args:
            X: Feature matrix indexed by observation timestamp.
            t1: Label end-time for each observation (event's outcome time).

        Returns:
            Iterator of (train_idx, test_idx) index arrays. Train indices
            exclude any observation whose [t0, t1] interval overlaps the
            test window (purge), plus an embargo buffer after the test
            window (embargo).
        """
```

## Temporal-Correctness Rules (hard constraints)

These are the rules the whole project exists to demonstrate — violating them
silently defeats the purpose:

1. **No global fit before splitting.** Never fit a scaler/normalizer/model
   on the full dataset and then split. Fit only on the train fold, inside
   the walk-forward loop.
2. **Rolling-only feature engineering.** Every feature in `features/` must
   use a trailing rolling or expanding window ending at `t`, never the full
   series mean/std. If a feature can't be computed causally, it doesn't go
   in.
3. **Every sample carries a label end-time (`t1`).** `labeling/targets.py`
   must attach `t1` (observation time + horizon) to each row — this is what
   `PurgedEmbargoedSplitter` purges against. Don't compute purge from label
   *count* or row position; compute it from actual timestamps.
4. **Two splitters, one interface.** `WalkForwardSplitter` (naive: rolling
   train/test, no purge/embargo) and `PurgedEmbargoedSplitter` must share
   the same call signature so `experiments/run_comparison.py` swaps one for
   the other with a single argument — nothing else in the pipeline changes
   between the two runs.
5. **No shuffling.** Never shuffle rows for a time series split. If a
   library default shuffles (e.g. sklearn `KFold`), don't use it — use the
   custom splitters.
6. **Costs and position sizing are identical across both runs.** The Sharpe
   gap must come only from the validation methodology, not from a
   difference in backtest mechanics.

## Backtest Engine Rules

- Position sizing: signal-proportional with a hard risk cap (max position
  size), configurable in `config.py`.
- Costs: spread + slippage always on; square-root-law market impact is
  optional and toggled via config, off by default.
- `metrics.py` functions are pure: `(returns: pd.Series, ...) -> float`. No
  hidden state, no I/O.
- `deflation.py` implements the deflated Sharpe ratio and must be applied
  whenever `run_comparison.py` reports results across multiple
  feature/hyperparameter configurations — plain Sharpe alone is not
  sufficient in that case.

## Tech Stack

| Area | Choice |
| --- | --- |
| Language | Python ≥ 3.11, full type hints |
| Data | `pandas`, `yfinance`, local parquet cache (avoid re-downloading) |
| Modeling | `scikit-learn` + `lightgbm` (prefer LightGBM if both are viable) |
| Backtesting | In-house (`ml_backtester/backtest/`) — do not pull in `backtrader` or similar |
| Testing | `pytest`, `streamlit.testing.v1.AppTest` for the UI |
| Formatting/Linting | `black`, `ruff` |
| Config | `dataclasses` in `config.py`, no scattered magic numbers |
| UI | `streamlit` — see "UI (Streamlit)" below |

## Testing Expectations

- Every module in `validation/` and `labeling/` needs unit tests asserting
  the temporal invariant directly (e.g., assert no train index has
  `t1 >= test_start - embargo` overlap with the test window) — not just
  that the code runs.
- `metrics.py` functions get tests against hand-computed values on a small
  fixed series.
- `tests/` mirrors `ml_backtester/` module-for-module: e.g.
  `ml_backtester/validation/splitters.py` → `tests/validation/test_splitters.py`.

## UI (Streamlit)

`app/streamlit_app.py` is an interactive front end over the exact same
`ml_backtester.experiments` functions the CLI and reports use — it does not
reimplement any pipeline logic. Two rules keep it testable without a browser:

- **Pure logic lives in `app/logic.py`, not in `streamlit_app.py`.** Config
  building from widget values, input validation, chart/table data
  preparation — anything that doesn't call `st.*` — goes in `logic.py` and
  gets plain `pytest` unit tests (`tests/app/test_logic.py`). If a function
  needs `streamlit` imported to be tested, it's in the wrong file.
- **`streamlit_app.py` itself is tested with `streamlit.testing.v1.AppTest`**
  (`tests/app/test_streamlit_app.py`), which runs the real script in a
  simulated runtime and lets tests click buttons and assert on rendered
  output — no browser needed. These tests hit the real (cached) pipeline,
  not mocks; they're slower than the rest of the suite for that reason.
- Expensive calls (`build_dataset`, `run_walk_forward`, `run_config_sweep`)
  are wrapped in `st.cache_data` keyed by the whole `Config` object, so
  identical reruns are instant. Note this means changing *any* Config field
  invalidates `cached_build_dataset`'s cache even for fields it doesn't
  actually use (e.g. `risk_cap`) — an accepted inefficiency, not a bug to
  chase.

## Build Order

Follow this sequence; each step should be runnable/testable before the next:

1. `config.py` + `data/loader.py` (fetch + cache universe)
2. `features/engineering.py` (rolling features)
3. `labeling/targets.py` (forward return + `t1`)
4. `validation/splitters.py` (both splitters, with tests proving the
   invariant)
5. `models/train.py` (fit/predict wrapper, model-agnostic)
6. `backtest/engine.py` + `costs.py` + `metrics.py`
7. `experiments/run_comparison.py` (naive vs corrected, writes
   `reports/bias_comparison.md`)
8. `deflation.py` applied once multiple configs are being compared
9. `reports/` docs filled in last, from actual run output — never write
   numbers into the reports that weren't produced by a real run
10. `app/logic.py` + `app/streamlit_app.py` (interactive UI over steps 1-9,
    once the pipeline itself is correct — see "UI (Streamlit)" above)

## What Not To Do

- Don't add a database, background job queue, or auth — `app/streamlit_app.py`
  is the one sanctioned UI layer, a thin front end over the same pipeline
  the CLI and reports use, not a service with its own backend.
- Don't reach for `backtrader`/`zipline`/`vectorbt` for the core backtest —
  the point is owning the purge/embargo logic.
- Don't add config knobs or abstractions for scenarios the README doesn't
  ask for (multi-asset-class, live trading, etc.).
- Don't skip the "naive" path to save time — the comparison *is* the
  deliverable.