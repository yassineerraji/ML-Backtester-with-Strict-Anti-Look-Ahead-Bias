# Limitations

Documented deliberately, per the project's goal of demonstrating
methodology rather than claiming a production-ready trading system. Numbers
below reflect the actual runs behind [bias_comparison.md](bias_comparison.md)
and [deflation_study.md](deflation_study.md), both produced with `Config()`'s
plain defaults (`DEFAULT_UNIVERSE`, 15 large-cap US equities, 2015-01-01 to
2024-12-31).

- **Restricted universe.** 15 large-cap US equities (`DEFAULT_UNIVERSE` in
  `config.py`) — not a market-representative sample, and results including
  the sign and magnitude of the naive-vs-corrected Sharpe gap are not
  claimed to generalize beyond this universe or period.
- **Train window sized to make the leak visible, not to be realistic.**
  `train_window_days=30` was chosen because it's small enough that purge
  removes a meaningful fraction (17%) of each fold's training data — with a
  multi-year training window more typical of an institutional retrain
  cadence, the same leak exists but purges away too small a fraction to
  move the Sharpe ratio noticeably. See
  [purge_embargo.md](purge_embargo.md) for both cases side by side.
- **Naive and corrected runs don't cover identical calendar time.** Because
  the naive splitter skips no embargo gap, it packs more folds into the
  same date range than the corrected splitter (60 vs. 54 in the current
  run) — so part of the measured Sharpe gap reflects the corrected run
  being evaluated over a slightly different (smaller) set of test periods,
  not the leak alone. This is inherent to comparing "no embargo" against
  "embargo" over a fixed date range, not a bug.
- **No financing or borrow costs.** `backtest/costs.py` models spread and
  slippage (and optional square-root-law market impact) but not the cost of
  borrowing shares to hold a short position, nor cash financing costs on
  leveraged exposure.
- **Daily data only.** Features and labels use daily OHLCV bars; there is no
  intraday data, so microstructure effects (e.g. within-day mean reversion,
  open/close auction dynamics) aren't captured.
- **Simplified position sizing.** `backtest/engine.py` sizes positions by
  cross-sectional z-scoring the model's signal and capping at `risk_cap` —
  not a portfolio optimizer, and it ignores correlation between assets when
  sizing.
- **One model family.** Only LightGBM is exercised end-to-end (via
  `models/train.py`); the pipeline was built model-agnostic, but no
  comparison across model families (e.g. logistic regression vs. gradient
  boosting) has been run.
- **Small, non-independent hyperparameter grid.** The deflated Sharpe ratio
  study (`experiments/run_deflation_study.py`) tests 5 hyperparameter
  configurations, all LightGBM variants trained on the same data — the
  deflation formula's assumption of independent trials is only
  approximately true here, since the configurations are correlated by
  sharing a model family and dataset.
- **No true holdout period.** The naive-vs-corrected comparison and the
  hyperparameter sweep both draw folds from the same 2015–2024 window.
  There is no separate, never-touched final holdout period to validate the
  winning configuration against.
- **Deflated Sharpe ratio is itself an approximation.** `backtest/deflation.py`
  uses the extreme-value approximation from Bailey & López de Prado (2014)
  for the expected maximum Sharpe under the null, which assumes
  approximately normal per-trial Sharpe ratios — a simplification, not an
  exact result, particularly for small `n_trials`.
- **Fixed walk-forward geometry.** `train_window_days`, `test_window_days`,
  and `embargo_days` are constant across the whole backtest; the pipeline
  does not adapt fold sizes to changing volatility regimes.
