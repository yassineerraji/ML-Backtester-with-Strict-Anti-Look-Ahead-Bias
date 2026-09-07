# Naive vs. Corrected Walk-Forward Comparison

| Metric | Naive (no purge/embargo) | Corrected (purged/embargoed) |
| --- | --- | --- |
| sharpe | 1.7625 | 0.4962 |
| sortino | 2.8178 | 0.7869 |
| max_drawdown | -0.1508 | -0.2298 |
| turnover | 0.0699 | 0.0784 |
| hit_ratio | 0.5433 | 0.5000 |

**Sharpe gap (naive − corrected): 1.2664**

The naive Sharpe is inflated because training samples whose label window overlaps the test period leak test-period information into the model (no purge), and no embargo buffer removes residual autocorrelation across the train/test boundary.