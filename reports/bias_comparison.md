# Naive vs. Corrected Walk-Forward Comparison

| Metric | Naive (no purge/embargo) | Corrected (purged/embargoed) |
| --- | --- | --- |
| sharpe | 1.5717 | 0.4578 |
| sortino | 2.4660 | 0.7281 |
| max_drawdown | -0.1498 | -0.2348 |
| turnover | 0.0719 | 0.0783 |
| hit_ratio | 0.5383 | 0.5037 |

**Sharpe gap (naive − corrected): 1.1139**

The naive Sharpe is inflated because training samples whose label window overlaps the test period leak test-period information into the model (no purge), and no embargo buffer removes residual autocorrelation across the train/test boundary.