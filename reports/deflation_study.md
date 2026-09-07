# Hyperparameter Sweep — Deflated Sharpe Ratio

5 configurations tested on identical data, splitter, and costs (purged/embargoed walk-forward); only model hyperparameters vary.

| Hyperparameters | Annualized Sharpe |
| --- | --- |
| {'num_leaves': 7, 'max_depth': 2} | 1.2613 |
| {'num_leaves': 15, 'max_depth': 3} | 1.0048 |
| {'num_leaves': 31, 'max_depth': 5} | 0.8405 |
| {'num_leaves': 63, 'max_depth': 7} | 0.7469 |
| {'num_leaves': 15, 'max_depth': 7} | 0.6523 |

**Best observed Sharpe: 1.2613** ({'num_leaves': 7, 'max_depth': 2})
**Deflated Sharpe ratio: 0.9247**

The deflated Sharpe ratio is the probability the best trial's true Sharpe exceeds the Sharpe an average skill-less strategy would achieve by chance, given 5 configurations were tried. A value near 0.5 means the best-observed result is indistinguishable from noise; a value near 1.0 means it is unlikely to be a product of the search itself.