# Benchmark findings

Computed from [raw results](raw/). The paired model tables contain median and IQR for the full fit and test prediction timings alongside the test errors.

| Check | Threshold | Observed | Status |
| --- | --- | --- | --- |
| RQ1 Direct/Scaled speedup | At least 2× at n=10⁴, then increasing | 58.65×, 600.71×, 3197.90× | PASS |
| RQ1 dimension slopes | Direct in [0.7, 1.1], Scaled ≤ 0.35 | Direct 1.002, Scaled 0.133 | PASS |
| RQ1 fully dense speedup | In [0.5, 3.0] | 1.12× | PASS |
| RQ1 Scaled max/min time per update | ≤ 1.30 | 1.11 | PASS |
| RQ1 Direct/Scaled relative weight error | < 1e-10 | 5.11e-14 | PASS |
| RQ2 Direct/scikit-learn | Relative RMSE < 1e-6 and weight error < 1e-10 | RMSE 9.32e-16, weights 2.49e-14 | PASS |
| RQ2 Direct/scikit-learn relative Ridge distance error | < 1e-6 | 9.15e-14 | PASS |
| RQ2 Scaled/scikit-learn | Relative RMSE < 1e-6 and weight error < 1e-10 | RMSE 0.00e+00, weights 0.00e+00 | PASS |
| RQ2 Scaled/scikit-learn relative Ridge distance error | < 1e-6 | 0.00e+00 | PASS |

- [RQ1 paired table](tables/rq1_paired.csv)
- [Taiwan](tables/rq2_taiwan.csv), [California](tables/rq2_california.csv), and [King County](tables/rq2_kingcounty.csv)
