# Architectural & Methodological Decisions Log

This document records all methodological decisions, deviations, and refinements relative to the original Reading Report (dated 17/10/2026, CS161).
Any modifications to parameters, definitions, or procedures must be documented here with justification.

---

## Initial Decisions (Initialized from plan.md Section 2)

### Decision 1: Exact Relationship for Ridge Reference Regularization
- **Date**: 2026-10-23
- **Report Statement**: `alpha_Ridge ≈ m · alpha_SGD`
- **Adopted Specification**: `alpha_Ridge = m_train · alpha_SGD` (exact equivalence of objective function).
- **Rationale**: Multiplying the SGD objective `(1/m) * sum(0.5 * (y_hat - y)^2) + (alpha / 2) * ||w||^2` by `2 * m` yields `sum((y_hat - y)^2) + m * alpha * ||w||^2`, which matches scikit-learn's `Ridge(alpha = m_train * alpha_SGD, fit_intercept=True)`. The approximation sign in the report only reflects the fact that SGD has not reached complete asymptotic convergence. `m` is strictly the training set sample count `m_train`. Verified via unit test `test_alpha_ridge`.

### Decision 2: Relative Error Thresholds for Quality Verification
- **Date**: 2026-10-23
- **Report Statement**: Absolute threshold `|RMSE_cpp - RMSE_sklearn| < 1e-6`.
- **Adopted Specification**: Relative threshold `|Delta RMSE| / RMSE_sklearn < 1e-6` (initial threshold, calibrated after Phase 5 pilot).
- **Rationale**: Target variables (e.g. King County house prices on the order of 10^5 USD) make absolute thresholds of 1e-6 meaningless or overly strict. Relative threshold scales properly across datasets.

### Decision 3: Regularization Parameter alpha for RQ2
- **Date**: 2026-10-23
- **Report Statement**: `alpha` left blank / unspecified for RQ2.
- **Adopted Specification**: Fixed default `alpha = 1e-4` (matching scikit-learn default and RQ1).
- **Rationale**: Established prior to benchmarking to prevent post-hoc parameter tuning.

### Decision 4: Epoch Protocol for RQ2
- **Date**: 2026-10-23
- **Report Statement**: Epoch count `E` "shared / unspecified".
- **Adopted Specification**: Fixed primary epoch count `E_main = 100`, plus an explicit sweep `E in {1, 5, 20, 100, 500}`. Each run is trained from scratch (`w_0 = 0`, no warm start).
- **Rationale**: Avoids test-set leakage in epoch selection while enabling convergence trajectory analysis.

### Decision 5: Density Scaling Behavior in Figure 2
- **Date**: 2026-10-23
- **Report Statement**: Expected the 3 timing curves to fully "converge" as density -> 100%.
- **Adopted Specification**: The ratio `t_direct / t_scaled` approaches an O(1) constant (approximately 1 to 2), not necessarily exactly 1.
- **Rationale**: At 100% density, direct updates perform 1 contiguous shrinkage pass (amenable to compiler SIMD auto-vectorization) + 2 gather/scatter passes; scaled updates perform 2 gather/scatter passes. Memory layout and instruction pipeline differences mean the execution constants differ slightly.

### Decision 6: Sample Traversal Ordering
- **Date**: 2026-10-23
- **Report Statement**: "Same sample order".
- **Adopted Specification**: Deterministic row order in data file (`i = 0 .. m-1` per epoch), `shuffle=False` in scikit-learn; global step `t = 1, 2, ...` advances continuously across epochs.
- **Rationale**: Guarantees identical numerical trajectory between C++ and scikit-learn down to step-level operations.

### Decision 7: Metric Computation Responsibility
- **Date**: 2026-10-23
- **Report Statement**: Metrics computed separately on each side.
- **Adopted Specification**: Metrics are exclusively computed in Python (`metrics.py`) from exported `(w, b)`; C++ solely performs training and exports `w, b` and timing.
- **Rationale**: Eliminates metric discrepancies caused by differing implementations of RMSE, MAE, or R^2.

### Decision 8: Data Representation for RQ2
- **Date**: 2026-10-23
- **Report Statement**: Dense data representation for C++.
- **Adopted Specification**: Python writes preprocessed training sets to CSR binary (`train.bin`), while scikit-learn receives dense contiguous numpy arrays with `intercept_decay = 1.0`.
- **Rationale**: Reuses a single robust CSR loader in C++. Mathematically equivalent because dot products iterate over identical column indices.

### Decision 9: Primary Data Source for King County Dataset
- **Date**: 2026-10-23
- **Report Statement**: Kaggle or OpenML 42092.
- **Adopted Specification**: OpenML dataset ID 42092 as primary source (requires no API credentials or interactive login), with local/fallback support. Enforced assertion of 21,613 rows and exactly 17 expected feature columns.
- **Rationale**: Ensures reproducible, automated environment setup without third-party account requirements.

### Decision 10: scikit-learn Version Pinning
- **Date**: 2026-10-23
- **Report Statement**: Pin to scikit-learn 1.9.1.
- **Adopted Specification**: Pin to the exact verified version installed in the virtual environment (scikit-learn 1.6.1 / 1.6.x as available from PyPI) and conduct a comprehensive Cython source audit on that exact version.
- **Rationale**: Does not claim versions that cannot be validated.

### Decision 11: Bottou Schedule Offset
- **Date**: 2026-10-23
- **Report Statement**: Formula `gamma_0 * (1 + gamma_0 * lambda * t)^(-1)`.
- **Adopted Specification**: Implemented with `t_prime = t - 1`, i.e., `eta_t = gamma_0 / (1 + gamma_0 * alpha * (t - 1))`.
- **Rationale**: Ensures that at step 1 (`t = 1`), `eta_1 = gamma_0`. When `gamma_0 = 1 / alpha`, `eta_t = 1 / (alpha * t)` exactly matches Pegasos.

### Decision 12: Visualization Axes Formatting
- **Date**: 2026-10-23
- **Report Statement**: Listed axes only.
- **Adopted Specification**: Figure 2 uses log-x (5 points spanning 3 orders of magnitude) and log-y; error bars represent IQR (interquartile range).
- **Rationale**: Maximizes readability and accurately depicts scaling across multiple density orders of magnitude.

### Decision 13: Pre-registered Hypothesis Testing Criteria
- **Date**: 2026-10-23
- **Report Statement**: No formal acceptance thresholds specified prior to testing.
- **Adopted Specification**: Adopted strict pre-registered criteria (Section 8.3 of `plan.md`).
- **Rationale**: Eliminates post-hoc hypothesis framing ("p-hacking" or moving the goalposts).

### Decision 14: Verification Testing Ladder
- **Date**: 2026-10-23
- **Report Statement**: No unit tests specified.
- **Adopted Specification**: Introduced step ladder tests (N in {1, 2, 3, 10, 100, 1000}) and targeted component tests (Section 12 of `plan.md`).
- **Rationale**: Isolates the exact step at which any numerical divergence occurs between C++ and scikit-learn.

### Decision 15: Adaptive Rejection vs. Partition Sampling for Synthetic Data
- **Date**: 2026-10-23
- **Report Statement**: Report proposed threshold `k / n <= 0.1` for rejection sampling based on approximation `k^2 / (2n)` being small.
- **Adopted Specification**: Use rejection sampling strictly when `(k^2) / (2n) < 0.4` and `k <= 50`. For higher densities or large `k` (such as `k = 500, n = 5000`), switch to `argpartition` sampling over bounded sub-chunks.
- **Rationale**: At `k = 500, n = 5000`, collision probability is $1 - e^{-25} \approx 1 - 10^{-11}$, which causes near 100% rejection in rejection sampling and stalls generation indefinitely. Partition sampling eliminates rejection entirely with bounded memory consumption.

### Decision 16: OpenML King County ARFF Categorical Parsing Resolution
- **Date**: 2026-10-23
- **Report Statement**: Fetch King County via `fetch_openml(data_id=42092)`.
- **Adopted Specification**: Download and directly parse the OpenML ARFF archive via `load_king_county` in `prep_rq2.py`.
- **Rationale**: scikit-learn 1.9.1's `_pandas_arff_parser` contains a bug where integer categorical categories in the ARFF definition (`@ATTRIBUTE zipcode {98001, ...}`) trigger a `TypeError: expected string or bytes-like object, got 'int'` in `strip_single_quotes`. Directly streaming and parsing the raw ARFF table into pandas circumvents the bug while retrieving the exact identical raw OpenML dataset and verifying all 21,613 rows and 17 feature columns.

### Decision 17: Pilot Calibration and Final Equivalence Thresholds
- **Date**: 2026-10-23
- **Observed Metrics in Pilot (Phase 5)**:
  - C++ Direct timing noise IQR/median: 0.48% to 0.63% (well within < 5% target).
  - C++ Scaled timing noise IQR/median: 1.60% to 4.14% (< 5% target).
  - scikit-learn per-epoch slope $R^2$: > 0.998 across all runs.
  - Direct vs. Scaled weight relative discrepancy: $1.72 \times 10^{-15}$ to $5.31 \times 10^{-15}$.
  - Scaled vs. scikit-learn weight relative discrepancy: $0.00 \times 10^0$ (exact bitwise parity).
  - Test RMSE relative discrepancy across 100 epochs: $0.00 \times 10^0$.
- **Final Calibrated Thresholds**:
  - Weight discrepancy: $\|\Delta w\| / \|w\| < 10^{-10}$ (enforced for both C++ direct vs. scaled and C++ scaled vs. scikit-learn).
  - Intercept discrepancy: $|\Delta b| < 10^{-9}$.
  - Test RMSE relative discrepancy: $|\Delta \text{RMSE}| / \text{RMSE} < 10^{-6}$.
- **Runtime Estimates**: Full RQ1 benchmark (12 datasets x 3 methods): ~3 to 5 minutes; Full RQ2 benchmark (3 datasets x 5 epochs): ~1 minute.



