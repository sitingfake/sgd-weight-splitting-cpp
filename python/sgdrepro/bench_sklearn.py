"""
scikit-learn SGDRegressor Timing and Benchmark Harness.

Implements Section 7 timing protocol:
  - Fresh estimator per measurement, gc.disable() during timing, time.perf_counter().
  - For each repetition (1 warmup + 7 timed), measures T(E) for E = 1..5 epochs.
  - Linear regression T = a + b * E determines per-epoch slope b and fixed overhead a.
  - Computes median and IQR of b across repetitions, and reports us/update = (b / m) * 1e6.
"""

import gc
import time
import warnings
from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor


def measure_sklearn_slope(
    X: sp.csr_matrix | np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-4,
    fit_intercept: bool = False,
    schedule: str = "invscaling",
    eta0: float = 0.01,
    power_t: float = 0.25,
    epochs_range: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> Tuple[float, float, float]:
    """
    Measure execution time across E = 1..5 epochs and compute slope b (seconds per epoch).

    Returns
    -------
    slope_b : float
        Fitted seconds per epoch.
    intercept_a : float
        Fitted fixed overhead in seconds.
    r2_fit : float
        Goodness of fit R^2 for T = a + b * E.
    """
    times = []
    epochs_arr = np.array(epochs_range, dtype=np.float64)

    # Filter convergence warnings as tol=None triggers warning by design when max_iter reached
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for E in epochs_range:
            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=E,
                tol=None,
                shuffle=False,
                learning_rate=schedule,
                eta0=eta0,
                power_t=power_t,
                random_state=42,
            )

            gc.collect()
            gc.disable()
            try:
                t0 = time.perf_counter()
                reg.fit(X, y)
                t1 = time.perf_counter()
            finally:
                gc.enable()

            times.append(t1 - t0)

    times_arr = np.array(times, dtype=np.float64)

    # Linear regression: times = a + b * epochs
    p = np.polyfit(epochs_arr, times_arr, 1)
    slope_b = float(p[0])
    intercept_a = float(p[1])

    # Goodness of fit R^2
    y_pred = intercept_a + slope_b * epochs_arr
    ss_res = np.sum((times_arr - y_pred) ** 2)
    ss_tot = np.sum((times_arr - np.mean(times_arr)) ** 2)
    r2_fit = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

    return slope_b, intercept_a, r2_fit


def benchmark_sklearn_sgd(
    X: sp.csr_matrix | np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    epochs: int = 3,
    alpha: float = 1e-4,
    fit_intercept: bool = False,
    schedule: str = "invscaling",
    eta0: float = 0.01,
    power_t: float = 0.25,
    warmup: int = 1,
    repeats: int = 7,
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    """
    Execute full scikit-learn benchmark over warmup and repeated slope measurements.

    Returns
    -------
    record : dict
        Benchmark metadata and timing summary.
    w : np.ndarray
        Fitted weight array for the target epoch count.
    b : float
        Fitted intercept term.
    """
    m, n = X.shape
    nnz = X.nnz if sp.issparse(X) else m * n

    # Warmup
    for _ in range(warmup):
        measure_sklearn_slope(
            X, y, alpha=alpha, fit_intercept=fit_intercept,
            schedule=schedule, eta0=eta0, power_t=power_t
        )

    # Timed repeats
    slopes = []
    overheads = []
    r2_list = []
    for _ in range(repeats):
        b_slope, a_overhead, r2 = measure_sklearn_slope(
            X, y, alpha=alpha, fit_intercept=fit_intercept,
            schedule=schedule, eta0=eta0, power_t=power_t
        )
        slopes.append(b_slope)
        overheads.append(a_overhead)
        r2_list.append(r2)

    slopes_arr = np.array(slopes, dtype=np.float64)
    median_slope = float(np.median(slopes_arr))
    q75, q25 = np.percentile(slopes_arr, [75, 25])
    iqr_slope = float(q75 - q25)
    us_per_update = float((median_slope / m) * 1e6)

    # Fit final model at target epochs to retrieve weights
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_reg = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=epochs,
            tol=None,
            shuffle=False,
            learning_rate=schedule,
            eta0=eta0,
            power_t=power_t,
            random_state=42,
        )
        final_reg.fit(X, y)

    w = np.ascontiguousarray(final_reg.coef_, dtype=np.float64)
    b = float(final_reg.intercept_[0] if fit_intercept else 0.0)

    # Total loop nanoseconds equivalent for target epochs
    loop_ns = [int(s * epochs * 1e9) for s in slopes]
    median_ns = int(median_slope * epochs * 1e9)
    iqr_ns = int(iqr_slope * epochs * 1e9)

    record = {
        "impl": "sklearn_sgd",
        "dataset": dataset_name,
        "m": m,
        "n": n,
        "nnz": nnz,
        "params": {
            "alpha": alpha,
            "schedule": schedule,
            "eta0": eta0,
            "power_t": power_t,
            "optimal_init": 0.0,
            "epochs": epochs,
            "fit_intercept": fit_intercept,
            "intercept_decay": 0.01 if sp.issparse(X) else 1.0,
        },
        "timing": {
            "warmup": warmup,
            "repeats": repeats,
            "loop_ns": loop_ns,
            "median_ns": median_ns,
            "iqr_ns": iqr_ns,
            "min_ns": int(np.min(slopes_arr) * epochs * 1e9),
            "max_ns": int(np.max(slopes_arr) * epochs * 1e9),
            "us_per_update": us_per_update,
            "finalize_ns": 0,
            "slope_s_per_epoch": median_slope,
            "overhead_s": float(np.median(overheads)),
            "linear_fit_r2_median": float(np.median(r2_list)),
        },
        "n_resets": 0,
        "diverged": False,
        "diverged_epoch": 0,
        "b": b,
        "build": {
            "compiler": "Cython / scikit-learn",
            "flags": "bundled wheel",
        },
    }

    return record, w, b
