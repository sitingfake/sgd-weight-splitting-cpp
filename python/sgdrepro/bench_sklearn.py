"""Fixed-epoch training benchmark for scikit-learn SGDRegressor."""

import warnings
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor

from sgdrepro.evaluation import benchmark_fit


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
    """Measure the requested fit and return its weights and intercept."""
    def make_estimator():
        return SGDRegressor(
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model, fit_timing = benchmark_fit(make_estimator, X, y, warmup, repeats)

    m, n = X.shape
    is_sparse = sp.issparse(X)
    w = np.ascontiguousarray(model.coef_, dtype=np.float64)
    b = float(model.intercept_[0])
    record = {
        "impl": "sklearn_sgd",
        "dataset": dataset_name,
        "m": m,
        "n": n,
        "nnz": X.nnz if is_sparse else m * n,
        "params": {
            "alpha": alpha,
            "schedule": schedule,
            "eta0": eta0,
            "power_t": power_t,
            "optimal_init": 0.0,
            "epochs": epochs,
            "fit_intercept": fit_intercept,
            "intercept_decay": 0.01 if is_sparse else 1.0,
        },
        "timing": {
            **fit_timing,
            "scope": "fit",
            "min_ns": min(fit_timing["samples_ns"]),
            "max_ns": max(fit_timing["samples_ns"]),
            "us_per_update": fit_timing["median_ns"] / (m * epochs * 1000),
        },
        "fit_timing": fit_timing,
        "n_resets": None,
        "diverged": False,
        "diverged_epoch": 0,
        "b": b,
        "build": {"compiler": "Cython / scikit-learn", "flags": "bundled wheel"},
    }
    return record, w, b
