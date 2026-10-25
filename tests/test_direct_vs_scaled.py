"""
Unit tests comparing C++ Direct vs. C++ Scaled updates (w = s * v).
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest

from sgdrepro.binio import write_csr_binary
from sgdrepro.metrics import relative_weight_difference
from sgdrepro.run_cpp import run_cpp_sgd


def test_direct_vs_scaled_standard():
    """Verify direct and scaled produce identical weights under standard settings."""
    data_path = Path("data/processed/rq1/n1000_k20_m10000.bin")
    assert data_path.exists()

    res_dir, w_dir = run_cpp_sgd(
        data_path, impl="direct", epochs=3, alpha=1e-4,
        schedule="invscaling", eta0=0.01, power_t=0.25,
        fit_intercept=False, warmup=0, repeats=1
    )
    res_sc, w_sc = run_cpp_sgd(
        data_path, impl="scaled", epochs=3, alpha=1e-4,
        schedule="invscaling", eta0=0.01, power_t=0.25,
        fit_intercept=False, warmup=0, repeats=1
    )

    rel_diff = relative_weight_difference(w_dir, w_sc)
    assert rel_diff < 1e-10, f"Relative difference {rel_diff} exceeds 1e-10"


def test_direct_vs_scaled_forced_reset():
    """
    Force scale factor s < 1e-9 by configuring large alpha (alpha = 10.0, eta0 = 0.1).
    Ensures that reset mechanism triggers (n_resets > 0) while maintaining relative
    error below 1e-10.
    """
    m, n = 500, 50
    rng = np.random.default_rng(777)
    X = sp.random(m, n, density=0.2, format="csr", dtype=np.float64, random_state=42)
    # Normalize rows
    for i in range(m):
        row_slice = slice(X.indptr[i], X.indptr[i + 1])
        norm = np.linalg.norm(X.data[row_slice])
        if norm > 0:
            X.data[row_slice] /= norm
    y = rng.standard_normal(m, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "forced_reset.bin"
        write_csr_binary(data_path, X, y)

        res_dir, w_dir = run_cpp_sgd(
            data_path, impl="direct", epochs=5, alpha=10.0,
            schedule="invscaling", eta0=0.1, power_t=0.25,
            fit_intercept=False, warmup=0, repeats=1
        )
        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=5, alpha=10.0,
            schedule="invscaling", eta0=0.1, power_t=0.25,
            fit_intercept=False, warmup=0, repeats=1
        )

        assert res_sc["n_resets"] > 0, f"Expected n_resets > 0, got {res_sc['n_resets']}"
        rel_diff = relative_weight_difference(w_dir, w_sc)
        assert rel_diff < 1e-10, f"Relative difference under reset {rel_diff} exceeds 1e-10"
