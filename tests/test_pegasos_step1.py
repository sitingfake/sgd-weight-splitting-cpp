"""
Unit tests for Pegasos schedule boundary conditions (step 1 shrinkage = 0).
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest

from sgdrepro.binio import write_csr_binary
from sgdrepro.run_cpp import run_cpp_sgd


def test_pegasos_step1_exact_update():
    """
    At step 1 of Pegasos (t = 1, eta = 1 / alpha):
      shrink = max(0, 1 - eta * alpha) = 0.
      Prior to update: w_0 = 0 -> p_1 = 0.
      Gradient: g = p - y_1 = -y_1.
      Update: u = -eta * g = eta * y_1 = (1 / alpha) * y_1.
      Final weight vector: w_1 = (1 / alpha) * y_1 * x_1.
    Verify both C++ direct and C++ scaled match this exact analytical step.
    """
    n = 10
    alpha = 0.05
    eta1 = 1.0 / alpha  # 20.0

    # Single sample (m = 1)
    x1 = np.zeros((1, n), dtype=np.float64)
    x1[0, 2] = 0.6
    x1[0, 7] = 0.8
    y1 = np.array([3.5], dtype=np.float64)

    expected_w1 = eta1 * y1[0] * x1[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "pegasos_m1.bin"
        write_csr_binary(data_path, sp.csr_matrix(x1), y1)

        # Run direct
        res_dir, w_dir = run_cpp_sgd(
            data_path, impl="direct", epochs=1, alpha=alpha,
            schedule="pegasos", warmup=0, repeats=1
        )
        assert not res_dir.get("diverged", False)
        np.testing.assert_allclose(w_dir, expected_w1, rtol=1e-12, atol=1e-12)

        # Run scaled (verifies s == 0 reset behavior)
        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=1, alpha=alpha,
            schedule="pegasos", warmup=0, repeats=1
        )
        assert not res_sc.get("diverged", False)
        np.testing.assert_allclose(w_sc, expected_w1, rtol=1e-12, atol=1e-12)

        # Direct vs Scaled equality
        np.testing.assert_allclose(w_sc, w_dir, rtol=1e-14, atol=1e-14)
