"""
Unit tests for RQ1 synthetic data generator.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest

from sgdrepro.gen_rq1 import generate_sparse_dataset
from sgdrepro.binio import write_csr_binary


def test_gen_rq1_properties():
    """Verify row normalization, index sorting, uniqueness, and noise standard deviation."""
    n = 1000
    k = 20
    m = 200
    seed = 42

    X, y, w_star = generate_sparse_dataset(n, k, m, seed=seed)

    assert X.shape == (m, n)
    assert X.nnz == m * k
    assert len(y) == m
    assert len(w_star) == n

    # Check 1: Euclidean norm of each row is 1.0 (within machine precision)
    for i in range(m):
        row_slice = slice(X.indptr[i], X.indptr[i + 1])
        row_indices = X.indices[row_slice]
        row_values = X.data[row_slice]

        # Exactly k elements
        assert len(row_indices) == k
        # Strictly sorted ascending with unique entries
        assert np.all(np.diff(row_indices) > 0)
        # Unit norm
        row_norm = np.linalg.norm(row_values)
        np.testing.assert_allclose(row_norm, 1.0, atol=1e-12)

    # Check 2: Noise standard deviation of y - X * w* is ~0.1
    residuals = y - X.dot(w_star)
    noise_std = np.std(residuals)
    assert 0.08 < noise_std < 0.12, f"Observed noise std {noise_std} outside expected [0.08, 0.12]"


def test_gen_rq1_determinism():
    """Verify that identical parameters and seed produce bitwise identical datasets and hashes."""
    n = 500
    k = 15
    m = 100
    seed = 999

    X1, y1, _ = generate_sparse_dataset(n, k, m, seed=seed)
    X2, y2, _ = generate_sparse_dataset(n, k, m, seed=seed)

    np.testing.assert_array_equal(X1.indptr, X2.indptr)
    np.testing.assert_array_equal(X1.indices, X2.indices)
    np.testing.assert_array_equal(X1.data, X2.data)
    np.testing.assert_array_equal(y1, y2)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "d1.bin"
        p2 = Path(tmpdir) / "d2.bin"
        h1 = write_csr_binary(p1, X1, y1)
        h2 = write_csr_binary(p2, X2, y2)
        assert h1 == h2
