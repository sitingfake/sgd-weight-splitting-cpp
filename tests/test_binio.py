"""
Unit tests for binary CSR I/O (SGDCSR01 format).
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pytest

from sgdrepro.binio import write_csr_binary, read_csr_binary, sha256_file


def test_binio_roundtrip():
    """Verify exact bit-level roundtrip between in-memory CSR and written SGDCSR01 file."""
    m, n = 50, 100
    density = 0.1
    rng = np.random.default_rng(12345)

    X_dense = rng.standard_normal((m, n))
    mask = rng.random((m, n)) < density
    X_dense[~mask] = 0.0

    X_csr = sp.csr_matrix(X_dense, dtype=np.float64)
    y = rng.standard_normal(m, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.bin"
        written_hash = write_csr_binary(filepath, X_csr, y)
        assert filepath.exists()
        assert sha256_file(filepath) == written_hash

        # Read back
        X_loaded, y_loaded = read_csr_binary(filepath)

        # Assert bit-for-bit equivalence
        assert X_loaded.shape == X_csr.shape
        assert X_loaded.nnz == X_csr.nnz
        np.testing.assert_array_equal(X_loaded.indptr, X_csr.indptr)
        np.testing.assert_array_equal(X_loaded.indices, X_csr.indices)
        np.testing.assert_array_equal(X_loaded.data, X_csr.data)
        np.testing.assert_array_equal(y_loaded, y)


def test_binio_truncation():
    """Verify max_samples subset reading."""
    m, n = 100, 20
    X_csr = sp.random(m, n, density=0.2, format="csr", dtype=np.float64, random_state=42)
    y = np.linspace(1.0, 100.0, m, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_trunc.bin"
        write_csr_binary(filepath, X_csr, y)

        X_sub, y_sub = read_csr_binary(filepath, max_samples=25)
        assert X_sub.shape == (25, n)
        assert len(y_sub) == 25
        np.testing.assert_array_equal(y_sub, y[:25])
        np.testing.assert_array_equal(X_sub.toarray(), X_csr[:25].toarray())
