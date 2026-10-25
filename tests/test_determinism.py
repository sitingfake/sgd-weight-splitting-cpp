"""
Unit tests confirming determinism across runs and repeat iterations.
"""

from pathlib import Path
import numpy as np
import pytest

from sgdrepro.run_cpp import run_cpp_sgd


def test_cpp_determinism():
    """Verify that multiple independent executions yield bit-for-bit identical weights."""
    data_path = Path("data/processed/rq1/n1000_k20_m10000.bin")
    assert data_path.exists(), "Dataset n1000_k20_m10000.bin must exist"

    # Run scaled 1
    res1, w1 = run_cpp_sgd(
        data_path, impl="scaled", epochs=2, alpha=1e-4,
        schedule="invscaling", warmup=0, repeats=2
    )

    # Run scaled 2
    res2, w2 = run_cpp_sgd(
        data_path, impl="scaled", epochs=2, alpha=1e-4,
        schedule="invscaling", warmup=0, repeats=2
    )

    assert not res1.get("diverged", False)
    assert not res2.get("diverged", False)

    # Exact bit-level equality
    np.testing.assert_array_equal(w1, w2)
    assert res1["b"] == res2["b"]
