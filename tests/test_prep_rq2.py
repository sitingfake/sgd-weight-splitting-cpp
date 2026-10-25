"""
Unit tests for RQ2 housing datasets preprocessing.
"""

from pathlib import Path
import numpy as np
import pytest

from sgdrepro.binio import read_csr_binary


@pytest.mark.parametrize("dataset_name,expected_samples,expected_features,expected_train", [
    ("taiwan", 414, 6, 331),
    ("california", 20640, 8, 16512),
    ("kingcounty", 21613, 17, 17290),
])
def test_prep_rq2_dataset(dataset_name, expected_samples, expected_features, expected_train):
    """Verify shapes, non-emptiness, split consistency, and CSR bitwise equality."""
    dataset_dir = Path("data/processed/rq2") / dataset_name
    assert dataset_dir.exists(), f"Processed directory {dataset_dir} does not exist"

    # 1. Check split.npz
    split_file = dataset_dir / "split.npz"
    assert split_file.exists()
    with np.load(split_file) as split_data:
        train_idx = split_data["train_idx"]
        test_idx = split_data["test_idx"]
        assert len(train_idx) == expected_train
        assert len(train_idx) + len(test_idx) == expected_samples
        # Check indices are disjoint and within range
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert np.min(train_idx) >= 0 and np.max(train_idx) < expected_samples
        assert np.min(test_idx) >= 0 and np.max(test_idx) < expected_samples

    # 2. Check arrays.npz
    arrays_file = dataset_dir / "arrays.npz"
    assert arrays_file.exists()
    with np.load(arrays_file) as arr_data:
        X_tr = arr_data["X_tr"]
        y_tr_std = arr_data["y_tr_std"]
        X_te = arr_data["X_te"]
        y_te_orig = arr_data["y_te_orig"]

        assert X_tr.shape == (expected_train, expected_features)
        assert len(y_tr_std) == expected_train
        assert X_te.shape == (expected_samples - expected_train, expected_features)
        assert len(y_te_orig) == expected_samples - expected_train

        # Check finite and non-constant
        assert np.isfinite(X_tr).all()
        assert np.isfinite(y_tr_std).all()
        assert np.isfinite(X_te).all()
        assert np.isfinite(y_te_orig).all()
        assert (np.std(X_tr, axis=0) > 0.0).all()

        # Check mean ~ 0 and std ~ 1 on train
        np.testing.assert_allclose(np.mean(X_tr, axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(X_tr, axis=0), 1.0, atol=1e-7)
        np.testing.assert_allclose(np.mean(y_tr_std), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(y_tr_std), 1.0, atol=1e-7)

    # 3. Check train.bin matches arrays.npz exactly bit-for-bit
    bin_file = dataset_dir / "train.bin"
    assert bin_file.exists()
    X_csr, y_csr = read_csr_binary(bin_file)

    assert X_csr.shape == (expected_train, expected_features)
    assert len(y_csr) == expected_train
    np.testing.assert_array_equal(X_csr.toarray(), X_tr)
    np.testing.assert_array_equal(y_csr, y_tr_std)
