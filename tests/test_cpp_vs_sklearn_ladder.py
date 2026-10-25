"""
Step-Ladder Verification Gate: C++ vs. scikit-learn SGDRegressor.

Executes the four verification ladder configurations across N in {1, 2, 3, 10, 100, 1000}
for 1 epoch, and tests 100-epoch convergence across Taiwan, California, and King County.
Ensures ||Delta w|| / ||w|| < 1e-10 and |Delta RMSE| / RMSE < 1e-6 before benchmarking begins.
"""

import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor
import pytest

from sgdrepro.binio import write_csr_binary
from sgdrepro.metrics import relative_weight_difference, rmse
from sgdrepro.run_cpp import run_cpp_sgd


def fit_sklearn_sgd(
    X,
    y,
    epochs: int = 1,
    alpha: float = 1e-4,
    schedule: str = "invscaling",
    eta0: float = 0.01,
    power_t: float = 0.25,
    fit_intercept: bool = True,
):
    """Run scikit-learn SGDRegressor with deterministic parameters matching C++."""
    reg = SGDRegressor(
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
    reg.fit(X, y)
    return np.ascontiguousarray(reg.coef_, dtype=np.float64), float(reg.intercept_[0] if fit_intercept else 0.0)


@pytest.mark.parametrize("N", [1, 2, 3, 10, 100, 1000])
def test_ladder_config_a_dense_with_intercept(N):
    """
    Config (a): Dense dataset with intercept (intercept_decay = 1.0).
    """
    rng = np.random.default_rng(100 + N)
    n = 20
    X_dense = rng.standard_normal((N, n))
    y = rng.standard_normal(N)

    w_sk, b_sk = fit_sklearn_sgd(X_dense, y, epochs=1, fit_intercept=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / f"dense_a_{N}.bin"
        write_csr_binary(data_path, sp.csr_matrix(X_dense), y)

        # C++ Scaled
        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=1, fit_intercept=True,
            intercept_decay=1.0, warmup=0, repeats=1
        )
        # C++ Direct
        res_dir, w_dir = run_cpp_sgd(
            data_path, impl="direct", epochs=1, fit_intercept=True,
            intercept_decay=1.0, warmup=0, repeats=1
        )

        rel_w_sc = relative_weight_difference(w_sk, w_sc)
        rel_w_dir = relative_weight_difference(w_sk, w_dir)
        rel_w_inter = relative_weight_difference(w_sc, w_dir)

        assert rel_w_sc < 1e-10, f"N={N} Scaled vs Sklearn rel diff {rel_w_sc} >= 1e-10"
        assert rel_w_dir < 1e-10, f"N={N} Direct vs Sklearn rel diff {rel_w_dir} >= 1e-10"
        assert rel_w_inter < 1e-12, f"N={N} Scaled vs Direct diff {rel_w_inter} >= 1e-12"
        assert abs(b_sk - res_sc["b"]) < 1e-10, f"N={N} Scaled b diff {abs(b_sk - res_sc['b'])}"
        assert abs(b_sk - res_dir["b"]) < 1e-10, f"N={N} Direct b diff {abs(b_sk - res_dir['b'])}"


@pytest.mark.parametrize("N", [1, 2, 3, 10, 100, 1000])
def test_ladder_config_b_sparse_with_intercept(N):
    """
    Config (b): Sparse CSR with intercept (intercept_decay = 0.01 in sklearn).
    Verifies that passing sparse CSR to sklearn triggers intercept_decay = 0.01.
    """
    rng = np.random.default_rng(200 + N)
    n = 30
    X_csr = sp.random(N, n, density=0.2, format="csr", dtype=np.float64, random_state=42)
    y = rng.standard_normal(N)

    w_sk, b_sk = fit_sklearn_sgd(X_csr, y, epochs=1, fit_intercept=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / f"sparse_b_{N}.bin"
        write_csr_binary(data_path, X_csr, y)

        # C++ Scaled with intercept_decay = 0.01
        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=1, fit_intercept=True,
            intercept_decay=0.01, warmup=0, repeats=1
        )

        rel_w_sc = relative_weight_difference(w_sk, w_sc)
        assert rel_w_sc < 1e-10, f"N={N} Config B rel diff {rel_w_sc} >= 1e-10"
        assert abs(b_sk - res_sc["b"]) < 1e-10, f"N={N} Config B b diff {abs(b_sk - res_sc['b'])}"


@pytest.mark.parametrize("N", [1, 2, 3, 10, 100, 1000])
def test_ladder_config_c_sparse_no_intercept(N):
    """
    Config (c): Sparse CSR without intercept (RQ1 configuration).
    """
    rng = np.random.default_rng(300 + N)
    n = 50
    X_csr = sp.random(N, n, density=0.15, format="csr", dtype=np.float64, random_state=123)
    y = rng.standard_normal(N)

    w_sk, b_sk = fit_sklearn_sgd(X_csr, y, epochs=1, fit_intercept=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / f"sparse_c_{N}.bin"
        write_csr_binary(data_path, X_csr, y)

        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=1, fit_intercept=False,
            warmup=0, repeats=1
        )

        rel_w_sc = relative_weight_difference(w_sk, w_sc)
        assert rel_w_sc < 1e-10, f"N={N} Config C rel diff {rel_w_sc} >= 1e-10"
        assert b_sk == 0.0
        assert res_sc["b"] == 0.0


@pytest.mark.parametrize("N", [10, 100, 500])
def test_ladder_config_d_forced_reset(N):
    """
    Config (d): Large alpha (alpha = 10, eta0 = 0.1) forcing scale factor s resets.
    """
    rng = np.random.default_rng(400 + N)
    n = 20
    X_dense = rng.standard_normal((N, n))
    # Row normalize
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_dense /= norms
    y = rng.standard_normal(N)

    w_sk, b_sk = fit_sklearn_sgd(
        X_dense, y, epochs=1, alpha=10.0, eta0=0.1, fit_intercept=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / f"forced_d_{N}.bin"
        write_csr_binary(data_path, sp.csr_matrix(X_dense), y)

        res_sc, w_sc = run_cpp_sgd(
            data_path, impl="scaled", epochs=1, alpha=10.0, eta0=0.1,
            fit_intercept=False, warmup=0, repeats=1
        )

        assert res_sc["n_resets"] > 0
        rel_w_sc = relative_weight_difference(w_sk, w_sc)
        assert rel_w_sc < 1e-10, f"N={N} Config D rel diff {rel_w_sc} >= 1e-10"


@pytest.mark.parametrize("dataset_name", ["taiwan", "california", "kingcounty"])
def test_rq2_datasets_1_epoch_and_100_epochs(dataset_name):
    """
    Verify complete numerical parity on all three empirical datasets for 1 and 100 epochs.
    Checks ||Delta w|| / ||w|| < 1e-10 and |Delta RMSE| / RMSE < 1e-6.
    """
    data_dir = Path("data/processed/rq2") / dataset_name
    bin_path = data_dir / "train.bin"
    arrays_path = data_dir / "arrays.npz"

    with np.load(arrays_path) as arr:
        X_tr = arr["X_tr"]
        y_tr_std = arr["y_tr_std"]
        X_te = arr["X_te"]
        y_te_orig = arr["y_te_orig"]
        y_mean = float(arr["y_mean"])
        y_std = float(arr["y_std"])

    for epochs in [1, 100]:
        # Fit scikit-learn on dense array
        w_sk, b_sk = fit_sklearn_sgd(X_tr, y_tr_std, epochs=epochs, fit_intercept=True)

        # Fit C++ Scaled on binary CSR (with intercept_decay = 1.0)
        res_sc, w_sc = run_cpp_sgd(
            bin_path, impl="scaled", epochs=epochs, fit_intercept=True,
            intercept_decay=1.0, warmup=0, repeats=1
        )

        # Fit C++ Direct on binary CSR
        res_dir, w_dir = run_cpp_sgd(
            bin_path, impl="direct", epochs=epochs, fit_intercept=True,
            intercept_decay=1.0, warmup=0, repeats=1
        )

        # Evaluate predictions on test set in original units
        pred_sk_std = X_te.dot(w_sk) + b_sk
        pred_sk_orig = pred_sk_std * y_std + y_mean
        rmse_sk = rmse(y_te_orig, pred_sk_orig)

        pred_sc_std = X_te.dot(w_sc) + res_sc["b"]
        pred_sc_orig = pred_sc_std * y_std + y_mean
        rmse_sc = rmse(y_te_orig, pred_sc_orig)

        pred_dir_std = X_te.dot(w_dir) + res_dir["b"]
        pred_dir_orig = pred_dir_std * y_std + y_mean
        rmse_dir = rmse(y_te_orig, pred_dir_orig)

        # Weights relative discrepancy
        rel_w_sc = relative_weight_difference(w_sk, w_sc)
        rel_w_dir = relative_weight_difference(w_sk, w_dir)
        rel_w_inter = relative_weight_difference(w_sc, w_dir)

        # RMSE relative discrepancy
        rel_rmse_sc = abs(rmse_sc - rmse_sk) / rmse_sk
        rel_rmse_dir = abs(rmse_dir - rmse_sk) / rmse_sk

        assert rel_w_sc < 1e-10, f"{dataset_name} E={epochs} Scaled w diff {rel_w_sc} >= 1e-10"
        assert rel_w_dir < 1e-10, f"{dataset_name} E={epochs} Direct w diff {rel_w_dir} >= 1e-10"
        assert rel_w_inter < 1e-12, f"{dataset_name} E={epochs} Scaled vs Direct diff {rel_w_inter} >= 1e-12"

        assert rel_rmse_sc < 1e-6, f"{dataset_name} E={epochs} Scaled RMSE diff {rel_rmse_sc} >= 1e-6"
        assert rel_rmse_dir < 1e-6, f"{dataset_name} E={epochs} Direct RMSE diff {rel_rmse_dir} >= 1e-6"
        assert abs(b_sk - res_sc["b"]) < 1e-9, f"{dataset_name} E={epochs} b diff {abs(b_sk - res_sc['b'])}"
