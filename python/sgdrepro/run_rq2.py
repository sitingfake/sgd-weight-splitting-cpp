"""
RQ2 Experiment Runner: Convergence Quality and Ridge Trajectory Analysis.

Evaluates C++ Scaled vs. scikit-learn SGDRegressor vs. Closed-Form Ridge Reference
across Taiwan, California, and King County datasets for epochs in {1, 5, 20, 100, 500}.

Computes RMSE, MAE, R^2, relative distance to Ridge, and equivalence pass flags.
"""

import argparse
import json
import time
import warnings
from pathlib import Path
import tomllib
from typing import Any, Dict, List
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor

from sgdrepro.metrics import compute_metrics, relative_weight_difference
from sgdrepro.ridge import distance_to_ridge, fit_ridge_reference
from sgdrepro.run_cpp import run_cpp_sgd


def run_rq2_dataset(
    dataset_name: str,
    data_dir: Path,
    out_dir: Path,
    epochs_sweep: List[int],
    alpha: float = 1e-4,
    eta0: float = 0.01,
    power_t: float = 0.25,
) -> List[Dict[str, Any]]:
    """Execute Ridge, scikit-learn, and C++ Scaled across epoch sweep for one dataset."""
    print(f"\n=======================================================")
    print(f"Running RQ2 Experiments on: {dataset_name}")
    print(f"=======================================================")

    ds_dir = data_dir / dataset_name
    bin_path = ds_dir / "train.bin"
    arrays_path = ds_dir / "arrays.npz"

    with np.load(arrays_path) as arr:
        X_tr = arr["X_tr"]
        y_tr_std = arr["y_tr_std"]
        X_te = arr["X_te"]
        y_te_orig = arr["y_te_orig"]
        y_mean = float(arr["y_mean"])
        y_std = float(arr["y_std"])

    m_train, n_features = X_tr.shape
    m_test = X_te.shape[0]

    # 1. Closed-Form Ridge Reference Solution
    print("  -> Computing Closed-Form Ridge Reference...")
    t0_ridge = time.perf_counter()
    w_ridge, b_ridge = fit_ridge_reference(X_tr, y_tr_std, alpha_sgd=alpha)
    t1_ridge = time.perf_counter()
    time_ridge_ms = (t1_ridge - t0_ridge) * 1000.0

    pred_ridge_std = X_te.dot(w_ridge) + b_ridge
    pred_ridge_orig = pred_ridge_std * y_std + y_mean
    metrics_ridge = compute_metrics(y_te_orig, pred_ridge_orig)

    ridge_record = {
        "impl": "ridge_reference",
        "dataset": dataset_name,
        "m_train": m_train,
        "m_test": m_test,
        "n": n_features,
        "alpha_sgd": alpha,
        "alpha_ridge": float(m_train * alpha),
        "metrics": metrics_ridge,
        "distance_to_ridge": 0.0,
        "train_time_ms": time_ridge_ms,
        "b": b_ridge,
    }
    with open(out_dir / f"{dataset_name}_ridge.json", "w", encoding="utf-8") as f:
        json.dump(ridge_record, f, indent=2)

    results_table = []

    # 2. Sweep across epoch counts
    for E in epochs_sweep:
        print(f"  --- Epoch E = {E} ---")

        # (a) scikit-learn SGDRegressor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            sk_reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=alpha,
                fit_intercept=True,
                max_iter=E,
                tol=None,
                shuffle=False,
                learning_rate="invscaling",
                eta0=eta0,
                power_t=power_t,
                random_state=42,
            )
            t0_sk = time.perf_counter()
            sk_reg.fit(X_tr, y_tr_std)
            t1_sk = time.perf_counter()

        time_sk_ms = (t1_sk - t0_sk) * 1000.0
        w_sk = np.ascontiguousarray(sk_reg.coef_, dtype=np.float64)
        b_sk = float(sk_reg.intercept_[0])

        pred_sk_std = X_te.dot(w_sk) + b_sk
        pred_sk_orig = pred_sk_std * y_std + y_mean
        metrics_sk = compute_metrics(y_te_orig, pred_sk_orig)
        dist_ridge_sk = distance_to_ridge(w_sk, w_ridge)

        sk_record = {
            "impl": "sklearn_sgd",
            "dataset": dataset_name,
            "epochs": E,
            "m_train": m_train,
            "m_test": m_test,
            "n": n_features,
            "alpha": alpha,
            "metrics": metrics_sk,
            "distance_to_ridge": dist_ridge_sk,
            "train_time_ms": time_sk_ms,
            "b": b_sk,
        }
        with open(out_dir / f"{dataset_name}_E{E}_sklearn.json", "w", encoding="utf-8") as f:
            json.dump(sk_record, f, indent=2)

        # (b) C++ Scaled
        cpp_json, w_cpp = run_cpp_sgd(
            bin_path, impl="scaled", epochs=E, alpha=alpha,
            schedule="invscaling", eta0=eta0, power_t=power_t,
            fit_intercept=True, intercept_decay=1.0, warmup=1, repeats=5,
            dump_w=True
        )
        time_cpp_ms = float(cpp_json["timing"]["median_ns"]) / 1e6
        b_cpp = cpp_json["b"]

        pred_cpp_std = X_te.dot(w_cpp) + b_cpp
        pred_cpp_orig = pred_cpp_std * y_std + y_mean
        metrics_cpp = compute_metrics(y_te_orig, pred_cpp_orig)
        dist_ridge_cpp = distance_to_ridge(w_cpp, w_ridge)

        # Equivalence checks
        rel_rmse = abs(metrics_cpp["rmse"] - metrics_sk["rmse"]) / metrics_sk["rmse"]
        rel_mae = abs(metrics_cpp["mae"] - metrics_sk["mae"]) / metrics_sk["mae"]
        diff_r2 = abs(metrics_cpp["r2"] - metrics_sk["r2"])
        rel_w = relative_weight_difference(w_sk, w_cpp)
        rel_d = abs(dist_ridge_cpp - dist_ridge_sk) / dist_ridge_sk if dist_ridge_sk > 0 else 0.0

        equiv_pass = bool(rel_rmse < 1e-6 and rel_w < 1e-10)

        cpp_record = {
            "impl": "cpp_scaled",
            "dataset": dataset_name,
            "epochs": E,
            "m_train": m_train,
            "m_test": m_test,
            "n": n_features,
            "alpha": alpha,
            "metrics": metrics_cpp,
            "distance_to_ridge": dist_ridge_cpp,
            "train_time_ms": time_cpp_ms,
            "b": b_cpp,
            "equivalence": {
                "rel_rmse": rel_rmse,
                "rel_mae": rel_mae,
                "diff_r2": diff_r2,
                "rel_w": rel_w,
                "rel_dist_to_ridge": rel_d,
                "passed": equiv_pass,
            },
        }
        with open(out_dir / f"{dataset_name}_E{E}_cpp_scaled.json", "w", encoding="utf-8") as f:
            json.dump(cpp_record, f, indent=2)

        print(f"    E={E:3d} | RMSE sk={metrics_sk['rmse']:.4f}, cpp={metrics_cpp['rmse']:.4f} (rel diff: {rel_rmse:.2e}) | dist Ridge: {dist_ridge_cpp:.4f} | Pass: {equiv_pass}")

        results_table.append({
            "dataset": dataset_name,
            "epochs": E,
            "rmse_ridge": metrics_ridge["rmse"],
            "rmse_sk": metrics_sk["rmse"],
            "rmse_cpp": metrics_cpp["rmse"],
            "rel_rmse": rel_rmse,
            "rel_w": rel_w,
            "dist_ridge_sk": dist_ridge_sk,
            "dist_ridge_cpp": dist_ridge_cpp,
            "time_sk_ms": time_sk_ms,
            "time_cpp_ms": time_cpp_ms,
            "pass": equiv_pass,
        })

    return results_table


def run_all_rq2(config_path: Path, data_dir: Path, out_dir: Path):
    """Execute full RQ2 convergence sweeps across all 3 datasets."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    gen = cfg.get("general", {})
    alpha = gen.get("alpha", 1e-4)
    eta0 = gen.get("eta0", 0.01)
    power_t = gen.get("power_t", 0.25)
    epochs_sweep = gen.get("epochs_sweep", [1, 5, 20, 100, 500])
    dataset_names = cfg.get("datasets", {}).get("names", ["taiwan", "california", "kingcounty"])

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for name in dataset_names:
        table = run_rq2_dataset(
            name, data_dir, out_dir, epochs_sweep,
            alpha=alpha, eta0=eta0, power_t=power_t
        )
        all_results.extend(table)

    print(f"\nCompleted RQ2 on {len(dataset_names)} datasets across {len(epochs_sweep)} epoch levels.")


def main():
    parser = argparse.ArgumentParser(description="Run RQ2 convergence benchmarks.")
    parser.add_argument("--config", type=str, default="configs/rq2.toml", help="Path to config TOML")
    parser.add_argument("--quick", action="store_true", help="Quick mode (uses configs/rq2_quick.toml)")
    parser.add_argument("--data-dir", type=str, default="data/processed/rq2", help="Input dataset directory")
    parser.add_argument("--out-dir", type=str, default="results/raw/rq2", help="Output directory")

    args = parser.parse_args()
    config_file = Path("configs/rq2_quick.toml" if args.quick else args.config)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    run_all_rq2(config_file, data_dir, out_dir)


if __name__ == "__main__":
    main()
