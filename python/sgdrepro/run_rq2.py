"""Measure fit, held-out prediction and convergence on the three housing datasets."""

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from sgdrepro.bench_sklearn import benchmark_sklearn_sgd
from sgdrepro.binio import read_csr_binary, sha256_file
from sgdrepro.evaluation import benchmark_fit, evaluate_holdout, holdout_sha256
from sgdrepro.metrics import relative_weight_difference
from sgdrepro.ridge import distance_to_ridge, make_ridge_reference
from sgdrepro.run_cpp import run_cpp_sgd


def relative_difference(reference, value):
    return abs(reference - value) / abs(reference) if reference else abs(value)


def run_rq2_dataset(
    dataset_name: str,
    data_dir: Path,
    out_dir: Path,
    epochs_sweep: list[int],
    alpha: float = 1e-4,
    eta0: float = 0.01,
    power_t: float = 0.25,
    warmup: int = 1,
    repeats: int = 7,
) -> list[dict]:
    """Evaluate the same train/test split for Direct, Scaled, SGDRegressor and Ridge."""
    print(f"\nRunning RQ2 on {dataset_name}", flush=True)
    ds_dir = data_dir / dataset_name
    bin_path = ds_dir / "train.bin"
    with np.load(ds_dir / "arrays.npz") as arr:
        X_tr = arr["X_tr"]
        y_tr = arr["y_tr_std"]
        X_te = arr["X_te"]
        y_te = arr["y_te_orig"]
        y_mean = float(arr["y_mean"])
        y_std = float(arr["y_std"])

    cpp_X_tr, cpp_y_tr = read_csr_binary(bin_path)
    if not (
        np.array_equal(cpp_X_tr.toarray(), X_tr) and np.array_equal(cpp_y_tr, y_tr)
    ):
        raise ValueError(f"Mismatched C++ and Python training data for {dataset_name}")

    m_train, n_features = X_tr.shape
    common = {
        "dataset": dataset_name,
        "m_train": m_train,
        "m_test": len(y_te),
        "n": n_features,
        "train_sha256": sha256_file(bin_path),
        "test_sha256": holdout_sha256(X_te, y_te, y_std, y_mean),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(suffix, record):
        path = out_dir / f"{dataset_name}_{suffix}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    def evaluate(w, b, fit_timing):
        return {
            **common,
            **evaluate_holdout(X_te, y_te, w, b, warmup, repeats, y_std, y_mean),
            "fit_timing": fit_timing,
            "train_time_ms": fit_timing["median_ns"] / 1e6,
            "distance_to_ridge": distance_to_ridge(w, w_ridge),
            "b": b,
        }

    ridge, ridge_fit = benchmark_fit(
        lambda: make_ridge_reference(m_train, alpha), X_tr, y_tr, warmup, repeats
    )
    w_ridge = ridge.coef_
    ridge_record = {
        **evaluate(w_ridge, float(ridge.intercept_), ridge_fit),
        "impl": "ridge_reference",
        "alpha_sgd": alpha,
        "alpha_ridge": m_train * alpha,
    }
    save("ridge", ridge_record)

    results_table = []
    for epochs in epochs_sweep:
        params = {
            "epochs": epochs, "alpha": alpha, "fit_intercept": True,
            "eta0": eta0, "power_t": power_t, "schedule": "invscaling",
        }
        sk, w_sk, b_sk = benchmark_sklearn_sgd(
            X_tr, y_tr, dataset_name, **params, warmup=warmup, repeats=repeats
        )
        sk_record = {
            **evaluate(w_sk, b_sk, sk["fit_timing"]),
            **params,
            "impl": "sklearn_sgd",
        }
        sk_metrics = sk_record["metrics"]
        save(f"E{epochs}_sklearn", sk_record)

        comparisons = {}
        for impl in ("direct", "scaled"):
            cpp, w_cpp = run_cpp_sgd(
                bin_path, impl=impl, **params, intercept_decay=1.0,
                warmup=warmup, repeats=repeats, dump_w=True,
            )
            if w_cpp is None or "fit_timing" not in cpp:
                raise RuntimeError(f"C++ {impl} failed to produce a timed model for {dataset_name}, E={epochs}.")
            record = {
                **evaluate(w_cpp, cpp["b"], cpp["fit_timing"]),
                **params,
                "impl": cpp["impl"],
                "timing": cpp["timing"],
                "build": cpp["build"],
                "n_resets": cpp["n_resets"],
            }
            cpp_metrics = record["metrics"]
            rel_rmse = relative_difference(sk_metrics["rmse"], cpp_metrics["rmse"])
            rel_w = relative_weight_difference(w_sk, w_cpp)
            record["equivalence"] = {
                "rel_rmse": rel_rmse,
                "rel_mae": relative_difference(sk_metrics["mae"], cpp_metrics["mae"]),
                "diff_r2": abs(sk_metrics["r2"] - cpp_metrics["r2"]),
                "rel_w": rel_w,
                "rel_dist_to_ridge": relative_difference(
                    sk_record["distance_to_ridge"], record["distance_to_ridge"]
                ),
                "passed": bool(rel_rmse < 1e-6 and rel_w < 1e-10),
            }
            save(f"E{epochs}_{record['impl']}", record)
            comparisons[impl] = record

        scaled = comparisons["scaled"]
        eq = scaled["equivalence"]
        print(
            f"  E={epochs:3d} | RMSE={scaled['metrics']['rmse']:.6f} | "
            f"fit ms: Direct={comparisons['direct']['train_time_ms']:.3f}, "
            f"Scaled={scaled['train_time_ms']:.3f}, sklearn={sk_record['train_time_ms']:.3f} | "
            f"parity: Direct={comparisons['direct']['equivalence']['passed']}, Scaled={eq['passed']}",
            flush=True,
        )
        results_table.append({
            "dataset": dataset_name,
            "epochs": epochs,
            "rmse_ridge": ridge_record["metrics"]["rmse"],
            "rmse_sk": sk_metrics["rmse"],
            "rmse_cpp": scaled["metrics"]["rmse"],
            "rel_rmse": eq["rel_rmse"],
            "rel_w": eq["rel_w"],
            "dist_ridge_sk": sk_record["distance_to_ridge"],
            "dist_ridge_cpp": scaled["distance_to_ridge"],
            "time_sk_ms": sk_record["train_time_ms"],
            "time_cpp_ms": scaled["train_time_ms"],
            "pass": eq["passed"],
        })
    return results_table


def run_all_rq2(config_path: Path, data_dir: Path, out_dir: Path):
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    gen = cfg.get("general", {})
    epochs_sweep = gen.get("epochs_sweep", [1, 5, 20, 100, 500])
    dataset_names = cfg.get("datasets", {}).get("names", ["taiwan", "california", "kingcounty"])
    for name in dataset_names:
        run_rq2_dataset(
            name, data_dir, out_dir, epochs_sweep,
            alpha=gen.get("alpha", 1e-4), eta0=gen.get("eta0", 0.01),
            power_t=gen.get("power_t", 0.25), warmup=gen.get("warmup", 1),
            repeats=gen.get("repeats", 7),
        )
    print(f"\nCompleted RQ2 on {len(dataset_names)} datasets across {len(epochs_sweep)} epoch levels.")


def main():
    parser = argparse.ArgumentParser(description="Run RQ2 housing benchmarks.")
    parser.add_argument("--config", default="configs/rq2.toml")
    parser.add_argument("--quick", action="store_true", help="Use configs/rq2_quick.toml")
    parser.add_argument("--data-dir", default="data/processed/rq2")
    parser.add_argument("--out-dir", default="results/raw/rq2")
    args = parser.parse_args()
    config = Path("configs/rq2_quick.toml" if args.quick else args.config)
    run_all_rq2(config, Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
