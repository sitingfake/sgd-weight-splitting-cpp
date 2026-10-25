"""
RQ1 Benchmark Runner: Per-Update Cost Across Grids A, B, and C.

Executes sequential benchmark runs across:
  1. C++ Direct (full vector shrink O(n))
  2. C++ Scaled (lazy scalar shrink O(1), w = s * v)
  3. scikit-learn SGDRegressor (Cython WeightVector)

Computes final regularized objective losses, relative weight discrepancies,
and speedup ratios (t_direct / t_scaled).
"""

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any, Dict, List, Tuple
import numpy as np

from sgdrepro.binio import read_csr_binary
from sgdrepro.bench_sklearn import benchmark_sklearn_sgd
from sgdrepro.gen_rq1 import collect_unique_configurations
from sgdrepro.metrics import relative_weight_difference, sgd_objective_loss
from sgdrepro.run_cpp import run_cpp_sgd


def run_rq1_single_dataset(
    bin_path: Path,
    out_dir: Path,
    alpha: float = 1e-4,
    schedule: str = "invscaling",
    eta0: float = 0.01,
    power_t: float = 0.25,
    epochs: int = 3,
    warmup: int = 1,
    repeats: int = 7,
) -> Dict[str, Any]:
    """Execute direct, scaled, and sklearn benchmarks on one dataset."""
    base_name = bin_path.stem
    print(f"\n=======================================================")
    print(f"Running RQ1 Benchmark on: {base_name}")
    print(f"=======================================================")

    # 1. Load CSR matrix and target into memory
    X, y = read_csr_binary(bin_path)
    m, n = X.shape

    # 2. C++ Direct
    print("  -> Running C++ Direct...")
    json_dir, w_dir = run_cpp_sgd(
        bin_path, impl="direct", epochs=epochs, alpha=alpha,
        schedule=schedule, eta0=eta0, power_t=power_t,
        fit_intercept=False, warmup=warmup, repeats=repeats, dump_w=True
    )
    loss_dir = sgd_objective_loss(X, y, w_dir, b=json_dir["b"], alpha=alpha)
    json_dir["final_loss"] = loss_dir

    # 3. C++ Scaled
    print("  -> Running C++ Scaled...")
    json_sc, w_sc = run_cpp_sgd(
        bin_path, impl="scaled", epochs=epochs, alpha=alpha,
        schedule=schedule, eta0=eta0, power_t=power_t,
        fit_intercept=False, warmup=warmup, repeats=repeats, dump_w=True
    )
    loss_sc = sgd_objective_loss(X, y, w_sc, b=json_sc["b"], alpha=alpha)
    json_sc["final_loss"] = loss_sc

    # 4. scikit-learn SGDRegressor
    print("  -> Running scikit-learn SGDRegressor...")
    json_sk, w_sk, b_sk = benchmark_sklearn_sgd(
        X, y, dataset_name=str(bin_path), epochs=epochs, alpha=alpha,
        fit_intercept=False, schedule=schedule, eta0=eta0, power_t=power_t,
        warmup=warmup, repeats=repeats
    )
    loss_sk = sgd_objective_loss(X, y, w_sk, b=b_sk, alpha=alpha)
    json_sk["final_loss"] = loss_sk

    # Cross-method comparisons
    rel_w_sv_dir = relative_weight_difference(w_sc, w_dir)
    rel_w_sv_sk = relative_weight_difference(w_sc, w_sk)

    time_dir = json_dir["timing"]["median_ns"]
    time_sc = json_sc["timing"]["median_ns"]
    speedup = float(time_dir / time_sc) if time_sc > 0 else 1.0

    print(f"  Summary {base_name}:")
    print(f"    Direct:  {json_dir['timing']['us_per_update']:.4f} us/update | Loss: {loss_dir:.6f}")
    print(f"    Scaled:  {json_sc['timing']['us_per_update']:.4f} us/update | Loss: {loss_sc:.6f} | Resets: {json_sc['n_resets']}")
    print(f"    Sklearn: {json_sk['timing']['us_per_update']:.4f} us/update | Loss: {loss_sk:.6f}")
    print(f"    Speedup (Direct / Scaled): {speedup:.2f}x")
    print(f"    ||w_sv - w_dir|| / ||w||: {rel_w_sv_dir:.2e}")
    print(f"    ||w_sv - w_sk|| / ||w||:  {rel_w_sv_sk:.2e}")

    assert rel_w_sv_dir < 1e-10, f"Direct vs Scaled diff {rel_w_sv_dir} exceeded 1e-10 on {base_name}"

    # Attach comparisons to records
    for rec in (json_dir, json_sc, json_sk):
        rec["comparisons"] = {
            "rel_w_sv_dir": rel_w_sv_dir,
            "rel_w_sv_sk": rel_w_sv_sk,
            "speedup_direct_over_scaled": speedup,
        }

    # Save raw JSONs
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{base_name}_cpp_direct.json", "w", encoding="utf-8") as f:
        json.dump(json_dir, f, indent=2)
    with open(out_dir / f"{base_name}_cpp_scaled.json", "w", encoding="utf-8") as f:
        json.dump(json_sc, f, indent=2)
    with open(out_dir / f"{base_name}_sklearn_sgd.json", "w", encoding="utf-8") as f:
        json.dump(json_sk, f, indent=2)

    return {
        "dataset": base_name,
        "m": m,
        "n": n,
        "nnz": X.nnz,
        "speedup": speedup,
        "us_direct": json_dir["timing"]["us_per_update"],
        "us_scaled": json_sc["timing"]["us_per_update"],
        "us_sklearn": json_sk["timing"]["us_per_update"],
        "rel_w_sv_dir": rel_w_sv_dir,
        "rel_w_sv_sk": rel_w_sv_sk,
    }


def run_all_rq1(config_path: Path, data_dir: Path, out_dir: Path, pilot: bool = False):
    """Execute RQ1 across all unique configurations."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    gen = cfg.get("general", {})
    alpha = gen.get("alpha", 1e-4)
    schedule = gen.get("schedule", "invscaling")
    eta0 = gen.get("eta0", 0.01)
    power_t = gen.get("power_t", 0.25)
    epochs = gen.get("epochs", 3)
    warmup = gen.get("warmup", 1)
    repeats = gen.get("repeats", 7)

    configs = collect_unique_configurations(cfg)

    # In pilot mode, select 3 representative small configs to measure noise
    if pilot:
        print("=== PILOT CALIBRATION MODE ===")
        configs = [
            (1000, 20, 10000),
            (5000, 50, 5000),
            (10000, 20, 1000),
        ]
        out_dir = Path("results/pilot/raw/rq1")

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for n, k, m in configs:
        base_name = f"n{n}_k{k}_m{m}"
        bin_path = data_dir / f"{base_name}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing required dataset: {bin_path}; run `make data` first.")

        summary = run_rq1_single_dataset(
            bin_path, out_dir, alpha=alpha, schedule=schedule,
            eta0=eta0, power_t=power_t, epochs=epochs,
            warmup=warmup, repeats=repeats
        )
        summaries.append(summary)

    print(f"\nCompleted RQ1 on {len(summaries)} datasets. Raw results saved to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Run RQ1 benchmark.")
    parser.add_argument("--config", type=str, default="configs/rq1.toml", help="Path to config TOML")
    parser.add_argument("--quick", action="store_true", help="Run with quick configuration")
    parser.add_argument("--pilot", action="store_true", help="Run pilot timing noise calibration")
    parser.add_argument("--data-dir", type=str, default="data/processed/rq1", help="Dataset directory")
    parser.add_argument("--out-dir", type=str, default="results/raw/rq1", help="Output directory")

    args = parser.parse_args()
    config_file = Path("configs/rq1_quick.toml" if args.quick else args.config)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    run_all_rq1(config_file, data_dir, out_dir, pilot=args.pilot)


if __name__ == "__main__":
    main()
