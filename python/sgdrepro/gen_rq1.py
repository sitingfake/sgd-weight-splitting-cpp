"""
Synthetic Sparse Data Generator for RQ1.

Generates reproducible sparse linear regression benchmarks across Grids A, B, and C:
  - Features: k distinct non-zero entries per row, sorted ascending, N(0, 1), normalized to ||x||_2 = 1.
  - Ground truth: w* ~ N(0, 1) (dense, n-dimensional).
  - Target: y = X * w* + epsilon, where epsilon ~ N(0, 0.1^2).
  - Output: SGDCSR01 binary files and companion metadata JSONs.
"""

import argparse
import datetime
import json
from pathlib import Path
import tomllib
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

from sgdrepro.binio import write_csr_binary


def generate_sparse_dataset(
    n: int,
    k: int,
    m: int,
    seed: int = 42,
    chunk_size: int = 100_000,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Generate synthetic sparse regression dataset using deterministic NumPy Generator.

    Parameters
    ----------
    n : int
        Total feature dimensionality.
    k : int
        Number of non-zero entries per sample.
    m : int
        Number of samples (rows).
    seed : int
        Base seed component.
    chunk_size : int
        Row chunk size to prevent excessive memory consumption.

    Returns
    -------
    X : sp.csr_matrix
        Sparse feature matrix in CSR format with ||row||_2 = 1.
    y : np.ndarray
        1D target vector.
    w_star : np.ndarray
        Ground-truth dense weight vector.
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed feature count n ({n})")

    # Seed is uniquely conditioned on configuration tuple [42, n, k, m]
    rng = np.random.default_rng([seed, n, k, m])

    # Dense true parameter vector w* ~ N(0, 1)
    w_star = rng.standard_normal(n, dtype=np.float64)

    all_indices = []
    all_data = []

    remaining = m
    while remaining > 0:
        curr_chunk = min(remaining, chunk_size)

        # 1. Select k unique column indices per row
        if k == n:
            # Full density: every column is present
            chunk_indices = np.tile(np.arange(n, dtype=np.int32), (curr_chunk, 1))
        elif (k * k) / (2.0 * n) < 0.4 and k <= 50:
            # Rejection sampling for high sparsity (small k, low collision probability)
            chunk_indices = np.empty((curr_chunk, k), dtype=np.int32)
            row_idx = 0
            while row_idx < curr_chunk:
                needed = curr_chunk - row_idx
                candidates = rng.integers(0, n, size=(needed, k), dtype=np.int32)
                candidates.sort(axis=1)
                diffs = np.diff(candidates, axis=1)
                valid = (diffs > 0).all(axis=1)
                valid_candidates = candidates[valid]
                n_valid = len(valid_candidates)
                if n_valid > 0:
                    chunk_indices[row_idx : row_idx + n_valid] = valid_candidates
                    row_idx += n_valid
        else:
            # Argpartition sampling for moderate-to-high density (prevents exponential collision rates)
            # Memory bounded by limiting sub-chunks if n is large
            sub_chunk_size = min(curr_chunk, max(1000, 10_000_000 // max(1, n)))
            sub_indices = []
            for sub_start in range(0, curr_chunk, sub_chunk_size):
                sub_len = min(curr_chunk - sub_start, sub_chunk_size)
                random_keys = rng.random((sub_len, n))
                sub_idx = np.argpartition(random_keys, k, axis=1)[:, :k].astype(np.int32)
                sub_idx.sort(axis=1)
                sub_indices.append(sub_idx)
            chunk_indices = np.vstack(sub_indices)

        # 2. Draw standard normal values and normalize row Euclidean norm to 1
        raw_values = rng.standard_normal((curr_chunk, k), dtype=np.float64)
        row_norms = np.linalg.norm(raw_values, axis=1, keepdims=True)
        normalized_values = raw_values / row_norms

        all_indices.append(chunk_indices.ravel())
        all_data.append(normalized_values.ravel())
        remaining -= curr_chunk

    indices = np.concatenate(all_indices)
    data = np.concatenate(all_data)
    indptr = np.arange(0, (m + 1) * k, k, dtype=np.int32)

    X = sp.csr_matrix((data, indices, indptr), shape=(m, n))

    # 3. Generate linear regression target: y = X * w* + epsilon
    noise = rng.normal(loc=0.0, scale=0.1, size=m)
    y = X.dot(w_star) + noise

    return X, y, w_star


def collect_unique_configurations(config: dict) -> List[Tuple[int, int, int]]:
    """Extract distinct (n, k, m) tuples across Grids A, B, and C."""
    configs = set()

    # Grid A
    k_a = config["grid_a"]["k"]
    m_a = config["grid_a"]["m"]
    for n in config["grid_a"]["n"]:
        configs.add((int(n), int(k_a), int(m_a)))

    # Grid B
    n_b = config["grid_b"]["n"]
    m_b = config["grid_b"]["m"]
    for k in config["grid_b"]["k"]:
        configs.add((int(n_b), int(k), int(m_b)))

    # Grid C
    n_c = config["grid_c"]["n"]
    k_c = config["grid_c"]["k"]
    for m in config["grid_c"]["m"]:
        configs.add((int(n_c), int(k_c), int(m)))

    return sorted(list(configs))


def generate_all_rq1(config_path: Path, out_dir: Path) -> dict:
    """Generate all specified datasets, write to disk, and return file checksums."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    configurations = collect_unique_configurations(cfg)
    seed = cfg.get("general", {}).get("seed", 42)

    checksums = {}
    print(f"Generating {len(configurations)} RQ1 datasets into {out_dir}...")

    for n, k, m in configurations:
        base_name = f"n{n}_k{k}_m{m}"
        bin_path = out_dir / f"{base_name}.bin"
        meta_path = out_dir / f"{base_name}.meta.json"

        print(f"  -> Generating {base_name} (density: {k/n*100:.3f}%)...")
        X, y, _ = generate_sparse_dataset(n, k, m, seed=seed)

        sha256 = write_csr_binary(bin_path, X, y)
        checksums[f"rq1/{base_name}.bin"] = sha256

        meta = {
            "dataset": f"rq1/{base_name}",
            "n": n,
            "k": k,
            "m": m,
            "nnz": int(X.nnz),
            "seed": seed,
            "sha256": sha256,
            "numpy_version": np.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return checksums


def update_checksums_file(new_checksums: dict, checksum_path: Path = Path("configs/data_checksums.json")):
    """Update configs/data_checksums.json with newly computed SHA256 values."""
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if checksum_path.exists():
        try:
            with open(checksum_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.update(new_checksums)
    with open(checksum_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Updated checksums in {checksum_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sparse datasets for RQ1.")
    parser.add_argument("--config", type=str, default="configs/rq1.toml", help="Path to TOML config")
    parser.add_argument("--quick", action="store_true", help="Use configs/rq1_quick.toml")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    config_file = Path("configs/rq1_quick.toml" if args.quick else args.config)
    out_dir = Path(args.out_dir if args.out_dir else "data/processed/rq1")

    checksums = generate_all_rq1(config_file, out_dir)
    if not args.quick:
        update_checksums_file(checksums)


if __name__ == "__main__":
    main()
