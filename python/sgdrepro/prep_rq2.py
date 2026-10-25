"""
Housing Datasets Preprocessor for RQ2.

Fetches and preprocesses three benchmark housing regression datasets:
  1. Real Estate Valuation (Taiwan) - UCI 477 (414 x 6)
  2. California Housing - scikit-learn (20,640 x 8)
  3. King County House Sales - OpenML 42092 (21,613 x 17)

Performs deterministic train/test splitting (80/20, seed 42), Z-score standardization
fitted exclusively on the training partition, and exports both NumPy archives and
SGDCSR01 binary files for C++ consumption.
"""

import argparse
import datetime
import gzip
import io
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from sgdrepro.binio import write_csr_binary
from sgdrepro.gen_rq1 import update_checksums_file


def load_taiwan_real_estate(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Real Estate Valuation (Taiwan) from UCI (ID 477).
    Expected shape: 414 samples, 6 features.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=477)
    df_features = dataset.data.features.copy()
    df_targets = dataset.data.targets.copy()

    # Drop index column 'No' if present
    if "No" in df_features.columns:
        df_features = df_features.drop(columns=["No"])

    expected_cols = [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]
    assert list(df_features.columns) == expected_cols, f"Mismatch in Taiwan columns: {df_features.columns}"
    assert len(df_features) == 414, f"Expected 414 rows, got {len(df_features)}"

    target_col = "Y house price of unit area"
    assert target_col in df_targets.columns, f"Target column {target_col} missing"

    X = df_features.to_numpy(dtype=np.float64)
    y = df_targets[target_col].to_numpy(dtype=np.float64).ravel()
    return X, y, expected_cols


def load_california_housing(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load California Housing dataset from scikit-learn.
    Expected shape: 20,640 samples, 8 features.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cal = fetch_california_housing(data_home=str(raw_dir), as_frame=True)
    X = cal.data.to_numpy(dtype=np.float64)
    y = cal.target.to_numpy(dtype=np.float64).ravel()
    feature_names = list(cal.feature_names)

    assert X.shape == (20640, 8), f"Expected (20640, 8), got {X.shape}"
    assert len(y) == 20640, f"Expected 20640 targets, got {len(y)}"
    return X, y, feature_names


def load_king_county(raw_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load King County House Sales dataset from OpenML (data_id=42092).
    Directly streams and parses the OpenML ARFF archive to avoid categorical parsing
    limitations in scikit-learn's internal ARFF parser.
    Expected shape: 21,613 samples, 17 features.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    arff_dir = raw_dir / "openml" / "openml.org" / "data" / "v1" / "download" / "21578898"
    arff_path = arff_dir / "house_sales.arff.gz"

    # Download from OpenML API if not already cached
    if not arff_path.exists():
        url = "https://api.openml.org/data/v1/download/21578898/house_sales.arff"
        print(f"Downloading King County dataset from OpenML: {url}...")
        arff_dir.mkdir(parents=True, exist_ok=True)
        response = urllib.request.urlopen(url)
        content = response.read()
        with gzip.open(arff_path, "wb") as gz:
            gz.write(content)

    # Parse ARFF header and data block
    with gzip.open(arff_path, "rt", encoding="utf-8") as f:
        attributes = []
        data_rows = []
        in_data = False
        for line in f:
            clean = line.strip()
            if not in_data:
                if clean.upper().startswith("@ATTRIBUTE"):
                    attr_name = clean.split()[1]
                    attributes.append(attr_name)
                elif clean.upper() == "@DATA":
                    in_data = True
            else:
                if clean and not clean.startswith("%"):
                    data_rows.append(clean)

    df = pd.read_csv(io.StringIO("\n".join(data_rows)), names=attributes)

    assert len(df) == 21613, f"Expected 21613 rows, got {len(df)}"

    expected_17_features = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "yr_renovated", "lat", "long",
        "sqft_living15", "sqft_lot15"
    ]

    for col in expected_17_features:
        assert col in df.columns, f"Expected column {col} missing from King County dataset"
    assert "price" in df.columns, "Target 'price' missing from King County dataset"

    # Select strictly the 17 numeric features and target price (dropping id, date, zipcode)
    X = df[expected_17_features].to_numpy(dtype=np.float64)
    y = df["price"].to_numpy(dtype=np.float64).ravel()

    assert X.shape == (21613, 17), f"Expected (21613, 17), got {X.shape}"
    return X, y, expected_17_features


def process_and_save_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> str:
    """
    Standardize, split, serialize to binary CSR and npz archives.
    """
    dest_dir = out_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Numerical invariants check
    assert np.isfinite(X).all(), f"Dataset {name} contains NaN or Inf features"
    assert np.isfinite(y).all(), f"Dataset {name} contains NaN or Inf targets"
    assert (np.std(X, axis=0) > 0).all(), f"Dataset {name} contains constant column (std=0)"

    n_samples, n_features = X.shape

    # 1. Deterministic train-test split
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # 2. Z-score standardization fitted on train partition
    x_mean = np.mean(X_train, axis=0)
    x_scale = np.std(X_train, axis=0, ddof=0)
    x_scale[x_scale == 0] = 1.0  # Safeguard against division by zero

    X_tr_std = (X_train - x_mean) / x_scale
    X_te_std = (X_test - x_mean) / x_scale

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train, ddof=0))
    if y_std == 0:
        y_std = 1.0

    y_tr_std = (y_train - y_mean) / y_std
    y_te_orig = y_test.copy()

    # 3. Save split indices
    split_path = dest_dir / "split.npz"
    np.savez_compressed(split_path, train_idx=train_idx, test_idx=test_idx)

    # 4. Save numpy arrays
    arrays_path = dest_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        X_tr=X_tr_std,
        y_tr_std=y_tr_std,
        X_te=X_te_std,
        y_te_orig=y_te_orig,
        y_mean=y_mean,
        y_std=y_std,
        x_mean=x_mean,
        x_scale=x_scale,
    )

    # 5. Convert standardized train to CSR matrix and save train.bin
    X_tr_csr = sp.csr_matrix(X_tr_std)
    bin_path = dest_dir / "train.bin"
    sha256 = write_csr_binary(bin_path, X_tr_csr, y_tr_std)

    # 6. Metadata JSON
    meta = {
        "dataset": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "feature_names": feature_names,
        "y_mean": y_mean,
        "y_std": y_std,
        "sha256_train_bin": sha256,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(dest_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  -> Saved {name}: {n_samples} total ({len(train_idx)} train, {len(test_idx)} test), {n_features} features.")
    return sha256


def prep_all_rq2(raw_dir: Path, out_dir: Path) -> Dict[str, str]:
    """Run preprocessing for all three housing datasets."""
    checksums = {}
    print(f"Preprocessing RQ2 datasets into {out_dir}...")

    # 1. Taiwan Real Estate
    X_tw, y_tw, feats_tw = load_taiwan_real_estate(raw_dir)
    sha_tw = process_and_save_dataset("taiwan", X_tw, y_tw, feats_tw, out_dir)
    checksums["rq2/taiwan/train.bin"] = sha_tw

    # 2. California Housing
    X_ca, y_ca, feats_ca = load_california_housing(raw_dir)
    sha_ca = process_and_save_dataset("california", X_ca, y_ca, feats_ca, out_dir)
    checksums["rq2/california/train.bin"] = sha_ca

    # 3. King County
    X_kc, y_kc, feats_kc = load_king_county(raw_dir)
    sha_kc = process_and_save_dataset("kingcounty", X_kc, y_kc, feats_kc, out_dir)
    checksums["rq2/kingcounty/train.bin"] = sha_kc

    return checksums


def main():
    parser = argparse.ArgumentParser(description="Preprocess housing datasets for RQ2.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw download directory")
    parser.add_argument("--out-dir", type=str, default="data/processed/rq2", help="Processed output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode flag (uses full dataset since small)")

    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    checksums = prep_all_rq2(raw_dir, out_dir)
    update_checksums_file(checksums)


if __name__ == "__main__":
    main()
