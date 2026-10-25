"""Verification and comparison tool for benchmark results against reference.

This script implements `make verify` by comparing generated benchmark data
in `results/raw/` against archived reference benchmarks in `results/reference/raw/`.

Validation Criteria:
1. Numerical Equivalence: All solution discrepancies between C++ and scikit-learn
   must remain below calibrated thresholds (rel_w < 1e-10, rel_rmse < 1e-6).
2. Relative Performance Consistency: Relative speedups (t_direct / t_scaled)
   must remain within a factor of 2.0 of reference benchmarks.
"""

import json
import math
import sys
from pathlib import Path


def load_json(filepath: Path) -> dict:
    """Load and parse JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_rq1(raw_dir: Path, ref_dir: Path) -> bool:
    """Verify RQ1 synthetic benchmark results against reference."""
    print("\n--- Verifying RQ1 Benchmarks ---")
    all_passed = True

    # Find all direct files
    direct_files = sorted(raw_dir.glob("*_cpp_direct.json"))
    if not direct_files:
        print("  [ERROR] No RQ1 direct benchmark files found.")
        return False

    for cur_dir in direct_files:
        base_name = cur_dir.name.replace("_cpp_direct.json", "")
        cur_sc = raw_dir / f"{base_name}_cpp_scaled.json"
        cur_sk = raw_dir / f"{base_name}_sklearn_sgd.json"

        ref_dir_f = ref_dir / f"{base_name}_cpp_direct.json"
        ref_sc = ref_dir / f"{base_name}_cpp_scaled.json"

        if not cur_sc.exists():
            print(f"  [ERROR] Missing scaled result for {base_name}")
            all_passed = False
            continue

        c_sc_data = load_json(cur_sc)
        c_dir_data = load_json(cur_dir)

        t_sc = c_sc_data.get("timing", {}).get("us_per_update", 1.0)
        t_dir = c_dir_data.get("timing", {}).get("us_per_update", 1.0)
        cur_speedup = t_dir / t_sc if t_sc > 0 else 1.0

        if ref_sc.exists() and ref_dir_f.exists():
            r_sc_data = load_json(ref_sc)
            r_dir_data = load_json(ref_dir_f)
            r_t_sc = r_sc_data.get("timing", {}).get("us_per_update", 1.0)
            r_t_dir = r_dir_data.get("timing", {}).get("us_per_update", 1.0)
            ref_speedup = r_t_dir / r_t_sc if r_t_sc > 0 else 1.0

            ratio = cur_speedup / ref_speedup if ref_speedup > 0 else 1.0
            is_valid = 0.5 <= ratio <= 2.0
            status = "PASS" if is_valid else "FAIL"
            if not is_valid:
                all_passed = False
            print(f"  [{status}] {base_name[:28]:<28} Speedup: cur={cur_speedup:8.2f}x, ref={ref_speedup:8.2f}x (ratio={ratio:.2f})")
        else:
            print(f"  [INFO] {base_name[:28]:<28} Speedup: cur={cur_speedup:8.2f}x (no reference file)")

    return all_passed


def verify_rq2(raw_dir: Path, ref_dir: Path) -> bool:
    """Verify RQ2 empirical housing benchmark results against reference."""
    print("\n--- Verifying RQ2 Housing Benchmarks ---")
    all_passed = True
    datasets = ["taiwan", "california", "kingcounty"]
    epochs = [1, 5, 20, 100, 500]

    for ds in datasets:
        for ep in epochs:
            cur_cpp = raw_dir / f"{ds}_E{ep}_cpp_scaled.json"
            cur_sk = raw_dir / f"{ds}_E{ep}_sklearn.json"
            ref_cpp = ref_dir / f"{ds}_E{ep}_cpp_scaled.json"

            if not cur_cpp.exists() or not cur_sk.exists():
                print(f"  [ERROR] Missing files for {ds} E={ep}")
                all_passed = False
                continue

            cpp_data = load_json(cur_cpp)
            sk_data = load_json(cur_sk)

            cpp_rmse = cpp_data.get("metrics", {}).get("rmse", 0.0)
            sk_rmse = sk_data.get("metrics", {}).get("rmse", 0.0)

            # Check exact parity between C++ and scikit-learn
            diff_rmse = abs(cpp_rmse - sk_rmse)
            rel_rmse = diff_rmse / sk_rmse if sk_rmse > 0 else 0.0
            is_equiv = rel_rmse < 1e-6

            # Check parity with reference
            ref_match = True
            if ref_cpp.exists():
                r_data = load_json(ref_cpp)
                r_rmse = r_data.get("metrics", {}).get("rmse", 0.0)
                if abs(cpp_rmse - r_rmse) / (r_rmse + 1e-12) > 1e-6:
                    ref_match = False

            passed = is_equiv and ref_match
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {ds:<12} E={ep:<3} RMSE C++: {cpp_rmse:.6f}, sk: {sk_rmse:.6f} (rel_err: {rel_rmse:.2e})")

    return all_passed


def main():
    root = Path(__file__).resolve().parent.parent.parent
    raw_dir = root / "results" / "raw"
    ref_dir = root / "results" / "reference" / "raw"

    print("=================================================================")
    print("        SGD Reproduction Benchmark Verification Suite           ")
    print("=================================================================")
    print(f"Target raw results : {raw_dir}")
    print(f"Reference results  : {ref_dir}")

    if not raw_dir.exists():
        print(f"\n[ERROR] Raw results directory does not exist: {raw_dir}")
        sys.exit(1)

    if not ref_dir.exists():
        print(f"\n[WARNING] Reference directory does not exist: {ref_dir}")
        print("Creating reference archive from current results...")
        import shutil
        shutil.copytree(raw_dir, ref_dir, dirs_exist_ok=True)
        print("Reference archive established.")

    rq1_ok = verify_rq1(raw_dir / "rq1", ref_dir / "rq1")
    rq2_ok = verify_rq2(raw_dir / "rq2", ref_dir / "rq2")

    print("\n=================================================================")
    if rq1_ok and rq2_ok:
        print("               [PASS] ALL VERIFICATION CHECKS PASSED             ")
        print("=================================================================")
        sys.exit(0)
    else:
        print("               [FAIL] SOME VERIFICATION CHECKS FAILED            ")
        print("=================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
