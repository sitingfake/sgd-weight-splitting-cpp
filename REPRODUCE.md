# Complete Reproduction Guide: SGD in C++ vs. scikit-learn

This document provides a step-by-step operational guide for reproducing all benchmarks, equivalence checks, tables, and figures from a clean checkout.

---

## 0. System Requirements

- **Operating System**: Windows 10/11 (with MinGW-w64 or WSL2) or Linux (Ubuntu 20.04+ / Debian 11+).
- **C++ Compiler**: GCC $\ge$ 11.0 (tested with GCC 14.2.0) or Clang $\ge$ 14.0 supporting C++17 (`-std=c++17`).
- **Python**: Python $\ge$ 3.11 (tested with Python 3.12.9).
- **RAM**: Minimum 16 GB (Grid C dataset `n=10,000, m=1,000,000` requires ~3.2 GB RAM during matrix instantiation).
- **Disk Space**: $\sim$ 2.5 GB for raw and processed binary datasets (`data/`).
- **Estimated Runtime**:
  - `make smoke`: $\sim$ 2–3 minutes.
  - `make test`: $\sim$ 30 seconds.
  - Full pipeline (`make all`): $\sim$ 10–15 minutes.

---

## 1. Environment Setup

Create and activate a virtual environment, install pinned dependencies, and verify system capabilities:

```bash
# Clone the repository
git clone <repo-url> sgd-repro
cd sgd-repro

# Create and activate virtual environment
python -m venv .venv

# On Windows (cmd / powershell):
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Inspect and log system capabilities
make setup
```

The setup command generates `results/env.json`, which documents CPU model, logical cores, available memory, compiler version, and pinned package versions.

---

## 2. Compiling the C++ Binary

Compile the standalone C++17 SGD executable:

```bash
make build
```

This invokes:
```bash
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Icpp/include -o build/sgd_cpp.exe cpp/src/main.cpp
```
Verify that compilation completes warning-free and produces `build/sgd_cpp.exe` (or `build/sgd_cpp` on POSIX systems).

---

## 3. Data Generation and Preprocessing

Generate synthetic binary CSR datasets (RQ1) and download/standardize empirical housing datasets (RQ2):

```bash
make data
```

What this does:
1. `python -m sgdrepro.gen_rq1`: Generates 12 synthetic CSR files in `data/processed/rq1/` using the binary `SGDCSR01` specification.
2. `python -m sgdrepro.prep_rq2`: Downloads and parses Real Estate Valuation (Taiwan), California Housing, and King County House Sales, partitions into 80/20 train/test splits, computes Z-score standardizations on training sets, and serializes `train.bin`, `arrays.npz`, and `meta.json` into `data/processed/rq2/`.

---

## 4. Running the Equivalence Test Suite

Before running benchmarks, the numerical equivalence test suite **must pass**:

```bash
make test
```

This executes all 38 pytest unit and step-ladder tests in `tests/`:
- Step ladder tests ($N \in \{1, 2, 3, 10, 100, 1000\}$): Verifies that C++ Direct, C++ Scaled, and scikit-learn produce identical weights at every intermediate iteration.
- Shrinkage, clipping, schedule, and CSR reader component tests.
- Ridge regularization equivalence test ($\alpha_{\text{ridge}} = m_{\text{train}} \cdot \alpha_{\text{sgd}}$).

---

## 5. Pilot Noise Calibration (Optional)

To check for background CPU interference and verify timing stability:

```bash
make pilot
```

This runs minimal configurations and measures interquartile range (IQR) noise. Target variance: $\text{IQR} / \text{median} < 5\%$. If noise exceeds 5%, close background browser tabs and background applications.

---

## 6. Running Benchmarks

### 6.1 RQ1: Synthetic CSR Scaling
```bash
make rq1
```
Runs 12 synthetic datasets across C++ Direct, C++ Scaled, and scikit-learn (7 repetitions each). Raw JSON files are saved to `results/raw/rq1/`.

### 6.2 RQ2: Empirical Housing Convergence
```bash
make rq2
```
Trains all 3 empirical datasets across $E \in \{1, 5, 20, 100, 500\}$ against closed-form Ridge analytical baselines. Raw JSON files are saved to `results/raw/rq2/`.

---

## 7. Generating Reports, Tables, and Figures

To compile all raw JSON outputs into publication tables, plots, and summary markdown:

```bash
make report
```

Output artifacts generated:
- **Tables** (`results/tables/`):
  - `rq1_gridA.{csv,md,tex}`: Dimension scaling table ($n \in [10^3, 10^6]$).
  - `rq1_gridB.{csv,md,tex}`: Density scaling table ($0.1\%$ to $100\%$).
  - `rq1_gridC.{csv,md,tex}`: Sample count invariance table ($m \in [10^3, 10^6]$).
  - `rq2_{taiwan,california,kingcounty}.{csv,md,tex}`: Evaluation metrics vs. Ridge.
  - `equivalence_summary.{csv,md,tex}`: Step-by-step equivalence audit.
- **Figures** (`results/figures/`):
  - `fig1_rq1_gridA_dim.{png,pdf}`: Dimension scaling plot.
  - `fig2_rq1_gridB_density.{png,pdf}`: Density scaling transition.
  - `fig3_rq1_gridC_samples.{png,pdf}`: Sample count invariance plot.
  - `figS1_rq2_convergence.{png,pdf}`: Convergence trajectory toward Ridge.
  - `figS2_rq2_rmse.{png,pdf}`: Test RMSE convergence vs. Ridge baseline.
  - `figS3_rq2_mae.{png,pdf}`: Test MAE convergence vs. Ridge baseline.
  - `figS4_rq2_r2.{png,pdf}`: Test R^2 score convergence vs. Ridge baseline.
- **Summary**:
  - `results/SUMMARY.md`: Formal pre-registered hypothesis verification table.

---

## 8. Verifying Against Reference Baseline

To verify that local results match pre-computed reference baselines:

```bash
make verify
```

The script `python/sgdrepro/compare_reference.py` checks:
1. Every RQ1 configuration preserves relative speedup within a factor of 2.0 ($0.5 \le \text{speedup}_{\text{local}} / \text{speedup}_{\text{ref}} \le 2.0$).
2. Every RQ2 configuration matches scikit-learn test RMSE down to $< 10^{-6}$ relative error.

---

## 9. Determinism and Cross-Machine Expectations

1. **Synthetic Data Checksums**:
   - Synthetic random generation uses NumPy's `np.random.default_rng(seed=42)`. While NumPy's generator algorithm is stable across minor releases, slight discrepancies in float representation can alter checksums on different platforms.
   - If `configs/data_checksums.json` reports a checksum mismatch on a different OS, run `make test`—if the numerical equivalence ladder tests pass, the mathematical logic is fully preserved.
2. **Timing Expectations**:
   - Absolute wall-clock times ($\mu$s/update) **will differ** across CPUs depending on clock speeds, cache sizes, and memory buses.
   - Cross-platform reproduction is validated by:
     - Log-log slope of Direct update $\approx 1.0$ ($O(n)$).
     - Log-log slope of Scaled update $\le 0.35$.
     - Relative speedup $t_{\text{direct}} / t_{\text{scaled}} \gg 10^3$ at $n = 10^6$.
     - Equivalence discrepancy $\le 10^{-10}$ across all configurations.

---

## 10. Troubleshooting Guide

| Symptom | Probable Cause | Recommended Resolution |
|---|---|---|
| `g++: command not found` | C++ compiler is not installed or not on system `PATH` | On Windows, install MinGW-w64 via MSYS2 or WinLibs and add `bin/` to `PATH`. On Linux, run `sudo apt install build-essential`. |
| OpenML King County download fails | Network connectivity issue or OpenML API rate limiting | Re-run `make data`. `prep_rq2.py` will retry streaming the raw ARFF archive. |
| Out of Memory (OOM) during Grid C generation | Machine has insufficient RAM for $m = 10^6, n = 10,000$ | Run `python -m sgdrepro.gen_rq1 --quick` to generate smaller sample grids. |
| Timing IQR / median $> 5\%$ | Background CPU processes or thermal throttling | Close intensive applications (web browsers, IDE indexing). Run `make pilot` to confirm quiet state. |
| Equivalence test failure (`test_ladder.py`) | Floating-point contraction or architecture divergence | Ensure C++ binary is compiled with `-O2 -DNDEBUG` without `-ffast-math`. Check `docs/DECISIONS.md` Decision 17. |

---

## 11. Platform Notes (Windows vs. Linux)

- **Windows**: Windows 11 does not support POSIX `taskset` for core affinity. Concurrency flags (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) are exported automatically by `Makefile` and `make.cmd` to enforce strict single-threaded operation.
- **Linux**: To pin core affinity on Linux systems, prefix benchmark commands with `taskset -c 2` (e.g., `taskset -c 2 make rq1`).
