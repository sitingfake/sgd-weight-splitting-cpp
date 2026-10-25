# Study Scope, Methodological Boundaries, and Limitations

This document articulates the explicit boundaries, experimental constraints, and theoretical limitations of this reproduction study comparing pure C++17 stochastic gradient descent against scikit-learn's `SGDRegressor`.

---

## 1. Single-Machine Hardware and Platform Scope

1. **Hardware Specificity**:
   - All empirical benchmarks were conducted on a single host machine equipped with an **AMD Ryzen 9 8940HX (16 cores, 32 threads)** and 64 GB DDR5 RAM running **Windows 11**.
   - Absolute wall-clock timings ($\mu$s/update and ms/epoch) are host-specific and reflect the instruction pipelines, cache hierarchy (L1/L2/L3), and memory bandwidth of this particular processor.
   - Cross-platform reproduction is expected to preserve **relative speedups**, asymptotic scaling slopes, and numerical equivalence, but absolute times will differ across microarchitectures.

2. **OS Scheduling and Core Pinning**:
   - Windows 11 does not provide a standard POSIX `taskset` interface for hard CPU core affinity without administrative privileges or third-party drivers.
   - Background interference was minimized by setting high process priority, disabling background applications, running single-threaded (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), and running 7 repetitions with interquartile range (IQR) noise verification (all IQRs were validated to be $< 5\%$ of the median).

3. **Compiler and Optimization Flags**:
   - The C++ binary was compiled with **MinGW GCC 14.2.0** with `-O2 -DNDEBUG -std=c++17`.
   - Compiler auto-vectorization (SIMD) affects contiguous array traversals (such as dense shrinkage in direct SGD). Different compilers (e.g., Clang, MSVC) or target architectures (x86-64 vs. ARM64) may show slight variations in direct-update loop efficiency.

---

## 2. Problem Formulation: Regression vs. Original Classification Context

1. **Squared Error vs. Margin Losses**:
   - The foundational papers introducing scaled SGD updates—Bottou (2010, 2012) and Shalev-Shwartz et al. (2011, *Pegasos*)—primarily investigated large-margin linear classification (support vector machines with hinge loss or logistic loss).
   - In hinge loss classification, subgradients are non-zero only for margin-violating samples ($y_i (w \cdot x_i) < 1$).
   - This reproduction evaluates **regularized linear regression** ($L_2$-regularized squared loss), where the gradient error residual $\hat{y}_i - y_i$ is almost never zero for continuous targets. Consequently, weight updates occur on *every single training sample*, which maximizes the frequency of shrinkage operations.

2. **Asymptotic Convergence Target**:
   - For regularized linear regression, the objective function is strictly convex and admits a unique closed-form solution:
     $$\hat{w}_{\text{ridge}} = (X^T X + m \alpha I)^{-1} X^T y$$
   - In classification, empirical risk minimization often converges to a separating hyperplane where multiple weight vectors achieve zero training error. Here, we can and do evaluate exact convergence trajectories against the closed-form Ridge solution.

---

## 3. Data Representation and Dimensional Sparsity

1. **Synthetic Dimensionality vs. Real-world Sparsity**:
   - In RQ1, synthetic datasets used uniform random sparsity structures where each sample had exactly $k$ non-zero features drawn uniformly across $n$ dimensions.
   - Real-world sparse datasets (e.g., TF-IDF text matrices, bag-of-words, genomic one-hot encodings) frequently exhibit **Zipfian (power-law) feature distributions**, where a tiny fraction of features appear in nearly every sample while the vast majority are extremely rare.
   - While the $O(k)$ per-update complexity of scaled SGD holds regardless of non-zero distribution, cache locality and memory access patterns will differ on power-law distributed data.

2. **Dense Datasets in RQ2**:
   - The empirical datasets in RQ2 (Taiwan Real Estate, California Housing, King County) are strictly **dense** ($100\%$ density, $n \in \{6, 8, 17\}$).
   - Consequently, RQ2 was designed to validate **solution equivalence, convergence trajectories, and numerical fidelity** between C++ and Cython, rather than computational speedup. On dense data, direct and scaled updates perform comparable operations, as confirmed by hypothesis **H1-c**.

---

## 4. Algorithmic and Implementation Scope

1. **Single-threaded Sequential Execution**:
   - Both C++ and scikit-learn implementations are evaluated in single-threaded mode to study fundamental algorithm-level complexity.
   - Parallel asynchronous SGD architectures (e.g., *Hogwild!*, multi-threaded lock-free updates) introduce non-deterministic race conditions and cache coherence overheads that are orthogonal to the mathematical equivalence of weight scaling.

2. **Absence of Heuristic Early Stopping**:
   - To ensure deterministic sample sequences and exact epoch-by-epoch tracking, early stopping based on validation loss stagnation (`early_stopping=False`, `tol=None`) was explicitly disabled.
   - Production deployments often employ adaptive convergence checks, which can terminate training in fewer epochs.

---

## 5. Scope of Secondary Experiments

To maintain strict adherence to the primary research questions (RQ1 computational speedup and RQ2 empirical equivalence), all secondary exploratory experiments (S1 through S5) have been formally excluded from the core production pipeline:
- The core study focuses exclusively on the pre-registered `invscaling` schedule ($\eta_0 = 0.01, \text{power\_t} = 0.25$) on standardized features, which exhibits stable, monotonic convergence across all datasets.
- Exploratory schedule searches (Bottou $\gamma_0$, pure Pegasos $1/(\alpha t)$, unstandardized raw feature variants, and artificial scale reset triggers) are treated as out-of-scope peripheral extensions and not required for reproducing the primary findings.

