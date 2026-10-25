# Benchmark Findings and Hypothesis Adjudication Summary

> **Provenance**: Automatically generated from `results/raw/*.json` by `summary_gen.py`.

## 1. Summary of Hypothesis Verifications

| ID | Pre-registered Hypothesis | Strict Acceptance Criterion | Observed Empirical Metric | Status |
|---|---|---|---|---|
| **H1-a** | Scaled update ($w = s \cdot v$) is significantly faster on sparse data | $t_\text{dir}/t_\text{sc} \ge 2$ at $n \ge 10^4$ and monotonic | $n=10^4$: 58.27x; $n=10^5$: 564.24x; $n=10^6$: 3446.04x | **SUPPORTED** |
| **H1-b** | Direct update scales $\sim$ linearly with $n$, while Scaled scales $\le O(n^{0.35})$ | Direct log-log slope $\in [0.7, 1.1]$; Scaled slope $\le 0.35$ | Direct slope: 1.009; Scaled slope: 0.123 | **SUPPORTED** |
| **H1-c** | Dense data (100%) yields identical asymptotic order of cost | Density 100%: $0.5 \le t_\text{dir}/t_\text{sc} \le 3.0$ | Observed ratio: 1.18x | **SUPPORTED** |
| **H1-d** | Scaled per-update cost is invariant to sample count $m$ | Grid C: $\max(t_\text{sc}) / \min(t_\text{sc}) \le 1.3$ | Observed ratio: 1.08 | **SUPPORTED** |
| **H1-e** | C++ Direct and C++ Scaled converge to identical solutions | $\lVert w_\text{sv} - w_\text{dir} \rVert / \lVert w \rVert < 10^{-10}$ | Maximum discrepancy: 5.60e-15 | **SUPPORTED** |
| **H2-a** | C++ Scaled is mathematically and numerically equivalent to scikit-learn | $\lvert \Delta \text{RMSE} \rvert / \text{RMSE} < 10^{-6}$ and $\lVert \Delta w \rVert / \lVert w \rVert < 10^{-10}$ | Max relative RMSE error: 0.00e+00; Max weight error: 0.00e+00 | **SUPPORTED** |
| **H2-b** | C++ Scaled and scikit-learn match Ridge reference distance | $\lvert d_\text{cpp} - d_\text{sk} \rvert / d_\text{sk} < 10^{-6}$ across all epochs | Maximum distance discrepancy: 0.00e+00 | **SUPPORTED** |

---

## 2. Key Empirical Insights

### RQ1: Computational Complexity & Speedup
1. **Dimension Scaling (Grid A)**: As feature dimensionality grows from $n = 10^3$ to $n = 10^6$ at fixed non-zeros ($k=20$), the direct vector update cost increases from 0.26 $\mu$s to 223.89 $\mu$s per update (log-log slope = 1.009, strictly $O(n)$). In contrast, the lazy scaled formulation maintains near-constant update latency (0.0363 $\mu$s to 0.0650 $\mu$s), yielding an enormous **3446.0x speedup** at $n = 10^6$.
2. **Density Transition (Grid B)**: At extreme sparsity (0.1%), Scaled updates outperform Direct updates by 55.0x. As density increases to 100%, the ratio smoothly converges to 1.18x, confirming that both methods operate at $O(n)$ when data is fully dense.
3. **Sample Count Invariance (Grid C)**: Across sample sizes spanning four orders of magnitude ($m = 10^3$ to $10^6$), per-update cost remains strictly bounded within a factor of 1.08, demonstrating that per-sample computational complexity is independent of dataset scale.

### RQ2: Convergence & Solution Equivalence
1. **Machine Precision Equivalence**: C++ Scaled updates and scikit-learn `SGDRegressor` yield **0.00e+00 relative discrepancy** in weights, intercept, and held-out test RMSE across all three empirical datasets (Taiwan, California, King County) across all evaluated epoch counts ($E \in \{1, 5, 20, 100, 500\}$).
2. **Asymptotic Convergence to Closed-Form Ridge**: Both SGD implementations steadily approach the exact closed-form Ridge reference solution $\hat{w}_\text{ridge} = (X^T X + m \alpha I)^{-1} X^T y$. On the California Housing dataset, relative distance $\|w - w_\text{ridge}\| / \|w_\text{ridge}\|$ shrinks monotonically from 1.231 at $E=1$ down to 0.0155 at $E=500$.