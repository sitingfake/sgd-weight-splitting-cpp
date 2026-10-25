# scikit-learn Cython Source Audit

This document presents the systematic source code audit of scikit-learn version **1.9.1** (specifically the underlying Cython `.pyx` and `.pyx.tp` implementations in `_sgd_fast`, `_weight_vector`, `_seq_dataset`, `_stochastic_gradient.py`, and `_loss`).
Its purpose is to formally verify that our C++ SGD implementation matches the exact mathematical, algorithmic, and numerical behaviors of scikit-learn's `SGDRegressor`.

---

## 1. Audit Matrix (10 Required Verification Checkpoints)

| # | Checkpoint Event | Scikit-Learn Value / Behavior | Source File & Line | C++ Conformance Status |
|---|---|---|---|---|
| **1** | `WeightVector` scaled representation (`w = s * v`) | Yes; maintains `wscale` (`s`) and raw vector `w_data_ptr` (`v`). Dot product: `innerprod *= wscale`. Gradient update: `w += val * (c / wscale)`. | `sklearn/utils/_weight_vector.pyx.tp:50, 69, 110, 182` | **Exact match**: `sgd_scaled.hpp` represents weights as scalar `s` and vector `v`, scaling dot products by `s` and updates by `u / s`. |
| **2** | Reset threshold for `wscale` & reset behavior | Threshold is `1e-9` for float64. When `wscale < 1e-9`, calls `reset_wscale()` which executes BLAS `_scal` (`v *= s`) and resets `wscale = 1.0`. | `sklearn/utils/_weight_vector.pyx.tp:16, 194, 197-207` | **Exact match**: In `sgd_scaled.hpp`, if `s < 1e-9`, all elements `v[i] *= s`, `s = 1.0`, and reset counter increments. Also safely handles `s == 0`. |
| **3** | Operation sequence & condition `update != 0` | Sequence: (1) prediction `p = w.dot(x) + intercept` using weights **prior** to shrinkage; (2) clip `dloss`; (3) L2 shrinkage `w.scale(max(0, 1 - (1 - l1_ratio) * eta * alpha))`; (4) add gradient `w.add(...)` only if `update != 0.0`. | `sklearn/linear_model/_sgd_fast.pyx.tp:507, 523-530, 545, 548` | **Exact match**: Both `sgd_direct.hpp` and `sgd_scaled.hpp` compute dot product using current weights, clip gradient, apply L2 shrink, and only add feature vector if update magnitude is non-zero. |
| **4** | Gradient clipping threshold (`MAX_DLOSS`) | `MAX_DLOSS = 1e12`. `dloss` is clipped to `[-1e12, 1e12]`. | `sklearn/linear_model/_sgd_fast.pyx.tp:439, 526-529` | **Exact match**: Gradient `g = std::max(-1e12, std::min(1e12, p - y_i))` enforced across both implementations. |
| **5** | `intercept_decay` for dense vs sparse | Dense inputs receive `intercept_decay = 1.0`. Sparse CSR inputs receive `intercept_decay = 0.01`. In update: `intercept += intercept_update * intercept_decay`. | `sklearn/linear_model/_base.py:make_dataset`; `_sgd_fast.pyx.tp:554` | **Exact match**: CLI exposes `--intercept_decay`, configured to `1.0` for dense RQ2 and `0.01` for sparse data with intercept. |
| **6** | Schedule `invscaling` & time step `t` initialization | Formula: `eta = eta0 / pow(t, power_t)`. Global counter starts at `t = 1.0`, increments after every sample update (`t += 1`). Persists across epochs via `self.t_`. | `sklearn/linear_model/_sgd_fast.pyx.tp:300, 488, 570`; `_stochastic_gradient.py:656, 1718` | **Exact match**: `t` starts at 1, continuous monotonic increment across epochs, `eta(t) = eta0 / std::pow(t, power_t)`. |
| **7** | Schedule `optimal` & `_init_t` (`typw`, `optimal_init`) | `typw = sqrt(1.0 / sqrt(alpha))`. `initial_eta0 = typw / max(1.0, loss.cy_gradient(1.0, -typw))`. For squared error (`alpha = 1e-4`), `grad = -11.0`, giving `initial_eta0 = 10.0` and `optimal_init = 1000.0`. Formula: `eta = 1.0 / (alpha * (optimal_init + t - 1))`. | `sklearn/linear_model/_sgd_fast.pyx.tp:459-463, 486`; `_loss/_loss.pyx.tp:317-321` | **Exact match**: Implemented with precomputed `optimal_init = 1.0 / (initial_eta0 * alpha)` matching sklearn step-for-step. |
| **8** | Traversal ordering under `shuffle=False` | `if shuffle: dataset.shuffle(seed)` is skipped. Samples are visited in deterministic row order `i = 0 .. m - 1`. | `sklearn/linear_model/_sgd_fast.pyx.tp:473-474` | **Exact match**: Iterates sequential rows `0 .. m - 1` per epoch. |
| **9** | Intercept regularization behavior | Intercept `b` is never shrunk by L2 regularization (`w.scale(...)` affects only weights). Intercept update is purely additive: `b += u * intercept_decay`. | `sklearn/linear_model/_sgd_fast.pyx.tp:545, 550-554` | **Exact match**: Neither direct nor scaled implementation regularizes the bias parameter `b`. |
| **10** | Loss derivative `dloss` for `squared_error` | `cgradient_half_squared_error` returns `raw_prediction - y_true`, which equals `p - y`. | `sklearn/_loss/_loss.pyx.tp:317-321` | **Exact match**: `dloss = p - y`. |

---

## 2. In-Depth Technical Verification

### 2.1 Probing `optimal` Learning Rate at alpha = 1e-4
In scikit-learn:
$$\text{typw} = \left(\frac{1}{\sqrt{\alpha}}\right)^{1/2} = \left(\frac{1}{\sqrt{10^{-4}}}\right)^{1/2} = \sqrt{100} = 10.0$$
For squared error loss $L(p, y) = \frac{1}{2}(p - y)^2$:
$$\text{loss.cy\_gradient}(1.0, -\text{typw}) = \text{raw\_prediction} - \text{y\_true} = -10.0 - 1.0 = -11.0$$
Because $-11.0 < 1.0$, the denominator takes $\max(1.0, -11.0) = 1.0$.
Consequently:
$$\eta_{0,\text{init}} = \frac{10.0}{1.0} = 10.0$$
$$\text{optimal\_init} = \frac{1}{10.0 \times 10^{-4}} = 1000.0$$
At step $t = 1$:
$$\eta_1 = \frac{1}{10^{-4} \times (1000 + 1 - 1)} = \frac{1}{0.1} = 10.0$$
**Empirical Confirmation**: An initial learning rate of $\eta_1 = 10.0$ is excessively aggressive for unstandardized or even standardized regression, explaining why the `optimal` schedule frequently triggers explosive divergence ($p \to \infty$) unless features are scaled to very small norms. This confirms Section 5.2 and Section 6 (S1) of `plan.md`.

### 2.2 Numerical Invariant between Direct and Scaled Updates
In exact real arithmetic:
- **Direct**:
  $$w_{t} = (1 - \eta_t \alpha) w_{t-1} + u_t x_i$$
- **Scaled**:
  $$s_t = (1 - \eta_t \alpha) s_{t-1}, \quad v_t = v_{t-1} + \frac{u_t}{s_t} x_i$$
  $$w_t = s_t v_t = s_t \left( v_{t-1} + \frac{u_t}{s_t} x_i \right) = \frac{s_t}{s_{t-1}} (s_{t-1} v_{t-1}) + u_t x_i = (1 - \eta_t \alpha) w_{t-1} + u_t x_i$$
The two formulations are algebraically identical. Under IEEE 754 float64 arithmetic, difference arises only due to:
1. Floating-point division by $s_t$ followed by multiplication by $s$ at finalization.
2. Reset events when $s < 10^{-9}$ (where accumulated rounding errors are bounded).
Expected relative discrepancy between direct and scaled updates across machine double precision is $\|w_{\text{sv}} - w_{\text{dir}}\| / \|w\| < 10^{-10}$.

---

## 3. Verified Implementations vs. Original Reading Report
Key verified technical alignments with ground truth include:
1. **Ridge reference penalty**: Confirmed exact objective alignment $\alpha_{\text{Ridge}} = m_{\text{train}} \alpha_{\text{SGD}}$.
2. **Scikit-learn version**: Confirmed installed scikit-learn version is **1.9.1**.
3. **`optimal` initial learning rate**: Verified $\eta_0 = 10.0$ at $\alpha = 10^{-4}$.
4. **Tolerance thresholds**: Calibrated to relative error scaling to account for unit differences across datasets.
