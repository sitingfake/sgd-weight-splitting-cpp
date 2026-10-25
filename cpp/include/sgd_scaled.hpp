/**
 * @file sgd_scaled.hpp
 * @brief Scaled vector update implementation of Stochastic Gradient Descent (w = s * v).
 *
 * Implements the lazy regularization technique from Bottou (2012) and Shalev-Shwartz et al. (2011).
 * Represents w as scalar s and vector v:
 *   - Prediction: p = s * <v, x_i> + b
 *   - L2 shrinkage: s *= max(0, 1 - eta * alpha)       [O(1)]
 *   - Reset check: if s < 1e-9: v *= s; s = 1.0       [O(n), amortized negligible]
 *   - Gradient update: v += (u / s) * x_i             [O(nnz)]
 *
 * Weight reconstruction w = s * v is performed outside the timed region.
 */

#ifndef SGD_SCALED_HPP
#define SGD_SCALED_HPP

#include "csr.hpp"
#include "schedule.hpp"
#include "timer.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

struct SgdScaledResult {
    std::vector<double> w;
    double b = 0.0;
    int64_t n_resets = 0;
    bool diverged = false;
    int diverged_epoch = 0;
    int64_t loop_ns = 0;
    int64_t finalize_ns = 0;
};

/**
 * @brief Train linear model using lazy scaled vector updates (w = s * v).
 */
inline SgdScaledResult train_scaled(
    const CsrMatrix& mat,
    int epochs,
    double alpha,
    ScheduleType schedule,
    double eta0,
    double power_t,
    double optimal_init,
    bool fit_intercept,
    double intercept_decay
) {
    SgdScaledResult res;
    res.b = 0.0;
    res.n_resets = 0;

    const int64_t m = mat.m;
    const int64_t n = mat.n;
    const int32_t* indptr = mat.indptr.data();
    const int32_t* indices = mat.indices.data();
    const double* data = mat.data.data();
    const double* y = mat.y.data();

    std::vector<double> v(n, 0.0);
    double* v_ptr = v.data();
    double s = 1.0;

    double t = 1.0;
    const double MAX_DLOSS = 1e12;
    const double RESET_THRESHOLD = 1e-9;

    Timer loop_timer;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int64_t i = 0; i < m; ++i) {
            // 1. Predict using weights before shrinkage: p = s * <v, x_i> + b
            double dot = 0.0;
            const int32_t start = indptr[i];
            const int32_t end = indptr[i + 1];
            for (int32_t j = start; j < end; ++j) {
                dot += v_ptr[indices[j]] * data[j];
            }
            double p = s * dot + res.b;

            // Check for divergence
            if (!std::isfinite(p)) {
                res.diverged = true;
                res.diverged_epoch = epoch + 1;
                res.loop_ns = loop_timer.elapsed_ns();
                return res;
            }

            // 2. Compute loss derivative dloss and clip to [-MAX_DLOSS, MAX_DLOSS]
            double dloss = p - y[i];
            if (dloss < -MAX_DLOSS) dloss = -MAX_DLOSS;
            else if (dloss > MAX_DLOSS) dloss = MAX_DLOSS;

            // 3. Learning rate and L2 shrinkage factor applied to scalar s: O(1)
            double eta = get_learning_rate(schedule, t, alpha, eta0, power_t, optimal_init);
            double shrink = std::max(0.0, 1.0 - eta * alpha);
            s *= shrink;

            // Numerical stability reset when s gets too small (or reaches zero)
            if (s < RESET_THRESHOLD) {
                for (int64_t j = 0; j < n; ++j) {
                    v_ptr[j] *= s;
                }
                s = 1.0;
                res.n_resets++;
            }

            // 4. Gradient update: v += (u / s) * x_i: O(nnz)
            double u = -eta * dloss;
            if (u != 0.0) {
                double scale_u = u / s;
                for (int32_t j = start; j < end; ++j) {
                    v_ptr[indices[j]] += scale_u * data[j];
                }
                if (fit_intercept) {
                    res.b += u * intercept_decay;
                }
            }

            // Advance global step counter
            t += 1.0;
        }
    }

    res.loop_ns = loop_timer.elapsed_ns();

    // Finalize weights w = s * v outside timed loop
    Timer finalize_timer;
    res.w.resize(n);
    double* w_ptr = res.w.data();
    for (int64_t j = 0; j < n; ++j) {
        w_ptr[j] = s * v_ptr[j];
    }
    res.finalize_ns = finalize_timer.elapsed_ns();

    return res;
}

#endif // SGD_SCALED_HPP
