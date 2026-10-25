/**
 * @file sgd_direct.hpp
 * @brief Direct vector update implementation of Stochastic Gradient Descent.
 *
 * Updates all n coordinates of weight vector w at every iteration to apply
 * L2 regularization shrinkage:
 *   w *= max(0, 1 - eta * alpha)  [O(n)]
 * Followed by sparse gradient accumulation:
 *   w += u * x_i                  [O(nnz)]
 */

#ifndef SGD_DIRECT_HPP
#define SGD_DIRECT_HPP

#include "csr.hpp"
#include "schedule.hpp"
#include "timer.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

struct SgdDirectResult {
    std::vector<double> w;
    double b = 0.0;
    int64_t n_resets = 0;  // Always 0 for direct
    bool diverged = false;
    int diverged_epoch = 0;
    int64_t loop_ns = 0;
    int64_t finalize_ns = 0;
};

/**
 * @brief Train linear model using direct dense vector shrinkage.
 */
inline SgdDirectResult train_direct(
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
    SgdDirectResult res;
    res.w.assign(mat.n, 0.0);
    res.b = 0.0;

    const int64_t m = mat.m;
    const int64_t n = mat.n;
    const int32_t* indptr = mat.indptr.data();
    const int32_t* indices = mat.indices.data();
    const double* data = mat.data.data();
    const double* y = mat.y.data();
    double* w = res.w.data();

    double t = 1.0;
    const double MAX_DLOSS = 1e12;

    Timer loop_timer;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int64_t i = 0; i < m; ++i) {
            // 1. Predict using weights before shrinkage: p = <w, x_i> + b
            double p = res.b;
            const int32_t start = indptr[i];
            const int32_t end = indptr[i + 1];
            for (int32_t j = start; j < end; ++j) {
                p += w[indices[j]] * data[j];
            }

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

            // 3. Learning rate and L2 shrinkage factor
            double eta = get_learning_rate(schedule, t, alpha, eta0, power_t, optimal_init);
            double shrink = std::max(0.0, 1.0 - eta * alpha);

            // Dense shrinkage pass over all n coordinates: O(n)
            for (int64_t j = 0; j < n; ++j) {
                w[j] *= shrink;
            }

            // 4. Gradient update: u = -eta * dloss
            double u = -eta * dloss;
            if (u != 0.0) {
                for (int32_t j = start; j < end; ++j) {
                    w[indices[j]] += u * data[j];
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
    res.finalize_ns = 0;  // Direct weights require no post-processing
    return res;
}

#endif // SGD_DIRECT_HPP
