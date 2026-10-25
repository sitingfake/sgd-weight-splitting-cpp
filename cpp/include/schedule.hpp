/**
 * @file schedule.hpp
 * @brief Learning rate schedule definitions for Stochastic Gradient Descent.
 *
 * Implements:
 *   - invscaling : eta_t = eta0 / (t ^ power_t)
 *   - bottou     : eta_t = eta0 / (1 + eta0 * alpha * (t - 1))
 *   - pegasos    : eta_t = 1 / (alpha * t)
 *   - optimal    : eta_t = 1 / (alpha * (optimal_init + t - 1))
 */

#ifndef SGD_SCHEDULE_HPP
#define SGD_SCHEDULE_HPP

#include <cmath>
#include <string>
#include <stdexcept>

enum class ScheduleType {
    INVSCALING,
    BOTTOU,
    PEGASOS,
    OPTIMAL
};

/**
 * @brief Parses schedule string to enum.
 */
inline ScheduleType parse_schedule(const std::string& name) {
    if (name == "invscaling") return ScheduleType::INVSCALING;
    if (name == "bottou") return ScheduleType::BOTTOU;
    if (name == "pegasos") return ScheduleType::PEGASOS;
    if (name == "optimal") return ScheduleType::OPTIMAL;
    throw std::invalid_argument("Unknown schedule name: " + name);
}

/**
 * @brief Computes learning rate eta at step t (1-indexed).
 *
 * Inline, non-virtual to ensure zero abstraction overhead.
 *
 * @param type Schedule type enum
 * @param t Global step index (t >= 1)
 * @param alpha L2 regularization parameter
 * @param eta0 Base learning rate
 * @param power_t Exponent for inverse scaling
 * @param optimal_init Offset for optimal schedule
 * @return Computed learning rate eta
 */
inline double get_learning_rate(ScheduleType type,
                               double t,
                               double alpha,
                               double eta0,
                               double power_t,
                               double optimal_init) {
    switch (type) {
        case ScheduleType::INVSCALING:
            return eta0 / std::pow(t, power_t);
        case ScheduleType::BOTTOU:
            return eta0 / (1.0 + eta0 * alpha * (t - 1.0));
        case ScheduleType::PEGASOS:
            return 1.0 / (alpha * t);
        case ScheduleType::OPTIMAL:
            return 1.0 / (alpha * (optimal_init + t - 1.0));
        default:
            return eta0;
    }
}

#endif // SGD_SCHEDULE_HPP
