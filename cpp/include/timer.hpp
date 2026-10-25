/**
 * @file timer.hpp
 * @brief High-precision timer and summary statistics for benchmarking.
 *
 * Uses std::chrono::steady_clock to accurately measure elapsed execution
 * time in nanoseconds, with utilities for computing median and IQR.
 */

#ifndef SGD_TIMER_HPP
#define SGD_TIMER_HPP

#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdint>

class Timer {
public:
    Timer() : start_time(std::chrono::steady_clock::now()) {}

    void reset() {
        start_time = std::chrono::steady_clock::now();
    }

    /**
     * @return Elapsed time in nanoseconds since last reset.
     */
    int64_t elapsed_ns() const {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_time;
};

struct TimingStats {
    int64_t median_ns = 0;
    int64_t iqr_ns = 0;
    int64_t min_ns = 0;
    int64_t max_ns = 0;
    double us_per_update = 0.0;

    /**
     * @brief Compute median and IQR from a collection of repeat measurements.
     * @param samples Vector of measured nanoseconds per repeat.
     * @param total_updates Total updates performed per repeat (epochs * m).
     */
    static TimingStats compute(std::vector<int64_t> samples, int64_t total_updates) {
        TimingStats stats;
        if (samples.empty()) return stats;

        std::sort(samples.begin(), samples.end());
        size_t n = samples.size();
        stats.min_ns = samples.front();
        stats.max_ns = samples.back();

        // Compute median
        if (n % 2 == 1) {
            stats.median_ns = samples[n / 2];
        } else {
            stats.median_ns = (samples[n / 2 - 1] + samples[n / 2]) / 2;
        }

        // Compute Q1 and Q3 using standard median of halves
        if (n >= 4) {
            size_t half = n / 2;
            int64_t q1 = (half % 2 == 1) ? samples[half / 2]
                                         : (samples[half / 2 - 1] + samples[half / 2]) / 2;
            size_t q3_idx = (n % 2 == 1) ? half + 1 : half;
            int64_t q3 = (half % 2 == 1) ? samples[q3_idx + half / 2]
                                         : (samples[q3_idx + half / 2 - 1] + samples[q3_idx + half / 2]) / 2;
            stats.iqr_ns = q3 - q1;
        } else {
            stats.iqr_ns = stats.max_ns - stats.min_ns;
        }

        if (total_updates > 0) {
            stats.us_per_update = (static_cast<double>(stats.median_ns) / total_updates) / 1000.0;
        }

        return stats;
    }
};

#endif // SGD_TIMER_HPP
