/**
 * @file json_writer.hpp
 * @brief Minimal, dependency-free JSON serializer for benchmark outputs.
 *
 * Formats C++ SGD experiment results according to Section 5.4 of plan.md.
 */

#ifndef SGD_JSON_WRITER_HPP
#define SGD_JSON_WRITER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>

struct BenchmarkJsonRecord {
    std::string impl;
    std::string dataset;
    int64_t m = 0;
    int64_t n = 0;
    int64_t nnz = 0;

    // Parameters
    double alpha = 0.0;
    std::string schedule;
    double eta0 = 0.0;
    double power_t = 0.0;
    double optimal_init = 0.0;
    int epochs = 0;
    bool fit_intercept = false;
    double intercept_decay = 1.0;
    int64_t max_samples = 0;

    // Timing
    int warmup = 0;
    int repeats = 0;
    std::vector<int64_t> loop_ns;
    int64_t median_ns = 0;
    int64_t iqr_ns = 0;
    int64_t min_ns = 0;
    int64_t max_ns = 0;
    double us_per_update = 0.0;
    int64_t finalize_ns = 0;

    // Diagnostics
    int64_t n_resets = 0;
    bool diverged = false;
    int diverged_epoch = 0;
    double b = 0.0;
    std::string w_file;
    std::string w_sha256;

    // Build metadata
    std::string compiler;
    std::string flags;

    static std::string escape_json(const std::string& str) {
        std::string res;
        for (char c : str) {
            if (c == '\\') {
                res += "/";
            } else if (c == '"') {
                res += "\\\"";
            } else {
                res += c;
            }
        }
        return res;
    }

    /**
     * @brief Serialize benchmark record to a formatted JSON string.
     */
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::setprecision(16);

        ss << "{\n";
        ss << "  \"impl\": \"" << escape_json(impl) << "\",\n";
        ss << "  \"dataset\": \"" << escape_json(dataset) << "\",\n";
        ss << "  \"m\": " << m << ",\n";
        ss << "  \"n\": " << n << ",\n";
        ss << "  \"nnz\": " << nnz << ",\n";

        // Parameters
        ss << "  \"params\": {\n";
        ss << "    \"alpha\": " << alpha << ",\n";
        ss << "    \"schedule\": \"" << schedule << "\",\n";
        ss << "    \"eta0\": " << eta0 << ",\n";
        ss << "    \"power_t\": " << power_t << ",\n";
        ss << "    \"optimal_init\": " << optimal_init << ",\n";
        ss << "    \"epochs\": " << epochs << ",\n";
        ss << "    \"fit_intercept\": " << (fit_intercept ? "true" : "false") << ",\n";
        ss << "    \"intercept_decay\": " << intercept_decay << ",\n";
        ss << "    \"max_samples\": " << max_samples << "\n";
        ss << "  },\n";

        // Timing
        ss << "  \"timing\": {\n";
        ss << "    \"warmup\": " << warmup << ",\n";
        ss << "    \"repeats\": " << repeats << ",\n";
        ss << "    \"loop_ns\": [";
        for (size_t i = 0; i < loop_ns.size(); ++i) {
            ss << loop_ns[i] << (i + 1 < loop_ns.size() ? ", " : "");
        }
        ss << "],\n";
        ss << "    \"median_ns\": " << median_ns << ",\n";
        ss << "    \"iqr_ns\": " << iqr_ns << ",\n";
        ss << "    \"min_ns\": " << min_ns << ",\n";
        ss << "    \"max_ns\": " << max_ns << ",\n";
        ss << "    \"us_per_update\": " << us_per_update << ",\n";
        ss << "    \"finalize_ns\": " << finalize_ns << "\n";
        ss << "  },\n";

        // Diagnostics
        ss << "  \"n_resets\": " << n_resets << ",\n";
        ss << "  \"diverged\": " << (diverged ? "true" : "false") << ",\n";
        ss << "  \"diverged_epoch\": " << diverged_epoch << ",\n";
        ss << "  \"b\": " << b << ",\n";
        ss << "  \"w_file\": \"" << escape_json(w_file) << "\",\n";
        ss << "  \"w_sha256\": \"" << escape_json(w_sha256) << "\",\n";

        // Build
        ss << "  \"build\": {\n";
        ss << "    \"compiler\": \"" << escape_json(compiler) << "\",\n";
        ss << "    \"flags\": \"" << escape_json(flags) << "\"\n";
        ss << "  }\n";
        ss << "}\n";

        return ss.str();
    }

    /**
     * @brief Write JSON string directly to file.
     */
    void write_to_file(const std::string& path) const {
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open JSON output file: " + path);
        }
        out << to_json();
    }
};

#endif // SGD_JSON_WRITER_HPP
