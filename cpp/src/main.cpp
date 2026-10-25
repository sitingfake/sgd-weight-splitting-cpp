/**
 * @file main.cpp
 * @brief Command Line Interface for C++ Stochastic Gradient Descent benchmarks.
 *
 * Implements warmup runs, repeated timed executions, determinism verification,
 * binary weight export, and JSON result recording.
 */

#include "csr.hpp"
#include "schedule.hpp"
#include "timer.hpp"
#include "json_writer.hpp"
#include "sgd_direct.hpp"
#include "sgd_scaled.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "Options:\n"
              << "  --data PATH            Path to binary CSR dataset (.bin)\n"
              << "  --impl {direct|scaled} Implementation to benchmark\n"
              << "  --epochs E             Number of training epochs (default: 3)\n"
              << "  --alpha A              L2 regularization parameter (default: 1e-4)\n"
              << "  --schedule S           Learning rate schedule (invscaling|bottou|pegasos|optimal)\n"
              << "  --eta0 X               Base learning rate (default: 0.01)\n"
              << "  --power_t X            Exponent for invscaling schedule (default: 0.25)\n"
              << "  --optimal_init X       Offset parameter for optimal schedule (default: 0.0)\n"
              << "  --fit_intercept {0|1}  Whether to fit bias term (default: 0)\n"
              << "  --intercept_decay X    Decay rate for intercept update (default: 1.0)\n"
              << "  --max_samples N        Limit training to the first N samples (default: -1, all)\n"
              << "  --warmup W             Warmup repetitions (default: 1)\n"
              << "  --repeats R            Timed repetitions (default: 7)\n"
              << "  --dump_w PATH          Path to dump final float64 weight array (.w.bin)\n"
              << "  --out PATH             Output JSON metadata file path\n";
}

std::string get_compiler_info() {
#if defined(__clang__)
    return "clang++ " + std::string(__clang_version__);
#elif defined(__GNUC__)
    return "g++ " + std::string(__VERSION__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::string data_path;
    std::string impl = "scaled";
    int epochs = 3;
    double alpha = 1e-4;
    std::string schedule_str = "invscaling";
    double eta0 = 0.01;
    double power_t = 0.25;
    double optimal_init = 0.0;
    bool fit_intercept = false;
    double intercept_decay = 1.0;
    int64_t max_samples = -1;
    int warmup = 1;
    int repeats = 7;
    std::string dump_w_path;
    std::string out_json_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--impl" && i + 1 < argc) {
            impl = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = std::atof(argv[++i]);
        } else if (arg == "--schedule" && i + 1 < argc) {
            schedule_str = argv[++i];
        } else if (arg == "--eta0" && i + 1 < argc) {
            eta0 = std::atof(argv[++i]);
        } else if (arg == "--power_t" && i + 1 < argc) {
            power_t = std::atof(argv[++i]);
        } else if (arg == "--optimal_init" && i + 1 < argc) {
            optimal_init = std::atof(argv[++i]);
        } else if (arg == "--fit_intercept" && i + 1 < argc) {
            fit_intercept = (std::atoi(argv[++i]) != 0);
        } else if (arg == "--intercept_decay" && i + 1 < argc) {
            intercept_decay = std::atof(argv[++i]);
        } else if (arg == "--max_samples" && i + 1 < argc) {
            max_samples = std::atoll(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--dump_w" && i + 1 < argc) {
            dump_w_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_json_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or unhandled argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (data_path.empty()) {
        std::cerr << "Error: --data argument is required.\n";
        return 1;
    }

    ScheduleType schedule_type;
    try {
        schedule_type = parse_schedule(schedule_str);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing schedule: " << e.what() << "\n";
        return 1;
    }

    // Load dataset from binary CSR
    CsrMatrix mat;
    try {
        mat = CsrMatrix::load(data_path, max_samples);
    } catch (const std::exception& e) {
        std::cerr << "Data loading error: " << e.what() << "\n";
        return 1;
    }

    bool is_scaled = (impl == "scaled");
    if (impl != "scaled" && impl != "direct") {
        std::cerr << "Error: unknown impl '" << impl << "'; must be 'direct' or 'scaled'.\n";
        return 1;
    }

    // Warmup executions
    for (int w = 0; w < warmup; ++w) {
        if (is_scaled) {
            train_scaled(mat, epochs, alpha, schedule_type, eta0, power_t,
                         optimal_init, fit_intercept, intercept_decay);
        } else {
            train_direct(mat, epochs, alpha, schedule_type, eta0, power_t,
                         optimal_init, fit_intercept, intercept_decay);
        }
    }

    // Timed repetitions
    std::vector<int64_t> loop_times;
    std::vector<double> final_w;
    double final_b = 0.0;
    int64_t total_resets = 0;
    int64_t final_finalize_ns = 0;
    bool diverged = false;
    int diverged_epoch = 0;

    for (int r = 0; r < repeats; ++r) {
        if (is_scaled) {
            SgdScaledResult res = train_scaled(
                mat, epochs, alpha, schedule_type, eta0, power_t,
                optimal_init, fit_intercept, intercept_decay
            );
            if (res.diverged) {
                diverged = true;
                diverged_epoch = res.diverged_epoch;
                break;
            }
            loop_times.push_back(res.loop_ns);
            final_finalize_ns = res.finalize_ns;
            total_resets = res.n_resets;
            final_b = res.b;

            // Determinism check across repeats
            if (r == 0) {
                final_w = std::move(res.w);
            } else {
                for (size_t j = 0; j < final_w.size(); ++j) {
                    if (final_w[j] != res.w[j]) {
                        std::cerr << "Warning: non-deterministic weights observed at repeat " << r
                                  << " index " << j << "\n";
                        break;
                    }
                }
            }
        } else {
            SgdDirectResult res = train_direct(
                mat, epochs, alpha, schedule_type, eta0, power_t,
                optimal_init, fit_intercept, intercept_decay
            );
            if (res.diverged) {
                diverged = true;
                diverged_epoch = res.diverged_epoch;
                break;
            }
            loop_times.push_back(res.loop_ns);
            final_finalize_ns = res.finalize_ns;
            total_resets = res.n_resets;
            final_b = res.b;

            // Determinism check across repeats
            if (r == 0) {
                final_w = std::move(res.w);
            } else {
                for (size_t j = 0; j < final_w.size(); ++j) {
                    if (final_w[j] != res.w[j]) {
                        std::cerr << "Warning: non-deterministic weights observed at repeat " << r
                                  << " index " << j << "\n";
                        break;
                    }
                }
            }
        }
    }

    // Dump weights to binary if path provided and not diverged
    if (!dump_w_path.empty() && !diverged && !final_w.empty()) {
        std::ofstream w_out(dump_w_path, std::ios::binary);
        if (!w_out.is_open()) {
            std::cerr << "Error opening dump_w file: " << dump_w_path << "\n";
            return 1;
        }
        w_out.write(reinterpret_cast<const char*>(final_w.data()), sizeof(double) * final_w.size());
    }

    // Prepare JSON record
    BenchmarkJsonRecord rec;
    rec.impl = is_scaled ? "cpp_scaled" : "cpp_direct";
    rec.dataset = data_path;
    rec.m = mat.m;
    rec.n = mat.n;
    rec.nnz = mat.nnz;

    rec.alpha = alpha;
    rec.schedule = schedule_str;
    rec.eta0 = eta0;
    rec.power_t = power_t;
    rec.optimal_init = optimal_init;
    rec.epochs = epochs;
    rec.fit_intercept = fit_intercept;
    rec.intercept_decay = intercept_decay;
    rec.max_samples = (max_samples > 0) ? max_samples : mat.m;

    rec.warmup = warmup;
    rec.repeats = repeats;
    rec.loop_ns = loop_times;

    int64_t total_updates = static_cast<int64_t>(epochs) * mat.m;
    TimingStats stats = TimingStats::compute(loop_times, total_updates);
    rec.median_ns = stats.median_ns;
    rec.iqr_ns = stats.iqr_ns;
    rec.min_ns = stats.min_ns;
    rec.max_ns = stats.max_ns;
    rec.us_per_update = stats.us_per_update;
    rec.finalize_ns = final_finalize_ns;

    rec.n_resets = total_resets;
    rec.diverged = diverged;
    rec.diverged_epoch = diverged_epoch;
    rec.b = final_b;
    rec.w_file = dump_w_path;
    rec.w_sha256 = "";

    rec.compiler = get_compiler_info();
    rec.flags = "-std=c++17 -O2 -DNDEBUG -Wall -Wextra";

    if (!out_json_path.empty()) {
        rec.write_to_file(out_json_path);
    } else {
        std::cout << rec.to_json() << "\n";
    }

    if (diverged) {
        std::cerr << "Model diverged at epoch " << diverged_epoch << " (exit code 3).\n";
        return 3;
    }

    return 0;
}
