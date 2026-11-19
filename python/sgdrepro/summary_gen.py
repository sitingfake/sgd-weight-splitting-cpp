"""Summarize the observed benchmark threshold checks."""

import math
from pathlib import Path

from sgdrepro.make_tables import load_json


def generate_summary():
    raw_rq1 = Path("results/raw/rq1")
    raw_rq2 = Path("results/raw/rq2")

    def rq1(n, k, m, method):
        return load_json(raw_rq1 / f"n{n}_k{k}_m{m}_cpp_{method}.json")

    def loop_us(n, k, m, method):
        return rq1(n, k, m, method)["timing"]["us_per_update"]

    def speedup(n, k, m):
        return loop_us(n, k, m, "direct") / loop_us(n, k, m, "scaled")

    grid_a_speedup = [speedup(n, 20, 10000) for n in (10**4, 10**5, 10**6)]
    slopes = [
        math.log10(loop_us(10**6, 20, 10000, method) / loop_us(10**4, 20, 10000, method)) / 2
        for method in ("direct", "scaled")
    ]
    dense_speedup = speedup(5000, 5000, 5000)
    grid_c_times = [loop_us(10000, 20, m, "scaled") for m in (1000, 10000, 100000, 1000000)]
    sample_ratio = max(grid_c_times) / min(grid_c_times)
    max_rq1_w = max(
        load_json(path)["comparisons"]["rel_w_sv_dir"]
        for path in raw_rq1.glob("*_cpp_scaled.json")
    )

    checks = [
        ("RQ1 Direct/Scaled speedup", "At least 2× at n=10⁴, then increasing",
         ", ".join(f"{value:.2f}×" for value in grid_a_speedup),
         2 <= grid_a_speedup[0] < grid_a_speedup[1] < grid_a_speedup[2]),
        ("RQ1 dimension slopes", "Direct in [0.7, 1.1], Scaled ≤ 0.35",
         f"Direct {slopes[0]:.3f}, Scaled {slopes[1]:.3f}",
         0.7 <= slopes[0] <= 1.1 and slopes[1] <= 0.35),
        ("RQ1 fully dense speedup", "In [0.5, 3.0]", f"{dense_speedup:.2f}×",
         0.5 <= dense_speedup <= 3.0),
        ("RQ1 Scaled max/min time per update", "≤ 1.30", f"{sample_ratio:.2f}",
         sample_ratio <= 1.30),
        ("RQ1 Direct/Scaled relative weight error", "< 1e-10", f"{max_rq1_w:.2e}",
         max_rq1_w < 1e-10),
    ]
    for method in ("direct", "scaled"):
        comparisons = [
            load_json(path)["equivalence"] for path in raw_rq2.glob(f"*_cpp_{method}.json")
        ]
        max_rmse = max(eq["rel_rmse"] for eq in comparisons)
        max_w = max(eq["rel_w"] for eq in comparisons)
        checks.append((
            f"RQ2 {method.title()}/scikit-learn", "Relative RMSE < 1e-6 and weight error < 1e-10",
            f"RMSE {max_rmse:.2e}, weights {max_w:.2e}",
            bool(comparisons) and all(eq["passed"] for eq in comparisons),
        ))
        max_distance = max(eq["rel_dist_to_ridge"] for eq in comparisons)
        checks.append((
            f"RQ2 {method.title()}/scikit-learn relative Ridge distance error",
            "< 1e-6", f"{max_distance:.2e}", max_distance < 1e-6,
        ))

    lines = [
        "# Benchmark findings",
        "",
        "Computed from [raw results](raw/). The paired model tables contain median and "
        "IQR for the full fit and test prediction timings alongside the test errors.",
        "",
        "| Check | Threshold | Observed | Status |",
        "| --- | --- | --- | --- |",
    ]
    for check, threshold, observed, passed in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {check} | {threshold} | {observed} | {status} |")
    lines.extend([
        "",
        "- [RQ1 paired table](tables/rq1_paired.csv)",
        "- [Taiwan](tables/rq2_taiwan.csv), [California](tables/rq2_california.csv), "
        "and [King County](tables/rq2_kingcounty.csv)",
        "",
    ])
    out_file = Path("results/SUMMARY.md")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary generated at {out_file}")


if __name__ == "__main__":
    generate_summary()
