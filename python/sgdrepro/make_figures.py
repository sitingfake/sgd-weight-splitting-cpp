"""
Scientific Figures Generator.

Generates publication-quality figures for:
  - Fig 1: RQ1 Grid A - Per-update cost vs. Dimension n (log-log, O(n) dashed reference)
  - Fig 2: RQ1 Grid B - Per-update cost vs. Density % (log-log)
  - Fig 3: RQ1 Grid C - Per-update cost vs. Sample count m (log-x, linear-y)
  - Fig S1: RQ2 - Convergence trajectories towards Ridge reference across epochs

Outputs both PDF and 300 DPI PNG formats to `results/figures/`.
"""

import json
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent styling tokens
COLOR_DIRECT = "#D9534F"   # Crimson / Coral
COLOR_SCALED = "#2E6DA4"   # Royal Blue
COLOR_SKLEARN = "#5CB85C"  # Forest Green
COLOR_REF = "#777777"      # Slate Gray

MARKER_DIRECT = "s"  # Square
MARKER_SCALED = "o"  # Circle
MARKER_SKLEARN = "^" # Triangle


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fig1_rq1_grid_a(raw_dir: Path, out_dir: Path):
    """Figure 1: Per-update cost vs. Dimension n (Grid A)."""
    data = []
    for f in raw_dir.glob("n*_k20_m10000_cpp_scaled.json"):
        base = f.name.replace("_cpp_scaled.json", "")
        f_dir = raw_dir / f"{base}_cpp_direct.json"
        f_sk = raw_dir / f"{base}_sklearn_sgd.json"
        if not (f_dir.exists() and f_sk.exists()):
            continue

        d_dir = load_json(f_dir)
        d_sc = load_json(f)
        d_sk = load_json(f_sk)

        n = d_sc["n"]
        data.append({
            "n": n,
            "direct_med": d_dir["timing"]["us_per_update"],
            "direct_iqr": (d_dir["timing"]["iqr_ns"] / (3 * 10000)) / 1000.0,
            "scaled_med": d_sc["timing"]["us_per_update"],
            "scaled_iqr": (d_sc["timing"]["iqr_ns"] / (3 * 10000)) / 1000.0,
            "sk_med": d_sk["timing"]["us_per_update"],
            "sk_iqr": ((d_sk["timing"].get("iqr_ns", 0) / (3 * 10000)) / 1000.0) if d_sk["timing"].get("iqr_ns") else 0.0,
        })

    df = pd.DataFrame(data).sort_values("n")
    if df.empty:
        return

    plt.figure(figsize=(7, 5), dpi=300)
    plt.errorbar(df["n"], df["direct_med"], yerr=df["direct_iqr"],
                 label="C++ Direct (dense shrink)", color=COLOR_DIRECT,
                 marker=MARKER_DIRECT, capsize=4, linewidth=1.8)
    plt.errorbar(df["n"], df["scaled_med"], yerr=df["scaled_iqr"],
                 label=r"C++ Scaled ($w = s \cdot v$)", color=COLOR_SCALED,
                 marker=MARKER_SCALED, capsize=4, linewidth=1.8)
    plt.errorbar(df["n"], df["sk_med"], yerr=df["sk_iqr"],
                 label=r"scikit-learn SGDRegressor", color=COLOR_SKLEARN,
                 marker=MARKER_SKLEARN, capsize=4, linewidth=1.8)

    # Theoretical O(n) reference line passing through direct at n=10^4
    ref_idx = df[df["n"] == 10000].index
    if not ref_idx.empty:
        y_anchor = df.loc[ref_idx[0], "direct_med"]
        n_anchor = 10000.0
        n_vals = np.array(df["n"], dtype=float)
        o_n = y_anchor * (n_vals / n_anchor)
        plt.plot(n_vals, o_n, linestyle="--", color=COLOR_REF, label="Theoretical O(n) slope", alpha=0.8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Feature Dimension $n$ (log scale)", fontsize=11)
    plt.ylabel(r"Time per Sample Update ($\mu$s, log scale)", fontsize=11)
    plt.title("RQ1 Grid A: Per-Update Execution Cost vs. Dimension ($k=20, m=10,000$)", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(out_dir / "fig1_rq1_gridA_dim.png", dpi=300)
    plt.savefig(out_dir / "fig1_rq1_gridA_dim.pdf")
    plt.close()
    print("Generated Fig 1 (Grid A Dimension scaling).")


def plot_fig2_rq1_grid_b(raw_dir: Path, out_dir: Path):
    """Figure 2: Per-update cost vs. Density % (Grid B)."""
    data = []
    for f in raw_dir.glob("n5000_k*_m5000_cpp_scaled.json"):
        base = f.name.replace("_cpp_scaled.json", "")
        f_dir = raw_dir / f"{base}_cpp_direct.json"
        f_sk = raw_dir / f"{base}_sklearn_sgd.json"
        if not (f_dir.exists() and f_sk.exists()):
            continue

        d_dir = load_json(f_dir)
        d_sc = load_json(f)
        d_sk = load_json(f_sk)

        k = int(d_sc["nnz"] // d_sc["m"])
        density = (k / 5000.0) * 100.0

        data.append({
            "k": k,
            "density": density,
            "direct_med": d_dir["timing"]["us_per_update"],
            "direct_iqr": (d_dir["timing"]["iqr_ns"] / (3 * 5000)) / 1000.0,
            "scaled_med": d_sc["timing"]["us_per_update"],
            "scaled_iqr": (d_sc["timing"]["iqr_ns"] / (3 * 5000)) / 1000.0,
            "sk_med": d_sk["timing"]["us_per_update"],
            "sk_iqr": ((d_sk["timing"].get("iqr_ns", 0) / (3 * 5000)) / 1000.0) if d_sk["timing"].get("iqr_ns") else 0.0,
        })

    df = pd.DataFrame(data).sort_values("density")
    if df.empty:
        return

    plt.figure(figsize=(7, 5), dpi=300)
    plt.errorbar(df["density"], df["direct_med"], yerr=df["direct_iqr"],
                 label="C++ Direct (dense shrink)", color=COLOR_DIRECT,
                 marker=MARKER_DIRECT, capsize=4, linewidth=1.8)
    plt.errorbar(df["density"], df["scaled_med"], yerr=df["scaled_iqr"],
                 label=r"C++ Scaled ($w = s \cdot v$)", color=COLOR_SCALED,
                 marker=MARKER_SCALED, capsize=4, linewidth=1.8)
    plt.errorbar(df["density"], df["sk_med"], yerr=df["sk_iqr"],
                 label=r"scikit-learn SGDRegressor", color=COLOR_SKLEARN,
                 marker=MARKER_SKLEARN, capsize=4, linewidth=1.8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Density % = $k/n \times 100$ (log scale)", fontsize=11)
    plt.ylabel(r"Time per Sample Update ($\mu$s, log scale)", fontsize=11)
    plt.title("RQ1 Grid B: Per-Update Execution Cost vs. Matrix Density ($n=5,000, m=5,000$)", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(out_dir / "fig2_rq1_gridB_density.png", dpi=300)
    plt.savefig(out_dir / "fig2_rq1_gridB_density.pdf")
    plt.close()
    print("Generated Fig 2 (Grid B Density scaling).")


def plot_fig3_rq1_grid_c(raw_dir: Path, out_dir: Path):
    """Figure 3: Per-update cost vs. Sample count m (Grid C)."""
    data = []
    for f in raw_dir.glob("n10000_k20_m*_cpp_scaled.json"):
        base = f.name.replace("_cpp_scaled.json", "")
        f_dir = raw_dir / f"{base}_cpp_direct.json"
        f_sk = raw_dir / f"{base}_sklearn_sgd.json"
        if not (f_dir.exists() and f_sk.exists()):
            continue

        d_dir = load_json(f_dir)
        d_sc = load_json(f)
        d_sk = load_json(f_sk)

        m = d_sc["m"]
        data.append({
            "m": m,
            "direct_med": d_dir["timing"]["us_per_update"],
            "scaled_med": d_sc["timing"]["us_per_update"],
            "sk_med": d_sk["timing"]["us_per_update"],
        })

    df = pd.DataFrame(data).sort_values("m")
    if df.empty:
        return

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(df["m"], df["direct_med"], label="C++ Direct (dense shrink)",
             color=COLOR_DIRECT, marker=MARKER_DIRECT, linewidth=1.8)
    plt.plot(df["m"], df["scaled_med"], label=r"C++ Scaled ($w = s \cdot v$)",
             color=COLOR_SCALED, marker=MARKER_SCALED, linewidth=1.8)
    plt.plot(df["m"], df["sk_med"], label=r"scikit-learn SGDRegressor",
             color=COLOR_SKLEARN, marker=MARKER_SKLEARN, linewidth=1.8)

    plt.xscale("log")
    plt.xlabel("Sample Count $m$ (log scale)", fontsize=11)
    plt.ylabel(r"Time per Sample Update ($\mu$s)", fontsize=11)
    plt.title("RQ1 Grid C: Invariance of Per-Update Cost to Sample Count ($n=10,000, k=20$)", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(out_dir / "fig3_rq1_gridC_samples.png", dpi=300)
    plt.savefig(out_dir / "fig3_rq1_gridC_samples.pdf")
    plt.close()
    print("Generated Fig 3 (Grid C Sample invariance).")


def plot_fig_s1_rq2_convergence(raw_rq2_dir: Path, out_dir: Path):
    """Figure S1: Trajectory of relative distance to Ridge reference across epochs."""
    datasets = ["taiwan", "california", "kingcounty"]
    titles = ["Taiwan Real Estate", "California Housing", "King County House Sales"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    for ax, ds, title in zip(axes, datasets, titles):
        epochs_list = [1, 5, 20, 100, 500]
        dist_sk = []
        dist_sc = []
        valid_epochs = []

        for E in epochs_list:
            f_sk = raw_rq2_dir / f"{ds}_E{E}_sklearn.json"
            f_sc = raw_rq2_dir / f"{ds}_E{E}_cpp_scaled.json"
            if f_sk.exists() and f_sc.exists():
                d_sk = load_json(f_sk)
                d_sc = load_json(f_sc)
                dist_sk.append(d_sk["distance_to_ridge"])
                dist_sc.append(d_sc["distance_to_ridge"])
                valid_epochs.append(E)

        if not valid_epochs:
            continue

        ax.plot(valid_epochs, dist_sk, label="scikit-learn SGD", color=COLOR_SKLEARN,
                marker=MARKER_SKLEARN, linewidth=1.8, markersize=6)
        ax.plot(valid_epochs, dist_sc, label="C++ Scaled", color=COLOR_SCALED,
                marker=MARKER_SCALED, linewidth=1.8, linestyle="--", markersize=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs $E$ (log scale)", fontsize=10)
        ax.set_ylabel(r"$\|w - w_{\mathrm{ridge}}\| / \|w_{\mathrm{ridge}}\|$ (log scale)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "figS1_rq2_convergence.png", dpi=300)
    plt.savefig(out_dir / "figS1_rq2_convergence.pdf")
    plt.close()
    print("Generated Fig S1 (RQ2 Convergence Trajectories).")


def plot_fig_s2_rq2_rmse(raw_rq2_dir: Path, out_dir: Path):
    """Figure S2: Held-out Test RMSE Convergence across epochs vs. Closed-form Ridge baseline."""
    datasets = ["taiwan", "california", "kingcounty"]
    titles = ["Taiwan Real Estate (NTD/ping)", "California Housing ($100k)", "King County House Sales ($)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    for ax, ds, title in zip(axes, datasets, titles):
        epochs_list = [1, 5, 20, 100, 500]
        rmse_sk = []
        rmse_sc = []
        valid_epochs = []

        # Load Ridge reference baseline
        f_ridge = raw_rq2_dir / f"{ds}_ridge.json"
        ridge_rmse = None
        if f_ridge.exists():
            d_ridge = load_json(f_ridge)
            ridge_rmse = d_ridge.get("metrics", {}).get("rmse")

        for E in epochs_list:
            f_sk = raw_rq2_dir / f"{ds}_E{E}_sklearn.json"
            f_sc = raw_rq2_dir / f"{ds}_E{E}_cpp_scaled.json"
            if f_sk.exists() and f_sc.exists():
                d_sk = load_json(f_sk)
                d_sc = load_json(f_sc)
                rmse_sk.append(d_sk["metrics"]["rmse"])
                rmse_sc.append(d_sc["metrics"]["rmse"])
                valid_epochs.append(E)

        if not valid_epochs:
            continue

        ax.plot(valid_epochs, rmse_sk, label="scikit-learn SGD", color=COLOR_SKLEARN,
                marker=MARKER_SKLEARN, linewidth=1.8, markersize=6)
        ax.plot(valid_epochs, rmse_sc, label=r"C++ Scaled ($w=s\cdot v$)", color=COLOR_SCALED,
                marker=MARKER_SCALED, linewidth=1.8, linestyle="--", markersize=6)

        if ridge_rmse is not None:
            ax.axhline(ridge_rmse, color="#333333", linestyle=":", linewidth=1.6,
                       label=f"Ridge Reference ({ridge_rmse:.4f})")

        ax.set_xscale("log")
        ax.set_xlabel("Epochs $E$ (log scale)", fontsize=10)
        ax.set_ylabel("Held-Out Test RMSE", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "figS2_rq2_rmse.png", dpi=300)
    plt.savefig(out_dir / "figS2_rq2_rmse.pdf")
    plt.close()
    print("Generated Fig S2 (RQ2 RMSE Convergence).")


def plot_fig_s3_rq2_mae(raw_rq2_dir: Path, out_dir: Path):
    """Figure S3: Held-out Test MAE Convergence across epochs vs. Closed-form Ridge baseline."""
    datasets = ["taiwan", "california", "kingcounty"]
    titles = ["Taiwan Real Estate (NTD/ping)", "California Housing ($100k)", "King County House Sales ($)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    for ax, ds, title in zip(axes, datasets, titles):
        epochs_list = [1, 5, 20, 100, 500]
        mae_sk = []
        mae_sc = []
        valid_epochs = []

        # Load Ridge reference baseline
        f_ridge = raw_rq2_dir / f"{ds}_ridge.json"
        ridge_mae = None
        if f_ridge.exists():
            d_ridge = load_json(f_ridge)
            ridge_mae = d_ridge.get("metrics", {}).get("mae")

        for E in epochs_list:
            f_sk = raw_rq2_dir / f"{ds}_E{E}_sklearn.json"
            f_sc = raw_rq2_dir / f"{ds}_E{E}_cpp_scaled.json"
            if f_sk.exists() and f_sc.exists():
                d_sk = load_json(f_sk)
                d_sc = load_json(f_sc)
                mae_sk.append(d_sk["metrics"]["mae"])
                mae_sc.append(d_sc["metrics"]["mae"])
                valid_epochs.append(E)

        if not valid_epochs:
            continue

        ax.plot(valid_epochs, mae_sk, label="scikit-learn SGD", color=COLOR_SKLEARN,
                marker=MARKER_SKLEARN, linewidth=1.8, markersize=6)
        ax.plot(valid_epochs, mae_sc, label=r"C++ Scaled ($w=s\cdot v$)", color=COLOR_SCALED,
                marker=MARKER_SCALED, linewidth=1.8, linestyle="--", markersize=6)

        if ridge_mae is not None:
            ax.axhline(ridge_mae, color="#333333", linestyle=":", linewidth=1.6,
                       label=f"Ridge Reference ({ridge_mae:.4f})")

        ax.set_xscale("log")
        ax.set_xlabel("Epochs $E$ (log scale)", fontsize=10)
        ax.set_ylabel("Held-Out Test MAE", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "figS3_rq2_mae.png", dpi=300)
    plt.savefig(out_dir / "figS3_rq2_mae.pdf")
    plt.close()
    print("Generated Fig S3 (RQ2 MAE Convergence).")


def plot_fig_s4_rq2_r2(raw_rq2_dir: Path, out_dir: Path):
    """Figure S4: Held-out Test R^2 Score across epochs vs. Closed-form Ridge baseline."""
    datasets = ["taiwan", "california", "kingcounty"]
    titles = ["Taiwan Real Estate", "California Housing", "King County House Sales"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    for ax, ds, title in zip(axes, datasets, titles):
        epochs_list = [1, 5, 20, 100, 500]
        r2_sk = []
        r2_sc = []
        valid_epochs = []

        # Load Ridge reference baseline
        f_ridge = raw_rq2_dir / f"{ds}_ridge.json"
        ridge_r2 = None
        if f_ridge.exists():
            d_ridge = load_json(f_ridge)
            ridge_r2 = d_ridge.get("metrics", {}).get("r2")

        for E in epochs_list:
            f_sk = raw_rq2_dir / f"{ds}_E{E}_sklearn.json"
            f_sc = raw_rq2_dir / f"{ds}_E{E}_cpp_scaled.json"
            if f_sk.exists() and f_sc.exists():
                d_sk = load_json(f_sk)
                d_sc = load_json(f_sc)
                r2_sk.append(d_sk["metrics"]["r2"])
                r2_sc.append(d_sc["metrics"]["r2"])
                valid_epochs.append(E)

        if not valid_epochs:
            continue

        ax.plot(valid_epochs, r2_sk, label="scikit-learn SGD", color=COLOR_SKLEARN,
                marker=MARKER_SKLEARN, linewidth=1.8, markersize=6)
        ax.plot(valid_epochs, r2_sc, label=r"C++ Scaled ($w=s\cdot v$)", color=COLOR_SCALED,
                marker=MARKER_SCALED, linewidth=1.8, linestyle="--", markersize=6)

        if ridge_r2 is not None:
            ax.axhline(ridge_r2, color="#333333", linestyle=":", linewidth=1.6,
                       label=f"Ridge Reference ($R^2={ridge_r2:.4f}$)")

        ax.set_xscale("log")
        ax.set_xlabel("Epochs $E$ (log scale)", fontsize=10)
        ax.set_ylabel(r"Held-Out Test $R^2$ Score", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "figS4_rq2_r2.png", dpi=300)
    plt.savefig(out_dir / "figS4_rq2_r2.pdf")
    plt.close()
    print("Generated Fig S4 (RQ2 R^2 Convergence).")


def main():
    raw_rq1 = Path("results/raw/rq1")
    raw_rq2 = Path("results/raw/rq2")
    figs_dir = Path("results/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures into results/figures/...")
    plot_fig1_rq1_grid_a(raw_rq1, figs_dir)
    plot_fig2_rq1_grid_b(raw_rq1, figs_dir)
    plot_fig3_rq1_grid_c(raw_rq1, figs_dir)
    plot_fig_s1_rq2_convergence(raw_rq2, figs_dir)
    plot_fig_s2_rq2_rmse(raw_rq2, figs_dir)
    plot_fig_s3_rq2_mae(raw_rq2, figs_dir)
    plot_fig_s4_rq2_r2(raw_rq2, figs_dir)
    print("Figures successfully generated.")


if __name__ == "__main__":
    main()


