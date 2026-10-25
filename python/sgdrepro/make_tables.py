"""
Results Table Generator.

Parses raw JSON benchmark files from `results/raw/` and generates summary tables
in CSV, LaTeX (booktabs format), and Markdown formats into `results/tables/`.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def format_val_iqr(val: float, iqr: float, precision: int = 4) -> str:
    """Format median [IQR] string."""
    return f"{val:.{precision}f} [{iqr:.{precision}f}]"


def make_rq1_tables(raw_dir: Path, out_dir: Path):
    """Generate Tables for Grid A, Grid B, and Grid C."""
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for f in raw_dir.glob("*_cpp_scaled.json"):
        base = f.name.replace("_cpp_scaled.json", "")
        f_dir = raw_dir / f"{base}_cpp_direct.json"
        f_sc = f
        f_sk = raw_dir / f"{base}_sklearn_sgd.json"

        if not (f_dir.exists() and f_sk.exists()):
            continue

        d_dir = load_json(f_dir)
        d_sc = load_json(f_sc)
        d_sk = load_json(f_sk)

        m = d_sc["m"]
        n = d_sc["n"]
        # Recover k from dataset name or nnz / m
        k = int(d_sc["nnz"] // m)

        time_dir_us = d_dir["timing"]["us_per_update"]
        time_sc_us = d_sc["timing"]["us_per_update"]
        time_sk_us = d_sk["timing"]["us_per_update"]

        speedup = float(d_dir["timing"]["median_ns"] / d_sc["timing"]["median_ns"]) if d_sc["timing"]["median_ns"] > 0 else 1.0

        comp = d_sc.get("comparisons", {})
        rel_w_dir = comp.get("rel_w_sv_dir", 0.0)
        rel_w_sk = comp.get("rel_w_sv_sk", 0.0)

        records.append({
            "base": base,
            "n": n,
            "k": k,
            "m": m,
            "density_pct": (k / n) * 100.0,
            "us_direct": time_dir_us,
            "us_scaled": time_sc_us,
            "us_sklearn": time_sk_us,
            "speedup": speedup,
            "rel_w_dir": rel_w_dir,
            "rel_w_sk": rel_w_sk,
            "loss_direct": d_dir.get("final_loss", 0.0),
            "loss_scaled": d_sc.get("final_loss", 0.0),
            "loss_sklearn": d_sk.get("final_loss", 0.0),
            "n_resets": d_sc.get("n_resets", 0),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("No RQ1 records found to generate tables.")
        return

    # 1. Grid A: Varying n (k=20, m=10,000)
    df_a = df[(df["k"] == 20) & (df["m"] == 10000)].sort_values("n")
    save_table(df_a, out_dir / "rq1_gridA", [
        ("n", "n"), ("us_direct", "Direct (us/up)"),
        ("us_scaled", "Scaled (us/up)"), ("us_sklearn", "sklearn (us/up)"),
        ("speedup", "Speedup (Dir/Sc)"), ("rel_w_dir", "||w_sv-w_dir||/||w||"),
        ("rel_w_sk", "||w_sv-w_sk||/||w||"), ("loss_scaled", "Final Loss")
    ])

    # 2. Grid B: Varying density via k (n=5,000, m=5,000)
    df_b = df[(df["n"] == 5000) & (df["m"] == 5000)].sort_values("k")
    save_table(df_b, out_dir / "rq1_gridB", [
        ("k", "k"), ("density_pct", "Density (%)"),
        ("us_direct", "Direct (us/up)"), ("us_scaled", "Scaled (us/up)"),
        ("us_sklearn", "sklearn (us/up)"), ("speedup", "Speedup (Dir/Sc)"),
        ("rel_w_dir", "||w_sv-w_dir||/||w||"), ("rel_w_sk", "||w_sv-w_sk||/||w||")
    ])

    # 3. Grid C: Varying m (n=10,000, k=20)
    df_c = df[(df["n"] == 10000) & (df["k"] == 20)].sort_values("m")
    save_table(df_c, out_dir / "rq1_gridC", [
        ("m", "m"), ("us_direct", "Direct (us/up)"),
        ("us_scaled", "Scaled (us/up)"), ("us_sklearn", "sklearn (us/up)"),
        ("speedup", "Speedup (Dir/Sc)"), ("loss_scaled", "Final Loss")
    ])


def make_rq2_tables(raw_dir: Path, out_dir: Path):
    """Generate Tables for RQ2 housing datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in ["taiwan", "california", "kingcounty"]:
        ridge_file = raw_dir / f"{ds}_ridge.json"
        if not ridge_file.exists():
            continue
        d_ridge = load_json(ridge_file)

        rows = [{
            "Epochs": "-",
            "Model": "Ridge Reference",
            "RMSE": f"{d_ridge['metrics']['rmse']:.4f}",
            "MAE": f"{d_ridge['metrics']['mae']:.4f}",
            "R2": f"{d_ridge['metrics']['r2']:.4f}",
            "Dist_to_Ridge": "0.0000",
            "Train_Time_ms": f"{d_ridge['train_time_ms']:.2f}",
        }]

        for E in [1, 5, 20, 100, 500]:
            f_sk = raw_dir / f"{ds}_E{E}_sklearn.json"
            f_sc = raw_dir / f"{ds}_E{E}_cpp_scaled.json"
            if not (f_sk.exists() and f_sc.exists()):
                continue

            d_sk = load_json(f_sk)
            d_sc = load_json(f_sc)

            rows.append({
                "Epochs": str(E),
                "Model": "scikit-learn SGD",
                "RMSE": f"{d_sk['metrics']['rmse']:.4f}",
                "MAE": f"{d_sk['metrics']['mae']:.4f}",
                "R2": f"{d_sk['metrics']['r2']:.4f}",
                "Dist_to_Ridge": f"{d_sk['distance_to_ridge']:.4f}",
                "Train_Time_ms": f"{d_sk['train_time_ms']:.2f}",
            })
            rows.append({
                "Epochs": str(E),
                "Model": "C++ Scaled",
                "RMSE": f"{d_sc['metrics']['rmse']:.4f}",
                "MAE": f"{d_sc['metrics']['mae']:.4f}",
                "R2": f"{d_sc['metrics']['r2']:.4f}",
                "Dist_to_Ridge": f"{d_sc['distance_to_ridge']:.4f}",
                "Train_Time_ms": f"{d_sc['train_time_ms']:.2f}",
            })

        df_ds = pd.DataFrame(rows)
        save_table(df_ds, out_dir / f"rq2_{ds}", [
            ("Epochs", "Epochs"), ("Model", "Model"),
            ("RMSE", "RMSE"), ("MAE", "MAE"), ("R2", "R^2"),
            ("Dist_to_Ridge", "||w - w_ridge|| / ||w_ridge||"),
            ("Train_Time_ms", "Time (ms)")
        ])


def make_equivalence_summary(raw_rq2_dir: Path, out_dir: Path):
    """Generate equivalence summary table across all housing runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for f_sc in sorted(raw_rq2_dir.glob("*_cpp_scaled.json")):
        d_sc = load_json(f_sc)
        ds = d_sc["dataset"]
        e = d_sc["epochs"]
        eq = d_sc.get("equivalence", {})

        rows.append({
            "Dataset": ds,
            "Epochs": e,
            "Rel_RMSE": f"{eq.get('rel_rmse', 0.0):.2e}",
            "Rel_MAE": f"{eq.get('rel_mae', 0.0):.2e}",
            "Diff_R2": f"{eq.get('diff_r2', 0.0):.2e}",
            "Rel_W": f"{eq.get('rel_w', 0.0):.2e}",
            "Threshold": "Rel_W < 1e-10, Rel_RMSE < 1e-6",
            "Status": "PASS" if eq.get("passed", False) else "FAIL",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        save_table(df, out_dir / "equivalence_summary", [
            ("Dataset", "Dataset"), ("Epochs", "Epochs"),
            ("Rel_RMSE", "|dRMSE| / RMSE"), ("Rel_MAE", "|dMAE| / MAE"),
            ("Diff_R2", "|dR^2|"), ("Rel_W", "||dw|| / ||w||"),
            ("Threshold", "Threshold"), ("Status", "Status")
        ])


def to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as GitHub-flavored Markdown table without tabulate."""
    headers = [str(col).replace("|", "\\|") for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                s = f"{val:.4f}" if abs(val) < 1e5 and abs(val) > 1e-4 else f"{val:.2e}"
            else:
                s = str(val)
            row_str.append(s.replace("|", "\\|"))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines) + "\n"


def to_latex_table(df: pd.DataFrame) -> str:
    """Format DataFrame as LaTeX table using standard booktabs syntax without jinja2."""
    headers = [str(col).replace("_", "\\_").replace("%", "\\%").replace("^", "\\^") for col in df.columns]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                row_str.append(f"{val:.4f}" if abs(val) < 1e5 and abs(val) > 1e-4 else f"{val:.2e}")
            else:
                row_str.append(str(val).replace("_", "\\_").replace("%", "\\%"))
        lines.append(" & ".join(row_str) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}\n"])
    return "\n".join(lines)


def save_table(df: pd.DataFrame, base_path: Path, col_mapping: List[Tuple[str, str]]):
    """Save DataFrame as CSV, Markdown, and LaTeX booktabs."""
    cols = [c[0] for c in col_mapping if c[0] in df.columns]
    names = [c[1] for c in col_mapping if c[0] in df.columns]

    df_out = df[cols].copy()
    df_out.columns = names

    # CSV
    df_out.to_csv(base_path.with_suffix(".csv"), index=False)

    # Markdown
    with open(base_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(to_markdown_table(df_out))

    # LaTeX booktabs
    with open(base_path.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(to_latex_table(df_out))


def main():
    raw_rq1 = Path("results/raw/rq1")
    raw_rq2 = Path("results/raw/rq2")
    tables_dir = Path("results/tables")

    print("Generating RQ1 summary tables...")
    make_rq1_tables(raw_rq1, tables_dir)
    print("Generating RQ2 summary tables...")
    make_rq2_tables(raw_rq2, tables_dir)
    print("Generating equivalence summary table...")
    make_equivalence_summary(raw_rq2, tables_dir)
    print(f"All tables successfully written to {tables_dir}")


if __name__ == "__main__":
    main()
