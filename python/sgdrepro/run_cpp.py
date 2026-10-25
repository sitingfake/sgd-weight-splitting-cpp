"""
C++ SGD Runner and Provenance Wrapper.

Invokes the compiled C++ binary `build/sgd_cpp`, collects output JSON metrics,
loads binary weight dumps, and attaches git provenance metadata.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

CPP_BIN_DEFAULT = Path("build/sgd_cpp.exe" if os.name == "nt" else "build/sgd_cpp")


def get_git_provenance() -> Dict[str, Any]:
    """Retrieve current git commit hash and dirty status."""
    commit_hash = "unknown"
    dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            commit_hash = res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_res.returncode == 0:
            dirty = bool(status_res.stdout.strip())
    except Exception:
        pass

    return {"git_commit": commit_hash, "dirty": dirty}


def run_cpp_sgd(
    data_path: Path | str,
    impl: str = "scaled",
    epochs: int = 3,
    alpha: float = 1e-4,
    schedule: str = "invscaling",
    eta0: float = 0.01,
    power_t: float = 0.25,
    optimal_init: float = 0.0,
    fit_intercept: bool = False,
    intercept_decay: float = 1.0,
    max_samples: Optional[int] = None,
    warmup: int = 1,
    repeats: int = 7,
    dump_w: bool = True,
    bin_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Execute the C++ SGD binary and parse results.

    Returns
    -------
    result_json : dict
        Parsed JSON output enriched with git provenance.
    w : np.ndarray or None
        Loaded float64 weight array if dump_w was enabled.
    """
    executable = bin_path or CPP_BIN_DEFAULT
    if not executable.exists():
        raise FileNotFoundError(f"C++ binary not found at {executable}; run `make build` first.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        json_out_path = tmp_path / "result.json"
        w_out_path = tmp_path / "w.bin" if dump_w else None

        cmd = [
            str(executable),
            "--data", str(data_path),
            "--impl", impl,
            "--epochs", str(epochs),
            "--alpha", str(alpha),
            "--schedule", schedule,
            "--eta0", str(eta0),
            "--power_t", str(power_t),
            "--optimal_init", str(optimal_init),
            "--fit_intercept", "1" if fit_intercept else "0",
            "--intercept_decay", str(intercept_decay),
            "--warmup", str(warmup),
            "--repeats", str(repeats),
            "--out", str(json_out_path),
        ]

        if max_samples is not None and max_samples > 0:
            cmd.extend(["--max_samples", str(max_samples)])

        if dump_w:
            cmd.extend(["--dump_w", str(w_out_path)])

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check divergence or failure
        if proc.returncode not in (0, 3):
            raise RuntimeError(
                f"C++ binary failed with exit code {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        if not json_out_path.exists():
            raise RuntimeError(
                f"C++ binary exited with {proc.returncode} but did not produce {json_out_path}.\n"
                f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )

        with open(json_out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.update(get_git_provenance())

        w = None
        if dump_w and w_out_path.exists() and not data.get("diverged", False):
            w = np.fromfile(w_out_path, dtype=np.float64)

        return data, w
