"""
Environment Information Collector.

Collects system metadata including CPU model, core counts, RAM, operating system,
C++ compiler details, Python package versions, and CPU pinning / affinity status.
Saves the provenance metadata to `results/env.json`.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_cpu_info() -> dict:
    """Detect CPU model and core counts across operating systems."""
    cpu_name = platform.processor() or "Unknown CPU"
    # On Windows, try querying wmic or PowerShell for detailed brand string
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                cpu_name = res.stdout.strip().splitlines()[0]
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return {
        "model": cpu_name,
        "logical_cores": os.cpu_count() or 1,
        "architecture": platform.machine(),
    }


def get_ram_info() -> dict:
    """Detect total system RAM in gigabytes."""
    total_ram_gb = None
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                bytes_ram = int(res.stdout.strip())
                total_ram_gb = round(bytes_ram / (1024**3), 2)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        total_ram_gb = round(kb / (1024**2), 2)
                        break
        except Exception:
            pass

    return {"total_gb": total_ram_gb}


def get_compiler_info() -> dict:
    """Detect available C++ compiler (g++ or clang++) and version."""
    for comp in ["g++", "clang++"]:
        try:
            res = subprocess.run(
                [comp, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                version_line = res.stdout.strip().splitlines()[0]
                return {
                    "compiler": comp,
                    "version_string": version_line,
                    "default_flags": "-std=c++17 -O2 -DNDEBUG -Wall -Wextra",
                }
        except Exception:
            continue

    return {
        "compiler": "unknown",
        "version_string": "none found",
        "default_flags": "-std=c++17 -O2 -DNDEBUG -Wall -Wextra",
    }


def get_python_env() -> dict:
    """Query versions of Python and primary scientific dependencies."""
    packages = {}
    for pkg in ["numpy", "scipy", "sklearn", "pandas", "matplotlib", "pytest", "ucimlrepo"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not installed"

    return {
        "python_version": sys.version.split()[0],
        "packages": packages,
    }


def get_pinning_status() -> dict:
    """Check if process CPU affinity or thread pinning is configured."""
    affinity = None
    try:
        # Check if psutil or os.sched_getaffinity is available
        if hasattr(os, "sched_getaffinity"):
            affinity = list(os.sched_getaffinity(0))
    except Exception:
        pass

    return {
        "affinity_supported": affinity is not None,
        "affinity_cores": affinity,
        "notes": (
            "Single-threaded execution enforced via OMP_NUM_THREADS=1, "
            "OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1."
        ),
    }


def collect_environment() -> dict:
    """Aggregate all environment information into a single dictionary."""
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "compiler": get_compiler_info(),
        "python": get_python_env(),
        "concurrency": get_pinning_status(),
    }


def main():
    """Main entrypoint to collect and save environment information."""
    env = collect_environment()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "env.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(f"Environment metadata written to {out_file}")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
