"""
Check that the local environment can run Vectron AI training and inference.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET}    {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET}  {msg}")


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        fail(f"Python {major}.{minor} found. Vectron requires Python 3.10+.")
        return False
    ok(f"Python {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    return True


def check_package(name: str, min_version: str | None = None, optional: bool = False) -> bool:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        if min_version and version != "unknown":
            from packaging.version import Version

            if Version(version) < Version(min_version):
                warn(f"{name} {version} found but >= {min_version} recommended.")
                return False
        ok(f"{name} {version}")
        return True
    except ImportError:
        if optional:
            warn(f"{name} not installed (optional).")
        else:
            fail(f"{name} not installed. Run: pip install -r requirements.txt")
        return False


def check_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
                ok(f"GPU {i}: {name} ({vram:.1f} GB VRAM)")
            cuda_version = torch.version.cuda or "unknown"
            ok(f"CUDA {cuda_version} - PyTorch can use GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            warn("No CUDA GPU. Apple MPS available - inference works, training is slow.")
        else:
            warn("No GPU detected. CPU-only mode - inference only, training infeasible.")
    except ImportError:
        fail("PyTorch not installed.")


def check_ram() -> None:
    try:
        import psutil

        gb = psutil.virtual_memory().total / 1024**3
        if gb < 8:
            fail(f"{gb:.1f} GB RAM. Minimum 8 GB needed.")
        elif gb < 16:
            warn(f"{gb:.1f} GB RAM. 16+ GB recommended for data preprocessing.")
        else:
            ok(f"{gb:.1f} GB RAM")
    except ImportError:
        warn("psutil not installed - skipping RAM check.")


def check_disk() -> None:
    free_gb = shutil.disk_usage(".").free / 1024**3
    if free_gb < 20:
        fail(f"{free_gb:.1f} GB free disk. Need 20+ GB for datasets + checkpoints.")
    elif free_gb < 50:
        warn(f"{free_gb:.1f} GB free. 50+ GB recommended for full Iconify dataset.")
    else:
        ok(f"{free_gb:.1f} GB free disk")


def check_external_tools() -> None:
    for tool in ("git", "node"):
        path = shutil.which(tool)
        if path:
            ok(f"{tool} found at {path}")
        else:
            warn(f"{tool} not found (optional but useful).")

    if shutil.which("svgo"):
        ok("svgo found (used by post-processor)")
    else:
        warn("svgo not found. Install: npm i -g svgo  (Python fallback will be used)")


def main() -> int:
    print("=" * 60)
    print("  Vectron AI - Environment Check")
    print("=" * 60)

    checks_passed = check_python()
    print()

    print("--- Required Python packages ---")
    for pkg, ver in [
        ("torch", "2.4.0"),
        ("transformers", "4.45.0"),
        ("datasets", "3.0.0"),
        ("peft", "0.13.0"),
        ("trl", "0.11.0"),
        ("accelerate", "1.0.0"),
        ("lxml", None),
        ("PIL", None),
        ("yaml", None),
    ]:
        check_package(pkg, ver)

    print()
    print("--- Optional packages ---")
    for pkg in ("bitsandbytes", "wandb", "cairosvg", "fastapi", "psutil"):
        check_package(pkg, optional=True)

    print()
    print("--- Hardware ---")
    check_cuda()
    check_ram()
    check_disk()

    print()
    print("--- External tools ---")
    check_external_tools()

    print()
    print("=" * 60)
    if checks_passed:
        print("  Environment looks good. Ready to train!")
    else:
        print("  Fix the issues above before training.")
    print("=" * 60)
    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
