# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Check relative CUDA library RPATHs in a Linux wheel, without importing Aer.

Usage: python tools/verify_cuda_wheel.py path/to/qiskit_aer_gpu.whl

Requires readelf on PATH. Use on an unrepaired wheel built with
AER_PYTHON_CUDA_ROOT; auditwheel-repaired wheels can use a different layout.
No GPU or CUDA installation is needed. This checks ELF metadata, not runtime
library availability; also test the installed wheel without LD_LIBRARY_PATH.
"""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

CUDA_LIBRARY_DIRS = (
    "nvidia/cublas/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/cuda_runtime/lib",
    "cutensor/lib",
    "cuquantum/lib",
)


def verify(wheel):
    """Check the actual extension's dynamic section for all six relative paths."""
    with zipfile.ZipFile(wheel) as archive:
        extensions = [
            name
            for name in archive.namelist()
            if name.startswith("qiskit_aer/backends/controller_wrappers.") and name.endswith(".so")
        ]
        if len(extensions) != 1:
            raise ValueError("Expected exactly one Linux controller_wrappers extension")
        with tempfile.TemporaryDirectory(prefix="aer-rpath-") as directory:
            # Use a fixed filename rather than extracting paths supplied by the archive.
            extension = Path(directory) / "controller_wrappers.so"
            extension.write_bytes(archive.read(extensions[0]))
            result = subprocess.run(
                ["readelf", "--wide", "--dynamic", str(extension)],
                check=True,
                capture_output=True,
                text=True,
            )
    paths = set()
    for entry in re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]*)\]", result.stdout):
        paths.update(entry.split(":"))
    expected = {f"$ORIGIN/../../{directory}" for directory in CUDA_LIBRARY_DIRS}
    missing = expected - paths
    if missing:
        raise ValueError("Missing relative CUDA paths: " + ", ".join(sorted(missing)))
    malformed = {f"/../../{directory}" for directory in CUDA_LIBRARY_DIRS} & paths
    if malformed:
        raise ValueError("Found expanded-away ORIGIN paths: " + ", ".join(sorted(malformed)))
    print("PASS: all six relative CUDA library paths are preserved in the wheel ELF")


def main():
    """Validate the specified wheel without installing or executing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        verify(args.wheel)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
