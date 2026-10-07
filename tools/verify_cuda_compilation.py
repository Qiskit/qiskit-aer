# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Check compiler selection in a Linux x86-64 CUDA compilation database.

Configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON, then run::

    python tools/verify_cuda_compilation.py build/compile_commands.json

For an existing Ninja build, no reconfiguration is needed::

    ninja -C build -t compdb | python tools/verify_cuda_compilation.py -

This checks generated build commands, not successful compilation or execution.
Run it in addition to building and testing the wheel.
"""

import argparse
import json
from pathlib import Path
import shlex
import sys


def verify(entries):
    """Reject CUDA compilation of host AVX2 code and misplaced compiler flags."""
    sources = {"bindings.cc": [], "qv_avx2.cpp": []}
    for entry in entries:
        name = Path(entry["file"]).name
        if name in sources:
            args = entry.get("arguments")
            if args is None:
                args = shlex.split(entry["command"])
            sources[name].append(args)

    for name, commands in sources.items():
        if not commands:
            raise ValueError(f"No command for {name}; expected an x86-64 CUDA build")
        for args in commands:
            uses_nvcc = any(Path(arg).name == "nvcc" for arg in args)
            host_options = any(
                arg in ("-Xcompiler", "--compiler-options")
                or arg.startswith(("-Xcompiler=", "--compiler-options="))
                for arg in args
            )
            if name == "bindings.cc":
                if not uses_nvcc:
                    raise ValueError("bindings.cc must be compiled with nvcc")
                if "-mavx2" in args or "-mfma" in args:
                    raise ValueError("bindings.cc must not inherit host AVX2/FMA flags")
            else:
                if uses_nvcc or host_options:
                    raise ValueError("qv_avx2.cpp must use C++ without nvcc host-flag syntax")
                if not {"-mavx2", "-mfma"}.issubset(args):
                    raise ValueError("qv_avx2.cpp is missing AVX2/FMA compiler flags")
    print("PASS: CUDA bindings and host AVX2 compiler selection and flags")


def main():
    """Read a CMake compilation database or Ninja compdb output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", help="Compilation database JSON, or - for stdin")
    args = parser.parse_args()
    try:
        if args.database == "-":
            entries = json.load(sys.stdin)
        else:
            entries = json.loads(Path(args.database).read_text(encoding="utf-8"))
        verify(entries)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
