# -*- coding: utf-8 -*-

# Copyright 2026, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Check that built wheels carry the licenses of the libraries they bundle.

This inspects the wheel archives directly, rather than an installed copy, so
that it can run for wheels that cannot be imported on the build machine.  The
GPU wheels need a CUDA device to import, so they skip cibuildwheel's test
step and the equivalent check in tools/verify_wheels.py never runs for them.

Usage: python tools/verify_wheel_licenses.py wheelhouse/*.whl
"""

import fnmatch
import pathlib
import sys
import zipfile

# Each entry maps the name a library is given in the "Name:" field of the
# notices under tools/wheels/ to the filenames it is vendored under.
BUNDLED = {
    "OpenBLAS": ["libopenblas*"],
    "GCC runtime library": ["libgfortran*", "libgomp*"],
    "libquadmath": ["libquadmath*"],
    "LLVM OpenMP runtime library": ["libomp.dylib"],
}

HEADER = "This binary distribution of Qiskit Aer bundles"


def bundled_libraries(names):
    """Return the names of the notices a wheel containing ``names`` needs."""
    # auditwheel and delvewheel vendor into a sibling <distribution>.libs
    # directory, and delocate alongside the extension modules.
    candidates = [
        name for name in names if ".libs/" in name or name.startswith("qiskit_aer/backends/")
    ]
    return sorted(
        library
        for library, patterns in BUNDLED.items()
        if any(
            fnmatch.fnmatch(pathlib.PurePosixPath(name).name, pattern)
            for name in candidates
            for pattern in patterns
        )
    )


def verify(path):
    """Check one wheel, returning a list of problems found."""
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        expected = bundled_libraries(names)
        if not expected:
            print(f"{path.name}: no bundled libraries; nothing to check")
            return []

        licenses = [
            name
            for name in names
            if ".dist-info/" in name and pathlib.PurePosixPath(name).name == "LICENSE.txt"
        ]
        if not licenses:
            return [f"{path.name}: no LICENSE.txt in dist-info"]

        problems = []
        for name in licenses:
            text = wheel.read(name).decode("utf-8")
            if HEADER not in text:
                problems.append(f"{path.name}: {name} is missing the bundled-library notices")
                continue
            for library in expected:
                if f"Name: {library}" not in text:
                    problems.append(f"{path.name}: {name} is missing the notice for {library}")
            # The notices name the directory the libraries are vendored
            # into, which differs between the CPU and GPU distributions.  A
            # glob matching nothing only warns.  The Linux notices are the
            # same for every architecture, but libquadmath is not bundled on
            # all of them.
            for pattern in (
                line[len("Files:") :].strip()
                for line in text.splitlines()
                if line.startswith("Files:")
            ):
                for glob in (part.strip() for part in pattern.split(",")):
                    if not any(fnmatch.fnmatch(entry, glob) for entry in names):
                        print(f"{path.name}: warning: {name} lists {glob}, which matches nothing")

    if not problems:
        print(f"{path.name}: licenses for {', '.join(expected)} verified")
    return problems


def main(argv):
    paths = [pathlib.Path(arg) for arg in argv]
    if not paths:
        print("usage: verify_wheel_licenses.py WHEEL [WHEEL ...]", file=sys.stderr)
        return 2
    problems = [problem for path in paths for problem in verify(path)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
