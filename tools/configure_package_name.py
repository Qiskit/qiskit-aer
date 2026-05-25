#!/usr/bin/env python3
# pylint: disable=invalid-name

"""Rewrite ``pyproject.toml`` in place to publish a non-default wheel name.

PEP 621 (and scikit-build-core) require ``[project].name`` to be a static
string. Aer publishes three wheels — ``qiskit-aer``, ``qiskit-aer-gpu``,
``qiskit-aer-gpu-cu11`` — from the same source tree. The legacy ``setup.py``
read ``QISKIT_AER_PACKAGE_NAME`` and ``QISKIT_AER_CUDA_MAJOR`` from the
environment to switch between them at install time. With ``setup.py`` gone,
this script does the equivalent: invoked from CI before ``cibuildwheel``,
it patches the project name and adds the relevant CUDA runtime dependencies
to ``[project].dependencies``.

Environment variables read:

  QISKIT_AER_PACKAGE_NAME
      The PyPI distribution name to use. Defaults to ``qiskit-aer``.
  QISKIT_AER_CUDA_MAJOR
      ``11`` or ``12``. Selects which family of nvidia-* runtime requirements
      to add. Only consulted when the package name contains ``gpu`` and not
      ``rocm``.
  QISKIT_ADD_CUDA_REQUIREMENTS
      ``false``/``off``/``no`` disables the CUDA-runtime requirement
      injection (e.g. when CUDA is provided by the host system). Default
      is to inject.

The script is idempotent against an already-patched ``pyproject.toml`` if
the requested package name matches the current one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

CUDA11_REQUIREMENTS = [
    "nvidia-cuda-runtime-cu11>=11.8.89",
    "nvidia-cublas-cu11>=11.11.3.6",
    "nvidia-cusolver-cu11>=11.4.1.48",
    "nvidia-cusparse-cu11>=11.7.5.86",
    "cuquantum-cu11>=23.3.0,<24.11.0",
]
CUDA12_REQUIREMENTS = [
    "nvidia-cuda-runtime-cu12>=12.1.105",
    "nvidia-nvjitlink-cu12",
    "nvidia-cublas-cu12>=12.1.3.1",
    "nvidia-cusolver-cu12>=11.4.5.107",
    "nvidia-cusparse-cu12>=12.1.0.106",
    "cuquantum-cu12>=23.3.0,<24.11.0",
]
CUDA11_CLASSIFIER = "Environment :: GPU :: NVIDIA CUDA :: 11"
CUDA12_CLASSIFIER = "Environment :: GPU :: NVIDIA CUDA :: 12"


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() not in {"false", "off", "no", "0", ""}


def main() -> int:
    package_name = os.getenv("QISKIT_AER_PACKAGE_NAME", "qiskit-aer")
    cuda_major = os.getenv("QISKIT_AER_CUDA_MAJOR", "12")
    add_cuda = _bool_env("QISKIT_ADD_CUDA_REQUIREMENTS", default=True)

    text = PYPROJECT.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    current_name = parsed.get("project", {}).get("name", "qiskit-aer")

    # Patch the [project].name line. We do this with a string replace rather
    # than a TOML round-trip to preserve comments, formatting, and key order.
    if current_name != package_name:
        old = f'name = "{current_name}"'
        new = f'name = "{package_name}"'
        if old not in text:
            print(f"ERROR: could not find {old!r} in pyproject.toml", file=sys.stderr)
            return 1
        text = text.replace(old, new, 1)

    needs_cuda_reqs = add_cuda and "gpu" in package_name and "rocm" not in package_name
    if needs_cuda_reqs:
        if cuda_major == "11":
            extra_reqs = CUDA11_REQUIREMENTS
            extra_classifier = CUDA11_CLASSIFIER
        else:
            extra_reqs = CUDA12_REQUIREMENTS
            extra_classifier = CUDA12_CLASSIFIER

        # Inject extra dependencies and a CUDA classifier. We splice into the
        # existing [project] table by appending lines to the ``dependencies``
        # and ``classifiers`` arrays; the markers are the closing bracket of
        # each array on its own line.
        existing_deps = parsed["project"]["dependencies"]
        new_deps_block = "dependencies = [\n"
        for dep in existing_deps + extra_reqs:
            new_deps_block += f'  "{dep}",\n'
        new_deps_block += "]"

        # Replace the dependencies array. Use a regex-free approach: find
        # the start and end markers.
        start = text.find("dependencies = [")
        if start == -1:
            print("ERROR: could not locate dependencies array", file=sys.stderr)
            return 1
        end = text.find("\n]", start)
        if end == -1:
            print("ERROR: could not find end of dependencies array", file=sys.stderr)
            return 1
        text = text[:start] + new_deps_block + text[end + 2 :]

        # Append the CUDA classifier.
        classifiers_start = text.find("classifiers = [")
        classifiers_end = text.find("\n]", classifiers_start)
        if extra_classifier not in text[classifiers_start:classifiers_end]:
            insertion = f'  "{extra_classifier}",\n'
            text = text[:classifiers_end] + "\n" + insertion.rstrip("\n") + text[classifiers_end:]

    PYPROJECT.write_text(text, encoding="utf-8")
    print(f"Configured pyproject.toml: name={package_name}", end="")
    if needs_cuda_reqs:
        print(f", cuda_major={cuda_major}")
    else:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
