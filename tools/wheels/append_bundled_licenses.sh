#!/usr/bin/env bash
#
# Append the bundled-library license notices for the platform being built to
# LICENSE.txt, so that they are picked up into the wheel's dist-info directory.
#
# Run as cibuildwheel's before-build step.  The wheels bundle shared libraries
# that Aer itself does not ship in the source tree (OpenBLAS, the GCC runtime
# libraries, LLVM's OpenMP runtime), and those licenses require that their
# terms accompany the binaries.

set -euo pipefail

PROJECT_DIR="${1:-$PWD}"
LICENSE_FILE="$PROJECT_DIR/LICENSE.txt"

case "$(uname -s)" in
    Linux*)   notices=LICENSE_linux.txt ;;
    Darwin*)  notices=LICENSE_macos.txt ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT) notices=LICENSE_windows.txt ;;
    *) echo "unrecognized platform: $(uname -s)" >&2; exit 1 ;;
esac

# Guard against running twice over the same checkout, which cibuildwheel does
# when it builds several wheels in a row.
if grep -q "^This binary distribution of Qiskit Aer bundles" "$LICENSE_FILE"; then
    echo "bundled-library notices already present in $LICENSE_FILE"
    exit 0
fi

{
    printf '\n\n----\n\n'
    cat "$PROJECT_DIR/tools/wheels/$notices"
} >> "$LICENSE_FILE"

echo "appended $notices to $LICENSE_FILE"
