# Native MPS regression

The C++23 `aer_mps_pauli_regression` target is registered with CTest when the
root project is configured with `BUILD_TESTS=ON`. The same target and tests can
be configured without building or importing the Python extension:

```sh
cmake -S test/native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target aer_mps_pauli_regression
ctest --test-dir build/native -R '^aer_mps_pauli_' --output-on-failure
```

This requires a C++23 compiler, CMake 3.20 or later, BLAS/LAPACK, OpenMP and the
same nlohmann_json dependency used by Aer. CMake package locations can be
supplied through the usual `CMAKE_PREFIX_PATH`/`<Package>_DIR` settings. On macOS,
OpenMP may also require the installed runtime's include/library paths.

`aer_mps_pauli_circuit` preserves the original four-qubit witness: Ry(pi/3) on
q0, CX(0,1), Ry(pi/4) on q2, CX(2,3), then a physical SWAP(1,2), with maximum
bond dimension two. Pre-SWAP and bond-four cases are controls. Both native and
LAPACK SVD paths are tested. Pauli strings use the simulator's reverse qubit
ordering, so `ZIII` on `{0,1,2,3}` means Z on q3.

`aer_mps_pauli_retained` imports the frozen retained tensors of that witness.
It can also be run under ASan/UBSan without executing a truncating SVD. In both
tests, a direct Gamma/lambda contraction supplies the raw norm and Z3 oracle;
no ideal, untruncated state is substituted for the retained state. Full-chain,
gapped and local queries must agree. Repeated queries, copies and a subsequent
X3 also check cache reuse and invalidation. The tests do not cover the Python
API, sampling, performance or general MPS canonicality.

`aer_mps_pauli_diagonal` checks the other production correction: diagonal gates
use a `1 x 2^n` row-vector representation when permuting gate qubits. For a
four-site state it tests all six permutations of the noncontiguous three-qubit
subset `{0,2,3}`, with both native and LAPACK SVD. Each expected amplitude is
obtained by direct contraction before the gate and multiplication by the phase
whose bits follow the explicitly requested subset order. The actual amplitudes
come from direct contraction of the exported result; no observable shortcut,
statevector method or independent SVD expansion is used as an oracle. The bond
cap is four, so this regression does not request bond-cap truncation. Controls
require the fixture to distinguish both a missing phase and reversed subset
order. The diagonal mode can be run separately under ASan/UBSan; the pre-fix
implementation is an expected memory-error negative control and must only be
run in a separate sanitizer-instrumented process.

This is a reduced permanent version of the R8 private `diagonal_checks.cpp`
control (2026-10-11), whose SHA256 is
`0e0082e5486c6de03c3689169b970dfb2ce95622d72b58737e1d0cbd48ce4e81`.
It retains its explicit basis-bit phase indexing and full-amplitude comparison,
with 12 focused cases rather than claiming to replay the original 60-case grid.

`aer_mps_pauli_lifecycle` checks non-default per-instance bond and truncation
settings through copy construction, assignment, self-assignment, destructive
export and reinitialization. It verifies that export empties the source payload
without resetting these settings, then restores the saved tensors after
initializing the source to the required number of sites. It does not establish
direct import into an empty object or repeated export of an empty object.

## Fixture and source provenance

The witness and direct-contraction oracle are reduced from the author's
retained C++23 probe used to investigate issue #2320 before PR #2472. The probe
SHA256 is `18640569bb8cd61d6d15cc88b0ac3594efab7a7164d8cf7bc6bc8290688109b9`.
The fixture is byte-identical to `minimal/0-2-1.json` from the retained native
validation archive `frozen-fixtures.tar.gz` (2026-10-11), generated on source
`fe964bf3ec37d27175325bb306e752d1c3cb9591` using native SVD, bond cap two and
truncation threshold zero. Its SHA256 is
`4b24768bed91055ee34d27638c89369b0cb4d6a87f11a309ac964d57b0cf7f0b`.
It is committed here, so the tests have no dependency on that private archive.
The oracle contracts the committed tensors independently; it does not trust
stored expected query outputs. On the implementation before the fix, the
retained/full Z3 agrees while the local Z3 does not.
