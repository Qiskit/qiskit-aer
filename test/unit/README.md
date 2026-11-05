# Native LAPACK SVD regression tests

These tests compile the actual Aer SVD implementation and matrix class. One
executable uses the selected BLAS/LAPACK library; the other injects LAPACK
statuses and invalid factors while retaining real BLAS reconstruction.

Run without building or importing the Python extension:

```sh
cmake -S test/unit -B build/lapack-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build/lapack-tests --parallel 2
ctest --test-dir build/lapack-tests --output-on-failure
```

The same targets are available in a normal Aer build with `BUILD_TESTS=ON`.
The Linux jobs in `.github/workflows/tests.yml` build and run both targets with
address and undefined-behavior sanitizers before the Python integration tests.
Failures propagate through the existing required `Finalize tests` check.
For local memory checks, configure a fresh test build with
`-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`.
No circuit simulation or performance benchmark is performed.
