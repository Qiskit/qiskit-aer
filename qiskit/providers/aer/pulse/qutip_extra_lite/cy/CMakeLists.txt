include(Linter)
include(cython_utils)
# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CYTHON_INSTALL_DIR "qiskit/providers/aer/pulse/qutip_extra_lite/cy")
add_cython_module(spmath)
