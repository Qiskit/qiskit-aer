# Cython OP extensions

include(Linter)
include(cython_utils)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CYTHON_INSTALL_DIR "qiskit/providers/aer/pulse/cy/")
add_cython_module(numeric_integrator_wrapper numeric_integrator.cpp)
add_cython_module(test_python_to_cpp)
