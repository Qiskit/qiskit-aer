# Cython OP extensions

include(Linter)
include(cython_utils)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -static " "" AER_COMPILER_FLAGS "${CMAKE_CXX_FLAGS}")

# Set some general flags
if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
                    solved at runtime by the loader")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup")
    else()
        # -flat_namespace linker flag is needed otherwise dynamic symbol resolution doesn't work as expected with GCC.
        # Symbols with the same name exist in different .so, so the loader just takes the first one it finds,
        # which is usually the one from the first .so loaded.
        # See: Two-Leve namespace symbol resolution
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup -flat_namespace")
    endif()
    unset(PYTHON_LIBRARIES)
endif()

set(CYTHON_INSTALL_DIR "qiskit/providers/aer/openpulse/cy/")
set(CYTHON_FLAGS "${AER_COMPILER_FLAGS} ${AER_LINKER_FLAGS}")
message("JUAN: ${AER_COMPILER_FLAGS} ${AER_LINKER_FLAGS}")
add_cython_module(numeric_integrator_wrapper numeric_integrator.cpp)
add_cython_module(test_python_to_cpp)
