include(Linter)
include(cython_utils)


# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

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

add_cython_module(qasm_controller_wrapper)
add_cython_module(statevector_controller_wrapper)
add_cython_module(unitary_controller_wrapper)

# Install redistributable dependencies
if(APPLE)
    install(FILES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/macos/lib/libomp.dylib" DESTINATION qiskit/providers/aer/backends)
endif()
