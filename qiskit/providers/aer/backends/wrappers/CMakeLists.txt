include(Linter)
find_package(Pybind11 REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
    # We build SIMD file separately, because they will be reached only if the
    # machine running the code has SIMD support
    set(SIMD_SOURCE_FILE "../../../../../src/simulators/statevector/qv_avx2.cpp")
    if(APPLE OR UNIX)
        set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "-mfma -mavx2")
    elseif(MSVC)
        set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    endif()
endif()

set(AER_SIMULATOR_SOURCES "bindings.cc"
     "${SIMD_SOURCE_FILE}")

if(CUDA_FOUND)
    set_source_files_properties(${AER_SIMULATOR_SOURCES} PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

basic_pybind11_add_module(controller_wrappers ${AER_SIMULATOR_SOURCES})
target_include_directories(controller_wrappers PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                               PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(controller_wrappers ${AER_LIBRARIES})
target_compile_definitions(controller_wrappers PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS controller_wrappers LIBRARY DESTINATION qiskit/providers/aer/backends)

# Install redistributable dependencies
if(APPLE)
    install(FILES "${CONAN_LIB_DIRS_LLVM-OPENMP}/libomp.dylib" DESTINATION qiskit/providers/aer/backends)
endif()
