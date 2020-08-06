include(Linter)
find_package(Pybind11 REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # We build SIMD filed separately, because they will be reached only if the
  # machine running the code has SIMD support
  set(SIMD_SOURCE_FILE "../../../../../src/simulators/statevector/qv_avx2.cpp")
endif()

set(AER_SIMULATOR_SOURCES "bindings.cc" "${SIMD_SOURCE_FILE}")
basic_pybind11_add_module(controller_wrappers "${AER_SIMULATOR_SOURCES}")

if(AER_THRUST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(bindings.cc PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bindings.cc PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")

    set(L ${SIMD_FLAGS_LIST})
    list(TRANSFORM L PREPEND " --compiler-options ")
    string(REPLACE ";" " " SIMD_FLAGS_OUT ${L})
    set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS} ${SIMD_FLAGS_OUT}")

    string(STRIP ${AER_COMPILER_FLAGS} AER_COMPILER_FLAGS_STRIPPED)
    string(REPLACE " " ";" L ${AER_COMPILER_FLAGS_STRIPPED})
    list(TRANSFORM L PREPEND " --compiler-options ")
    string(REPLACE ";" " " AER_COMPILER_FLAGS_OUT ${L})
    set_target_properties(controller_wrappers PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS_OUT}")
else()
    string(REPLACE ";" " " SIMD_FLAGS "${SIMD_FLAGS_LIST}")
    set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
    set_target_properties(controller_wrappers PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS}")
endif()
target_include_directories(controller_wrappers PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                               PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(controller_wrappers ${AER_LIBRARIES})
target_compile_definitions(controller_wrappers PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS controller_wrappers LIBRARY DESTINATION qiskit/providers/aer/backends)

# Install redistributable dependencies
if(APPLE)
    install(FILES "${CONAN_LIB_DIRS_LLVM-OPENMP}/libomp.dylib" DESTINATION qiskit/providers/aer/backends)
endif()
