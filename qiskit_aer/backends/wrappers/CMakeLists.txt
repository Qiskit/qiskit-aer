include(Linter)
find_package(Pybind11 REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(AER_LIBRARIES
	${AER_LIBRARIES}
	${THRUST_DEPENDANT_LIBS}
	${MPI_DEPENDANT_LIBS})

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64")
    if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      # We build SIMD filed separately, because they will be reached only if the
      # machine running the code has SIMD support
      set(SIMD_SOURCE_FILE "${PROJECT_SOURCE_DIR}/src/simulators/statevector/qv_avx2.cpp")
    endif()
endif()

set(AER_SIMULATOR_SOURCES "bindings.cc" "${SIMD_SOURCE_FILE}")
basic_pybind11_add_module(controller_wrappers "${AER_SIMULATOR_SOURCES}")

if(AER_THRUST_BACKEND STREQUAL "CUDA")
    include(nvcc_add_compiler_options)
    set_source_files_properties(bindings.cc PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bindings.cc PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")

    if(DEFINED SIMD_SOURCE_FILE)
       set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
       nvcc_add_compiler_options_list("${SIMD_FLAGS_LIST}" SIMD_FLAGS_OUT)
       set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS} ${SIMD_FLAGS_OUT}")
    endif()

    string(STRIP ${AER_COMPILER_FLAGS} AER_COMPILER_FLAGS_STRIPPED)
    nvcc_add_compiler_options(${AER_COMPILER_FLAGS_STRIPPED} AER_COMPILER_FLAGS_OUT)
    set_target_properties(controller_wrappers PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS_OUT}")
    enable_language(CUDA)
elseif(AER_THRUST_BACKEND STREQUAL "ROCM")

    if(NOT DEFINED SIMD_SOURCE_FILE)
      message(FATAL_ERROR "ROCm supported target machines are expected to be SIMD-enabled.") 
    endif()
    
    set_source_files_properties(
      bindings.cc 
      ${SIMD_SOURCE_FILE} 
      PROPERTIES LANGUAGE CXX)
    
    target_compile_options(controller_wrappers PRIVATE ${ROCM_EXTRA_FLAGS} ${SIMD_FLAGS_LIST})
    target_compile_definitions(controller_wrappers PRIVATE ${ROCM_EXTRA_DEFS} ${AER_COMPILER_DEFINITIONS})
    set_target_properties(controller_wrappers PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS}")
else()
    if(DEFINED SIMD_SOURCE_FILE)
       string(REPLACE ";" " " SIMD_FLAGS "${SIMD_FLAGS_LIST}")
       set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
    endif()
    set_target_properties(controller_wrappers PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS}")
endif()
target_include_directories(controller_wrappers PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                               PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(controller_wrappers ${AER_LIBRARIES})
target_compile_definitions(controller_wrappers PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS controller_wrappers LIBRARY DESTINATION qiskit_aer/backends)

# Install redistributable dependencies
install(FILES ${BACKEND_REDIST_DEPS} DESTINATION qiskit_aer/backends)
