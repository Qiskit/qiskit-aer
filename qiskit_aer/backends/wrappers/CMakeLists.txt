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

macro(configure_target target_name)
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
  target_link_libraries(${target_name} ${AER_LIBRARIES})
  target_compile_definitions(${target_name} PRIVATE ${AER_COMPILER_DEFINITIONS})
  install(TARGETS ${target_name} LIBRARY DESTINATION qiskit_aer/backends)
endmacro()

# Build the CPU backend
set(BACKEND_MODULE_NAME "controller_wrappers_cpu")
configure_file(bindings.cc.in bindings_cpu.cc)
basic_pybind11_add_module(controller_wrappers_cpu bindings_cpu.cc "${SIMD_SOURCE_FILE}")

if(DEFINED SIMD_SOURCE_FILE)
  string(REPLACE ";" " " SIMD_FLAGS "${SIMD_FLAGS_LIST}")
  set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
endif()

set_target_properties(controller_wrappers_cpu PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS}")
configure_target(controller_wrappers_cpu)

# Build the CUDA backend
if(AER_THRUST_BACKEND STREQUAL "CUDA")
    set(BACKEND_MODULE_NAME "controller_wrappers_cuda")
    configure_file(bindings.cc.in bindings_cuda.cc)
    basic_pybind11_add_module(controller_wrappers_cuda bindings_cuda.cc "${SIMD_SOURCE_FILE}")

    include(nvcc_add_compiler_options)
    set_source_files_properties(bindings_cuda.cc PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(bindings_cuda.cc PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")

    if(DEFINED SIMD_SOURCE_FILE)
       set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
       nvcc_add_compiler_options_list("${SIMD_FLAGS_LIST}" SIMD_FLAGS_OUT)
       set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS} ${SIMD_FLAGS_OUT}")
    endif()

    string(STRIP ${AER_COMPILER_FLAGS} AER_COMPILER_FLAGS_STRIPPED)
    nvcc_add_compiler_options(${AER_COMPILER_FLAGS_STRIPPED} AER_COMPILER_FLAGS_OUT)
    set_target_properties(controller_wrappers_cuda PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS_OUT}")
    enable_language(CUDA)
    configure_target(controller_wrappers_cuda)
# Build the ROCm backend
elseif(AER_THRUST_BACKEND STREQUAL "ROCM")
    set(BACKEND_MODULE_NAME "controller_wrappers_rocm")
    configure_file(bindings.cc.in bindings_rocm.cc)
    basic_pybind11_add_module(controller_wrappers_rocm bindings_rocm.cc "${SIMD_SOURCE_FILE}")

    if(NOT DEFINED SIMD_SOURCE_FILE)
      message(FATAL_ERROR "ROCm supported target machines are expected to be SIMD-enabled.") 
    endif()
    
    set_source_files_properties(
      bindings_rocm.cc
      ${SIMD_SOURCE_FILE} 
      PROPERTIES LANGUAGE CXX)
    
    target_compile_options(controller_wrappers_rocm PRIVATE ${ROCM_EXTRA_FLAGS} ${SIMD_FLAGS_LIST})
    target_compile_definitions(controller_wrappers_rocm PRIVATE ${ROCM_EXTRA_DEFS} ${AER_COMPILER_DEFINITIONS})
    set_target_properties(controller_wrappers_rocm PROPERTIES COMPILE_FLAGS "${AER_COMPILER_FLAGS}")
    configure_target(controller_wrappers_rocm)
endif()

# Install redistributable dependencies
install(FILES ${BACKEND_REDIST_DEPS} DESTINATION qiskit_aer/backends)
