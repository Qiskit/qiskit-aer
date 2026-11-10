# Native tests can also be configured without the Python extension.
cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aer_native_tests LANGUAGES CXX)
  enable_testing()
  get_filename_component(AER_SOURCE_ROOT "../.." ABSOLUTE)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(Threads REQUIRED)
  find_package(OpenMP REQUIRED)
  find_package(nlohmann_json 3.1.1 REQUIRED)
  set(AER_NATIVE_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} Threads::Threads
                          OpenMP::OpenMP_CXX
                          nlohmann_json::nlohmann_json)
else()
  set(AER_SOURCE_ROOT "${PROJECT_SOURCE_DIR}")
  set(AER_NATIVE_LIBRARIES ${AER_LIBRARIES})
  if(TARGET AER_DEPENDENCY_PKG::llvm-openmp)
    list(APPEND AER_NATIVE_LIBRARIES AER_DEPENDENCY_PKG::llvm-openmp)
  elseif(TARGET OpenMP::OpenMP_CXX)
    list(APPEND AER_NATIVE_LIBRARIES OpenMP::OpenMP_CXX)
  endif()
endif()

add_executable(aer_mps_pauli_regression mps_pauli.cpp)
set_target_properties(aer_mps_pauli_regression PROPERTIES
                      CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(aer_mps_pauli_regression PRIVATE
                           "${AER_SOURCE_ROOT}/src" ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(aer_mps_pauli_regression PRIVATE ${AER_NATIVE_LIBRARIES})

foreach(mode IN ITEMS circuit retained diagonal lifecycle)
  add_test(NAME aer_mps_pauli_${mode}
           COMMAND aer_mps_pauli_regression ${mode}
                   "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mps-pauli-truncated.json")
  set_tests_properties(aer_mps_pauli_${mode} PROPERTIES
                       ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                       TIMEOUT 30 LABELS "native;mps")
endforeach()
