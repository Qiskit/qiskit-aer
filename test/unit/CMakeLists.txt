cmake_minimum_required(VERSION 3.12)

# These tests can also be configured without building the Python extension.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aer_lapack_unit_tests LANGUAGES CXX)
  enable_testing()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

foreach(mode IN ITEMS numerical fault)
  add_executable(lapack_svd_${mode} lapack_svd.cpp)
  target_compile_features(lapack_svd_${mode} PRIVATE cxx_std_14)
  target_include_directories(lapack_svd_${mode} PRIVATE ../../src)
  target_link_libraries(lapack_svd_${mode} PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  if(mode STREQUAL "fault")
    target_compile_definitions(lapack_svd_${mode} PRIVATE AER_TEST_INJECT_LAPACK)
  endif()
  add_test(NAME lapack_svd_${mode} COMMAND lapack_svd_${mode})
  set_tests_properties(lapack_svd_${mode} PROPERTIES
    ENVIRONMENT "OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;OMP_NUM_THREADS=1")
endforeach()
