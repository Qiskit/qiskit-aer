find_package(Pybind11 REQUIRED)
find_package(NumPy REQUIRED)
# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

function(numpy_pybind11_add_module target_name)
    basic_pybind11_add_module(${target_name} ${ARGN})
    target_include_directories(${target_name} PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} PRIVATE ${NumPy_INCLUDE_DIRS})
    target_link_libraries(${target_name} ${AER_LIBRARIES})
endfunction()

numpy_pybind11_add_module(pulse_utils pulse_utils_bindings.cpp pulse_utils.cpp numeric_integrator.cpp zspmv.cpp)
numpy_pybind11_add_module(test_python_to_cpp test_python_to_cpp.cpp)

install(TARGETS pulse_utils test_python_to_cpp LIBRARY DESTINATION "qiskit/providers/aer/pulse/de_solvers/")
