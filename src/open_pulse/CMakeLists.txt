include(Linter)
find_package(Pybind11 REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

basic_pybind11_add_module(numeric_integrator_wrapper numeric_integrator.cpp)

target_include_directories(numeric_integrator_wrapper PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
        PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} PRIVATE ${PYTHON_INCLUDE_DIRS} PRIVATE ${NumPy_INCLUDE_DIRS})
target_link_libraries(numeric_integrator_wrapper ${AER_LIBRARIES})
target_compile_definitions(numeric_integrator_wrapper PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS numeric_integrator_wrapper LIBRARY DESTINATION "qiskit/providers/aer/pulse/de_solvers/")
python_extension_module(${module}
        FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
        FORWARD_DECL_MODULES_LIST ${fdecl_module_list})

basic_pybind11_add_module(pulse_utils pulse_utils.cpp zspmv.cpp)
target_link_libraries(pulse_utils ${AER_LIBRARIES})
target_compile_definitions(pulse_utils PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS pulse_utils LIBRARY DESTINATION "qiskit/providers/aer/pulse/de_solvers/")

basic_pybind11_add_module(test_python_to_cpp test_python_to_cpp.cpp)
target_include_directories(test_python_to_cpp PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
        PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} PRIVATE ${PYTHON_INCLUDE_DIRS} PRIVATE ${NumPy_INCLUDE_DIRS})
target_link_libraries(test_python_to_cpp ${AER_LIBRARIES})
target_compile_definitions(test_python_to_cpp PRIVATE ${AER_COMPILER_DEFINITIONS})
install(TARGETS test_python_to_cpp LIBRARY DESTINATION "qiskit/providers/aer/pulse/de_solvers/")
