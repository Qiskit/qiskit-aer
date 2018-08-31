cmake_minimum_required(VERSION 3.6.0)
project(aer_qv_wrapper LANGUAGES CXX)

find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

add_cython_target(aer_qv_wrapper qv_wrapper.pyx)
add_library(aer_qv_wrapper MODULE ${aer_qv_wrapper})
set_target_properties(aer_qv_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(aer_qv_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(aer_qv_wrapper
    ${PYTHON_LIBRARIES}
    ${AER_LIBRARIES})
install(TARGETS aer_qv_wrapper LIBRARY DESTINATION aer_qv_wrapper)