find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_cython_target(qasm_controller_wrapper qasm_controller_wrapper.pyx CXX)
add_library(qasm_controller_wrapper MODULE ${qasm_controller_wrapper})
set_target_properties(qasm_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(qasm_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})

if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
        solved at runtime by the loader")
    set_target_properties(qasm_controller_wrapper PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()

target_link_libraries(qasm_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(qasm_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})

set(SITE_PACKAGES "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
install(TARGETS qasm_controller_wrapper LIBRARY DESTINATION ${SITE_PACKAGES})


add_cython_target(statevector_controller_wrapper statevector_controller_wrapper.pyx CXX)
add_library(statevector_controller_wrapper MODULE ${statevector_controller_wrapper})
set_target_properties(statevector_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(statevector_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})

if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
        solved at runtime by the loader")
    set_target_properties(statevector_controller_wrapper PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()

target_link_libraries(statevector_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(statevector_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})

set(SITE_PACKAGES "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
install(TARGETS statevector_controller_wrapper LIBRARY DESTINATION ${SITE_PACKAGES})


add_cython_target(unitary_controller_wrapper unitary_controller_wrapper.pyx CXX)
add_library(unitary_controller_wrapper MODULE ${unitary_controller_wrapper})
set_target_properties(unitary_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(unitary_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})

if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
        solved at runtime by the loader")
    set_target_properties(unitary_controller_wrapper PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()

target_link_libraries(unitary_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(unitary_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})

set(SITE_PACKAGES "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
install(TARGETS unitary_controller_wrapper LIBRARY DESTINATION ${SITE_PACKAGES})
