include(Linter)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set some general flags
if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
                    solved at runtime by the loader")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup")
    else()
        # -flat_namespace linker flag is needed otherwise dynamic symbol resolution doesn't work as expected with GCC.
        # Symbols with the same name exist in different .so, so the loader just takes the first one it finds,
        # which is usually the one from the first .so loaded.
        # See: Two-Leve namespace symbol resolution
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup -flat_namespace")
    endif()
    unset(PYTHON_LIBRARIES)
endif()


# QASM Controller

add_cython_target(qasm_controller_wrapper qasm_controller_wrapper.pyx CXX)
add_library(qasm_controller_wrapper MODULE ${qasm_controller_wrapper})
set_target_properties(qasm_controller_wrapper PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(qasm_controller_wrapper PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
# We only need to pass the linter once, as the codebase is the same for
# all controllers
add_linter(qasm_controller_wrapper)
target_include_directories(qasm_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(qasm_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(qasm_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS qasm_controller_wrapper LIBRARY DESTINATION qiskit/providers/aer/backends)

# StateVector Controller

add_cython_target(statevector_controller_wrapper statevector_controller_wrapper.pyx CXX)
add_library(statevector_controller_wrapper MODULE ${statevector_controller_wrapper})
set_target_properties(statevector_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(statevector_controller_wrapper PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
target_include_directories(statevector_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(statevector_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(statevector_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS statevector_controller_wrapper LIBRARY DESTINATION qiskit/providers/aer/backends)

# Unitary Controller

add_cython_target(unitary_controller_wrapper unitary_controller_wrapper.pyx CXX)
add_library(unitary_controller_wrapper MODULE ${unitary_controller_wrapper})
set_target_properties(unitary_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(unitary_controller_wrapper PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
target_include_directories(unitary_controller_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(unitary_controller_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(unitary_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS unitary_controller_wrapper LIBRARY DESTINATION qiskit/providers/aer/backends)

# Install redistributable dependencies
if(APPLE)
    install(FILES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/macos/lib/libomp.dylib" DESTINATION qiskit/providers/aer/backends)
endif()
