find_package(Cython REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

# Frame change
add_cython_target(frame_change frame_change.pyx CXX)
add_library(frame_change MODULE ${frame_change})
set_target_properties(frame_change PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(frame_change PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()
target_link_libraries(frame_change
    ${PYTHON_LIBRARIES})

target_include_directories(frame_change
    PRIVATE ${PYTHON_INCLUDE_DIRS})

python_extension_module(frame_change
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})

include_directories(${modules_INCLUDE_DIRS})

install(TARGETS frame_change LIBRARY DESTINATION qiskit/providers/aer/pulse/cython)

# Solver
add_cython_target(solver solver.pyx CXX)
add_library(solver MODULE ${solver})
set_target_properties(solver PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(solver PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()
target_link_libraries(solver
    ${PYTHON_LIBRARIES})

target_include_directories(solver
    PRIVATE ${PYTHON_INCLUDE_DIRS})

python_extension_module(solver
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})

include_directories(${modules_INCLUDE_DIRS})

install(TARGETS solver LIBRARY DESTINATION qiskit/providers/aer/pulse/cython)