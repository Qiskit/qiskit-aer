find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

message(STATUS "Numpy include dir: ${NumPy_INCLUDE_DIRS}")

# Frame change
add_cython_target(frame_change frame_change.pyx CXX)

add_library(frame_change MODULE ${frame_change})
target_link_libraries(frame_change
    ${PYTHON_LIBRARIES})
set_target_properties(frame_change PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 11)
target_include_directories(frame_change
PRIVATE ${PYTHON_INCLUDE_DIRS})