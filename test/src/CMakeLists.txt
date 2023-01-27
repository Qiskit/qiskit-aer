add_test_executable(test_linalg "test_linalg.cpp")
add_custom_target(build_tests test_linalg)

add_subdirectory(simulators)
