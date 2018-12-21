# CMake config file to build the C++ Simulator
#
# For Linux and Mac, we can build both statically or dynamically. The latter is
# the default. If you want to build an static executable, you need to set
# STATIC_LINKING to True, example:
#     out$ cmake -DSTATIC_LINKING=True ..
#
# For Mac, you'll probably need to install static versions of the toolchain in
# order to make a static executable.
# Additionally, OpenMP support is only available in clang from

cmake_minimum_required(VERSION 3.6)

set(CH_SIMULATOR_CPP_MAIN
    "${PROJECT_SOURCE_DIR}/contrib/standalone/ch_simulator.cpp")
# Target definition
add_executable(ch_simulator ${CH_SIMULATOR_CPP_MAIN})

# Target properties: C++ program
set_target_properties(ch_simulator PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)

set_target_properties(ch_simulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)

target_include_directories(ch_simulator PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR})

target_link_libraries(ch_simulator    PRIVATE ${AER_LIBRARIES})
