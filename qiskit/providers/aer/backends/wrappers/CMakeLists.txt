include(Linter)
find_package(Pybind11 REQUIRED)

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(APPLE OR LINUX)
    string(CONCAT CMAKE_CXX_FLAGS "-fvisibility=hidden")
endif()

# Controllers
basic_pybind11_add_module(controller_wrappers bindings.cc)
target_include_directories(controller_wrappers PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                                               PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(controller_wrappers ${AER_LIBRARIES})
install(TARGETS controller_wrappers LIBRARY DESTINATION qiskit/providers/aer/backends)

# Install redistributable dependencies
if(APPLE)
    install(FILES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/macos/lib/libomp.dylib" DESTINATION qiskit/providers/aer/backends)
endif()
