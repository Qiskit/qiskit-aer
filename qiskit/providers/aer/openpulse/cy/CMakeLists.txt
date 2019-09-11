# Cython OP extensions

include(Linter)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
# find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

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


# Channel value

add_cython_target(channel_value channel_value.pyx CXX)
Python3_add_library(channel_value MODULE ${channel_value})
set_target_properties(channel_value PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(channel_value PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
# We only need to pass the linter once, as the codebase is the same for
# all controllers
# add_linter(channel_value)
target_include_directories(channel_value
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${Python3_INCLUDE_DIRS}
    PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(channel_value
    ${AER_LIBRARIES}
    ${Python3_LIBRARIES})

python_extension_module(channel_value
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
# TODO Where to put the target files
install(TARGETS channel_value LIBRARY DESTINATION qiskit/providers/aer/backends)


# Measure

add_cython_target(measure measure.pyx CXX)
Python3_add_library(measure MODULE ${measure})
set_target_properties(measure PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(measure PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
# We only need to pass the linter once, as the codebase is the same for
# all controllers
# add_linter(measure)
target_include_directories(measure
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${Python3_INCLUDE_DIRS}
    PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(measure
    ${AER_LIBRARIES}
    ${Python3_LIBRARIES})

python_extension_module(measure
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS measure LIBRARY DESTINATION qiskit/providers/aer/backends)


# Memory

add_cython_target(memory memory.pyx CXX)
Python3_add_library(memory MODULE ${memory})
set_target_properties(memory PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(memory PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
# We only need to pass the linter once, as the codebase is the same for
# all controllers
# add_linter(memory)
target_include_directories(memory
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${Python3_INCLUDE_DIRS}
    PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(memory
    ${AER_LIBRARIES}
    ${Python3_LIBRARIES})

python_extension_module(memory
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS memory LIBRARY DESTINATION qiskit/providers/aer/backends)

# Utils

add_cython_target(utils utils.pyx CXX)
Python3_add_library(utils MODULE ${utils})
set_target_properties(utils PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(utils PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
# We only need to pass the linter once, as the codebase is the same for
# all controllers
# add_linter(utils)
target_include_directories(utils
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${Python3_INCLUDE_DIRS}
    PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(utils
    ${AER_LIBRARIES}
    ${Python3_LIBRARIES})

python_extension_module(utils
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS utils LIBRARY DESTINATION qiskit/providers/aer/backends)
