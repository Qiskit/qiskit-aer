set(_find_pybind_includes_command "
import sys
import pybind11
sys.stdout.write(pybind11.get_include())
")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "${_find_pybind_includes_command}"
                OUTPUT_VARIABLE _py_output
                RESULT_VARIABLE _py_result)
if(_py_result EQUAL "0")
    message(STATUS "PYCOMM RAW: ${_py_output}")
    set(PYBIND_INCLUDE_DIRS "${_py_output}")
    message(STATUS "PYBIND INCLUDES FOUND: ${PYBIND_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "COULD NOT FIND PYBIND!")
endif()

find_package(PythonExtensions REQUIRED)
find_package(PythonLibs REQUIRED)

function(basic_pybind11_add_module target_name)
    set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS SYSTEM THIN_LTO)
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

    if(ARG_MODULE AND ARG_SHARED)
        message(FATAL_ERROR "Can't be both MODULE and SHARED")
    elseif(ARG_SHARED)
        set(lib_type SHARED)
    else()
        set(lib_type MODULE)
    endif()

    if(ARG_EXCLUDE_FROM_ALL)
        set(exclude_from_all EXCLUDE_FROM_ALL)
    endif()

    if(CUDA_FOUND)
        cuda_add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})
    else()
        add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})
    endif()

    # This sets various properties (python include dirs) and links to python libs
    target_include_directories(${target_name} PRIVATE ${PYTHON_INCLUDE_DIRS})
    set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
    set_target_properties(${target_name} PROPERTIES SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")

    target_include_directories(${target_name} PRIVATE ${PYBIND_INCLUDE_DIRS})
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
    set_target_properties(${target_name} PROPERTIES CXX_STANDARD 14)

    if(WIN32 OR CYGWIN)
        # Link against the Python shared library on Windows
        target_link_libraries(${target_name} ${PYTHON_LIBRARIES})
    elseif(APPLE)
        # It's quite common to have multiple copies of the same Python version
        # installed on one's system. E.g.: one copy from the OS and another copy
        # that's statically linked into an application like Blender or Maya.
        # If we link our plugin library against the OS Python here and import it
        # into Blender or Maya later on, this will cause segfaults when multiple
        # conflicting Python instances are active at the same time (even when they
        # are of the same version).
 
        # Windows is not affected by this issue since it handles DLL imports
        # differently. The solution for Linux and Mac OS is simple: we just don't
        # link against the Python library. The resulting shared library will have
        # missing symbols, but that's perfectly fine -- they will be resolved at
        # import time.
        # Set some general flags
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup")
        else()
            # -flat_namespace linker flag is needed otherwise dynamic symbol resolution doesn't work as expected with GCC.
            # Symbols with the same name exist in different .so, so the loader just takes the first one it finds,
            # which is usually the one from the first .so loaded.
            # See: Two-Leve namespace symbol resolution
            set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} -undefined dynamic_lookup -flat_namespace")
        endif()
        set_target_properties(${target_name} PROPERTIES
            LINK_FLAGS ${AER_LINKER_FLAGS}
            COMPILE_FLAGS ${AER_COMPILE_FLAGS}
            MACOSX_RPATH ON)
    endif()
endfunction()
