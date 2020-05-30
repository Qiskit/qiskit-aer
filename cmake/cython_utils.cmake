find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

# Variables for input user data:
#
# CYTHON_USER_INCLUDE_DIRS:
# - For Cython modules that need to import some header file not in the paths, example:
#   set(CYTHON_USER_INCLUDE_DIRS "/opt/my/include")
# CYTHON_USER_LIB_DIRS:
# - For Cython modules that need to link with external libs, example:
#   set(CYTHON_USER_LIB_DIRS "/opt/my/lib")
# CYTHON_INSTALL_DIR:
# - Where to install the resulting shared libraries
#   set(CYTHON_INSTALL_DIR "/opt/my/lib")


# Set default values
set(CYTHON_USER_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
unset(CYTHON_USER_LIB_DIRS)
set(CYTHON_INSTALL_DIR "qiskit/providers/aer/backends")

function(add_cython_module module)
    add_cython_target(${module} ${module}.pyx CXX)

    # Avoid warnings in cython cpp generated code
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS -Wno-everything)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS -w)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS /w)
    endif()

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
        cuda_add_library(${module} ${lib_type} ${exclude_from_all} ${module} ${ARG_UNPARSED_ARGUMENTS})
        set_source_files_properties(${module} PROPERTIES
            CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    else()
        add_library(${module} ${lib_type} ${exclude_from_all} ${module} ${ARG_UNPARSED_ARGUMENTS})
    endif()


    # We only need to pass the linter once, as the codebase is the same for
    # all controllers
    # add_linter(target)
    target_include_directories(${module} PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR})
    target_include_directories(${module} PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
    target_include_directories(${module} PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_include_directories(${module} PRIVATE ${NumPy_INCLUDE_DIRS})
    target_include_directories(${module} PRIVATE ${CYTHON_USER_INCLUDE_DIRS})

    target_link_libraries(${module} ${AER_LIBRARIES} ${CYTHON_USER_LIB_DIRS})

    if(WIN32 OR CYGWIN)
        # Link against the Python shared library on Windows
        target_link_libraries(${module} ${PYTHON_LIBRARIES})
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
        if(ARG_SHARED)
            set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ON)
        endif()
    endif()

    set_target_properties(${module} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
    set_target_properties(${module} PROPERTIES SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")
    set_target_properties(${module} PROPERTIES
        CXX_STANDARD 14
        LINKER_LANGUAGE CXX)
    # Warning: Do not merge PROPERTIES when one of the variables can be empty, it breaks
    # the rest of the properties so they are not properly added.
    set_target_properties(${module} PROPERTIES LINK_FLAGS ${AER_LINKER_FLAGS})
    set_target_properties(${module} PROPERTIES COMPILE_FLAGS ${AER_COMPILER_FLAGS})
    target_compile_definitions(${module} PRIVATE ${AER_COMPILER_DEFINITIONS})

    python_extension_module(${module}
        FORWARD_DECL_MODULES_VAR fdecl_module_list)

    python_modules_header(modules
        FORWARD_DECL_MODULES_LIST ${fdecl_module_list})

    include_directories(${modules_INCLUDE_DIRS})

    # TODO Where to put the target files
    install(TARGETS ${module} LIBRARY DESTINATION ${CYTHON_INSTALL_DIR})
endfunction()
