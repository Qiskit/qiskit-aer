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
    add_library(${module} MODULE ${module} ${ARGV1})
    set_target_properties(${module} PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 14)

    # Avoid warnings in cython cpp generated code
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS -Wno-everything)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS -w)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set_source_files_properties(${module}.cxx PROPERTIES COMPILE_FLAGS /w)
    endif()

    if(APPLE)
        set_target_properties(${module} PROPERTIES
            LINK_FLAGS ${AER_LINKER_FLAGS})
    endif()

    # We only need to pass the linter once, as the codebase is the same for
    # all controllers
    # add_linter(target)
    target_include_directories(${module}
        PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
        PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
        PRIVATE ${PYTHON_INCLUDE_DIRS}
        PRIVATE ${NumPy_INCLUDE_DIRS}
        PRIVATE ${CYTHON_USER_INCLUDE_DIRS})

    target_link_libraries(${module}
        ${AER_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${CYTHON_USER_LIB_DIRS})

    python_extension_module(${module}
        FORWARD_DECL_MODULES_VAR fdecl_module_list)

    python_modules_header(modules
        FORWARD_DECL_MODULES_LIST ${fdecl_module_list})

    include_directories(${modules_INCLUDE_DIRS})
    # TODO Where to put the target files
    install(TARGETS ${module} LIBRARY DESTINATION ${CYTHON_INSTALL_DIR})
endfunction()
