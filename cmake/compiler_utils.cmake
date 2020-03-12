# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

include(CheckCXXCompilerFlag)
function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()


function(get_version version_str)
    string(REPLACE "." ";" VERSION_LIST ${version_str})
    list(GET VERSION_LIST 0 TMP_MAJOR_VERSION)
    list(GET VERSION_LIST 1 TMP_MINOR_VERSION)
    list(GET VERSION_LIST 2 TMP_PATCH_VERSION)
    set(MAJOR_VERSION ${TMP_MAJOR_VERSION} PARENT_SCOPE)
    set(MINOR_VERSION ${TMP_MINOR_VERSION} PARENT_SCOPE)
    set(PATCH_VERSION ${TMP_PATCH_VERSION} PARENT_SCOPE)
endfunction()

function(is_dir_empty dir)
    file(GLOB RESULT ${dir}/*)
    list(LENGTH RESULT num_files)
    if(num_files EQUAL 0)
        set(dir_is_empty TRUE PARENT_SCOPE)
    else()
        set(dir_is_empty FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_get_library_source_code library repo_url version proof_of_existance)
    is_dir_empty(${PROJECT_SOURCE_DIR}/src/third-party/headers/${library})
    if(dir_is_empty)
        find_package(Git QUIET)
        if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
            # if we have cloned the sources, the library is a submodule, so we need
            # to initialize it
            if(EXISTS "${PROJECT_SOURCE_DIR}/.gitmodules")
                # Update library submodule as needed
                message(STATUS "Submodule update")
                execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive src/third-party/headers/${library}
                                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                                RESULT_VARIABLE GIT_SUBMOD_RESULT)
                if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                    message(FATAL_ERROR "git submodule update --init --recursive src/third-party/headers/${library} \
                            failed with ${GIT_SUBMOD_RESULT}, please checkout submodule ${library}")
                endif()
            else()
                message(WARNING "Git checkout but ${PROJECT_SOURCE_DIR}/.gitmodules file not found!")
            endif()
        # Not comming from git, so probably: pip install https://...zip or similar.
        # This time, we want to clone the library and change the latests stable release
        elseif(GIT_FOUND)
            execute_process(COMMAND ${GIT_EXECUTABLE} clone --branch ${version} ${repo_url} ${PROJECT_SOURCE_DIR}/src/third-party/headers/${library}
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE GIT_SUBMOD_RESULT)
            if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                message(FATAL_ERROR "git clone failed with ${GIT_SUBMOD_RESULT},\
                        please checkout ${library} manually from ${repo_url} and \
                        checkout latest stable relase")
            endif()
        # TODO: If there's no git, we have to get the library using other method (curl)
        endif()

        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/third-party/headers/${proof_of_existance}")
            message(FATAL_ERROR "${library} doesn't exist! GIT_SUBMODULE was turned off or download failed.\
                    Please download ${library} library from ${repo_url} \
                    and checkout latest stable release")
        endif()
    else()
        message(STATUS "${library} library source code already exists")
    endif()
    # TODO: We should be adding included directories to targets, and not globally
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/third-party/headers/${library})
endfunction()

function(get_muparserx_source_code)
    _get_library_source_code(
        muparserx
        https://github.com/beltoforion/muparserx.git
        v4.0.8
        muparserx/CMakeLists.txt
    )
endfunction()

function(get_thrust_source_code)
    _get_library_source_code(
        thrust
        https://github.com/thrust/thrust.git
        1.9.5
        thrust/thrust/version.h
    )
endfunction()

function(check_compiler_cpp11_abi)
    # This is needed in case the compiler doesn't work with the new C++11 ABI,
    # is the case of GCC in RHEL6 and RHEL7
    # https://bugzilla.redhat.com/show_bug.cgi?id=1546704
    # Consider also if -D_GLIBCXX_USE_CXX11_ABI has been passed as flag
    string(REGEX MATCH "-D_GLIBCXX_USE_CXX11_ABI=[(A-z)|(a-z)|(0-9)]+" CUSTOM_PREP_FLAGS ${CMAKE_CXX_FLAGS})
    # Preprocessor run to check if CXX11_ABI is set
    execute_process(COMMAND echo "#include <string>" COMMAND ${CMAKE_CXX_COMPILER} ${CUSTOM_PREP_FLAGS} -x c++ -E -dM -  COMMAND fgrep _GLIBCXX_USE_CXX11_ABI OUTPUT_VARIABLE CXX11_ABI_OUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "#define _GLIBCXX_USE_CXX11_ABI " "" CXX11_ABI "${CXX11_ABI_OUT}")
    set(CXX11_ABI ${CXX11_ABI} PARENT_SCOPE)
endfunction()

function(uncompress_muparsersx_lib)
    if(MSVC)
        set(PLATFORM "win64")
    elseif(APPLE)
        set(PLATFORM "macos")
    elseif(UNIX)
        check_compiler_cpp11_abi()
        if(CXX11_ABI EQUAL "0")
            set(MUPARSER_ABI_PREFIX oldabi_)
        endif()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
            set(MUPARSER_ARCH_POSTFIX ".ppc64le")
        endif()
        # Raspberry Pi (4)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
            set(MUPARSER_ARCH_POSTFIX ".armv7l")
        endif()
        set(PLATFORM "linux")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} -E tar "xvfj" "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${PLATFORM}/lib/${MUPARSER_ABI_PREFIX}muparserx${MUPARSER_ARCH_POSTFIX}.7z"
            WORKING_DIRECTORY  "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${PLATFORM}/lib/")
    set(MUPARSERX_LIB_PATH "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${PLATFORM}/lib" PARENT_SCOPE)
endfunction()

function(add_muparserx_lib)
    message(STATUS "Looking for muparserx library...")

    # Look first for already installed muparserx
    find_package(muparserx QUIET)

    if(muparserx_FOUND)
        message(STATUS "Muparserx library version '${muparserx_VERSION}' found")
        set(AER_LIBRARIES ${AER_LIBRARIES} PkgConfig::muparserx PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Muparserx library not found. Uncompressing muparserx static library...")

    uncompress_muparsersx_lib()

    find_library(MUPARSERX_LIB NAMES libmuparserx.a muparserx HINTS ${MUPARSERX_LIB_PATH})
    if(${MUPARSERX_LIB} MATCHES "MUPARSERX_LIB-NOTFOUND")
        message(FATAL_ERROR "No muparserx library found")
    endif()
    message(STATUS "Muparserx library found: ${MUPARSERX_LIB}")
    get_muparserx_source_code()
    # I keep this disabled on purpose, just in case I need to debug muparserx related problems
    # file(GLOB MUPARSERX_SOURCES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers/muparserx/parser/*.cpp")

    set(AER_LIBRARIES ${AER_LIBRARIES} ${MUPARSERX_LIB} PARENT_SCOPE)
endfunction()
