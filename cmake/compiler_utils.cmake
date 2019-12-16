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


function(get_muparserx_source_code)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        # if we have cloned the sources, muparserx is a submodule, so we need
        # to initialize it
        if(EXISTS "${PROJECT_SOURCE_DIR}/.gitmodules")
            # Update submodules as needed
            message(STATUS "Submodule update")
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE GIT_SUBMOD_RESULT)
            if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
            endif()
        endif()
    # Not comming from git, so probably: pip install https://...zip or similar.
    # This time, we want to clone muparserx and change the latests stable release
    elseif(GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} clone --branch v4.0.8 https://github.com/beltoforion/muparserx.git ${PROJECT_SOURCE_DIR}/src/third-party/headers/muparserx
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git clone failed with ${GIT_SUBMOD_RESULT},\
                    please checkout muparserx manually from https://github.com/beltoforion/muparserx.git and \
                    checkout latest stable relase")
        endif()
    # TODO: If there's no git, we have to get muparserx using other method (curl)
    endif()

    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/src/third-party/headers/muparserx/CMakeLists.txt")
        message(FATAL_ERROR "MuparserX doesn't exist! GIT_SUBMODULE was turned off or download failed.\
                Please download MuparserX library from https://github.com/beltoforion/muparserx.git \
                and checkout latest stable release")
    endif()
endfunction()

function(get_clang_version version_str)
    string(REPLACE "." ";" VERSION_LIST ${version_str})
    list(GET VERSION_LIST 0 TMP_MAJOR_VERSION)
    list(GET VERSION_LIST 1 TMP_MINOR_VERSION)
    list(GET VERSION_LIST 2 TMP_PATCH_VERSION)
    set(CLANG_MAJOR_VERSION ${TMP_MAJOR_VERSION} PARENT_SCOPE)
    set(CLANG_MINOR_VERSION ${TMP_MINOR_VERSION} PARENT_SCOPE)
    set(CLANG_PATCH_VERSION ${TMP_PATCH_VERSION} PARENT_SCOPE)
endfunction()
