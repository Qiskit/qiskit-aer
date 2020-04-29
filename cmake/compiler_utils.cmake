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
