# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


macro(setup_dependencies)
    # Defines AER_DEPENDENCY_PKG alias which refers to either conan-provided or system libraries.
    if(USE_CONAN)
        include(conan_utils)
        setup_conan()

        # NOTE: this assumes CONAN_PKG is static.
        # Might need to change to import CONAN_PKG if they need changed
        add_library(AER_DEPENDENCY_PKG ALIAS CONAN_PKG)
    else()
        # Use system libraries
        find_package(nlohmann_json 3.1.1 REQUIRED)
        add_library(AER_DEPENDENCY_PKG::nlohmann_json ALIAS nlohmann_json)
        find_package(spdlog 1.5.0 REQUIRED)
        add_library(AER_DEPENDENCY_PKG::spdlog ALIAS spdlog::spdlog)
        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            find_package(llvm-openmp 8.0.1 REQUIRED)
            add_library(AER_DEPENDENCY_PKG::llvm-openmp ALIAS llvm-openmp)
        endif()

        if(SKBUILD)
            find_package(muparserx 4.0.8 REQUIRED)
            add_library(AER_DEPENDENCY_PKG::muparserx ALIAS muparserx)
        endif()

        if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA")
            find_package(thrust 1.9.5 REQUIRED)
            string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
            add_library(AER_DEPENDENCY_PKG::thrust ALIAS thrust)
        endif()

        if(BUILD_TESTS)
            find_package(catch2 2.12.1 REQUIRED)
            add_library(AER_DEPENDENCY_PKG::catch2 ALIAS catch2)
        endif()
    endif()
endmacro()
