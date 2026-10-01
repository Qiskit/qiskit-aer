include(conan)

macro(_rename_conan_lib package)
    add_library(AER_DEPENDENCY_PKG::${package} INTERFACE IMPORTED)
    target_link_libraries(AER_DEPENDENCY_PKG::${package} PUBLIC INTERFACE CONAN_PKG::${package})
endmacro()

macro(setup_conan)

    # Right now every dependency shall be static
    set(CONAN_OPTIONS ${CONAN_OPTIONS} "*:shared=False")

    set(REQUIREMENTS nlohmann_json/3.1.1 spdlog/1.9.2)
    list(APPEND AER_CONAN_LIBS nlohmann_json spdlog)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND AER_CONAN_LIBS llvm-openmp)
        if (DEFINED ENV{AER_CMAKE_OPENMP_BUILD})
            if(SKBUILD)
                set(AER_CONAN_OPTIONS "llvm-openmp:shared=True")
            else()
                set(AER_CONAN_OPTIONS "llvm-openmp:shared=False")
            endif()
            conan_cmake_run(REQUIRES "llvm-openmp/12.0.1"
                            OPTIONS ${AER_CONAN_OPTIONS}
                            ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                            BASIC_SETUP
                            CMAKE_TARGETS
                            KEEP_RPATHS
                            BUILD llvm-openmp*)
        else()
            set(REQUIREMENTS ${REQUIREMENTS} llvm-openmp/12.0.1)
            if(SKBUILD)
                set(CONAN_OPTIONS ${CONAN_OPTIONS} "llvm-openmp:shared=True")
            endif()
        endif()
    endif()

    if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA" AND NOT AER_THRUST_BACKEND STREQUAL "ROCM")
        set(REQUIREMENTS ${REQUIREMENTS} thrust/1.9.5)
        list(APPEND AER_CONAN_LIBS thrust)
        string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
        set(CONAN_OPTIONS ${CONAN_OPTIONS} "thrust:device_system=${THRUST_BACKEND}")
        if(THRUST_BACKEND MATCHES "tbb")
            list(APPEND AER_CONAN_LIBS tbb)
        endif()
    endif()

    if(BUILD_TESTS)
        set(REQUIREMENTS ${REQUIREMENTS} catch2/2.13.6)
        list(APPEND AER_CONAN_LIBS catch2)
    endif()
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        # ARM macOS build - force GCC for dependencies if using ROCM/CUDA
        if(AER_THRUST_BACKEND STREQUAL "ROCM" OR AER_THRUST_BACKEND STREQUAL "CUDA")
            execute_process(COMMAND gcc --version OUTPUT_VARIABLE GCC_VERSION_OUTPUT)
            string(REGEX MATCH "gcc \\(.*\\) ([0-9]+)\\.([0-9]+)" GCC_VERSION_MATCH "${GCC_VERSION_OUTPUT}")
            if(GCC_VERSION_MATCH)
                set(GCC_MAJOR ${CMAKE_MATCH_1})
                message(STATUS "Conan: Using system GCC ${GCC_MAJOR} for building dependencies (ROCm/CUDA build)")
                set(ENV{CONAN_DISABLE_CHECK_COMPILER} 1)
                conan_cmake_run(REQUIRES ${REQUIREMENTS}
                                OPTIONS ${CONAN_OPTIONS}
                                PROFILE_AUTO NONE
                                SETTINGS compiler=gcc
                                SETTINGS compiler.version=${GCC_MAJOR}
                                SETTINGS compiler.libcxx=libstdc++11
                                SETTINGS build_type=Release
                                ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                                BASIC_SETUP SKIP_COMPILER_CHECK
                                CMAKE_TARGETS
                                KEEP_RPATHS
                                ARCH armv8
                                SETTINGS arch_build=armv8
                                BUILD missing)
            else()
                conan_cmake_run(REQUIRES ${REQUIREMENTS}
                                OPTIONS ${CONAN_OPTIONS}
                                ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                                BASIC_SETUP
                                CMAKE_TARGETS
                                KEEP_RPATHS
                                ARCH armv8
                                SETTINGS arch_build=armv8
                                BUILD missing)
            endif()
        else()
            conan_cmake_run(REQUIRES ${REQUIREMENTS}
                            OPTIONS ${CONAN_OPTIONS}
                            ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                            BASIC_SETUP
                            CMAKE_TARGETS
                            KEEP_RPATHS
                            ARCH armv8
                            SETTINGS arch_build=armv8
                            BUILD missing)
        endif()
    else()
        # When using ROCm or CUDA, force Conan to use system GCC for building dependencies
        # This avoids issues where Conan detects ROCm Clang but system uses GCC
        if(AER_THRUST_BACKEND STREQUAL "ROCM" OR AER_THRUST_BACKEND STREQUAL "CUDA")
            # Find system GCC version for Conan
            execute_process(COMMAND gcc --version OUTPUT_VARIABLE GCC_VERSION_OUTPUT)
            string(REGEX MATCH "gcc \\(.*\\) ([0-9]+)\\.([0-9]+)" GCC_VERSION_MATCH "${GCC_VERSION_OUTPUT}")
            if(GCC_VERSION_MATCH)
                set(GCC_MAJOR ${CMAKE_MATCH_1})
                message(STATUS "Conan: Using system GCC ${GCC_MAJOR} for building dependencies (ROCm/CUDA build)")
                # Set environment variable to disable compiler check in conanbuildinfo.cmake
                set(ENV{CONAN_DISABLE_CHECK_COMPILER} 1)
                # Override Conan's auto-detection by explicitly setting compiler to GCC
                # Use PROFILE_AUTO=NONE to prevent auto-detection
                conan_cmake_run(REQUIRES ${REQUIREMENTS}
                                OPTIONS ${CONAN_OPTIONS}
                                PROFILE_AUTO NONE
                                SETTINGS compiler=gcc
                                SETTINGS compiler.version=${GCC_MAJOR}
                                SETTINGS compiler.libcxx=libstdc++11
                                SETTINGS build_type=Release
                                ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                                BASIC_SETUP SKIP_COMPILER_CHECK
                                CMAKE_TARGETS
                                KEEP_RPATHS
                                BUILD missing)
            else()
                # Fallback to normal build if GCC detection fails
                conan_cmake_run(REQUIRES ${REQUIREMENTS}
                                OPTIONS ${CONAN_OPTIONS}
                                ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                                BASIC_SETUP
                                CMAKE_TARGETS
                                KEEP_RPATHS
                                BUILD missing)
            endif()
        else()
            # Normal CPU build - let Conan auto-detect
            conan_cmake_run(REQUIRES ${REQUIREMENTS}
                            OPTIONS ${CONAN_OPTIONS}
                            ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                            BASIC_SETUP
                            CMAKE_TARGETS
                            KEEP_RPATHS
                            BUILD missing)
        endif()
    endif()

    # Headers includes
    if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA" AND NOT AER_THRUST_BACKEND STREQUAL "ROCM")
        set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${CONAN_INCLUDE_DIRS_THRUST})
    endif()

    # Reassign targets from CONAN_PKG to AER_DEPENDENCY_PKG
    foreach(CONAN_LIB ${AER_CONAN_LIBS})
        _rename_conan_lib(${CONAN_LIB})
    endforeach()

    if(APPLE)
        set(OPENMP_FOUND TRUE)
        if(NOT SKBUILD)
            set(AER_LIBRARIES ${AER_LIBRARIES} AER_DEPENDENCY_PKG::llvm-openmp)
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CONAN_CXX_FLAGS_LLVM-OPENMP}")
            set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${CONAN_INCLUDE_DIRS_LLVM-OPENMP})
            set(BACKEND_REDIST_DEPS ${BACKEND_REDIST_DEPS} "${CONAN_LIB_DIRS_LLVM-OPENMP}/libomp.dylib")
        endif()
    endif()
endmacro()
