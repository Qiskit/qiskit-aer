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

    if(SKBUILD)
        set(REQUIREMENTS ${REQUIREMENTS} muparserx/4.0.8)
        list(APPEND AER_CONAN_LIBS muparserx)
        if(NOT MSVC)
            set(CONAN_OPTIONS ${CONAN_OPTIONS} "muparserx:fPIC=True")
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
        conan_cmake_run(REQUIRES ${REQUIREMENTS}
                        OPTIONS ${CONAN_OPTIONS}
                        ENV CONAN_CMAKE_PROGRAM=${CMAKE_COMMAND}
                        BASIC_SETUP
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
                        BUILD missing)
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
