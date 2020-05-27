include(conan)

macro(setup_conan)

    # Right now every dependency shall be static
    set(CONAN_OPTIONS ${CONAN_OPTIONS} "*:shared=False")

    set(REQUIREMENTS nlohmann_json/3.7.3 spdlog/1.5.0)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(REQUIREMENTS ${REQUIREMENTS} llvm-openmp/8.0.1)
        if(SKBUILD)
            set(CONAN_OPTIONS ${CONAN_OPTIONS} "llvm-openmp:shared=True")
        endif()
    endif()

    if(SKBUILD)
        set(REQUIREMENTS ${REQUIREMENTS} muparserx/4.0.8)
        if(NOT MSVC)
            set(CONAN_OPTIONS ${CONAN_OPTIONS} "muparserx:fPIC=True")
        endif()
    endif()

    if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA")
        set(REQUIREMENTS ${REQUIREMENTS} thrust/1.9.5)
        string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
        set(CONAN_OPTIONS ${CONAN_OPTIONS} "thrust:device_system=${THRUST_BACKEND}")
    endif()

    if(NOT BLAS_LIB_PATH)
        set(REQUIREMENTS ${REQUIREMENTS} openblas/0.3.7)
    endif()

    if(BUILD_TESTS)
        set(REQUIREMENTS ${REQUIREMENTS} catch2/2.12.1)
    endif()

    conan_cmake_run(REQUIRES ${REQUIREMENTS} OPTIONS ${CONAN_OPTIONS} BASIC_SETUP CMAKE_TARGETS KEEP_RPATHS BUILD missing)
endmacro()
