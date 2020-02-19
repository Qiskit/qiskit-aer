
macro(setup_conan)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake")
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)

    set(REQUIREMENTS nlohmann_json/3.7.3 spdlog/1.5.0)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(REQUIREMENTS ${REQUIREMENTS} llvm-openmp/9.0.1)
    endif()

    if(SKBUILD)
        set(REQUIREMENTS ${REQUIREMENTS} muparserx/4.0.8)
    endif()

    if(NOT AER_THRUST_BACKEND STREQUAL "CUDA")
        set(REQUIREMENTS ${REQUIREMENTS} thrust/1.9.5)
        if(AER_THRUST_BACKEND STREQUAL "TBB")
            set(REQUIREMENTS ${REQUIREMENTS} tbb/2020.0)
        endif()
    endif()

    if(NOT BLAS_LIB_PATH)
        set(REQUIREMENTS ${REQUIREMENTS} openblas/0.3.7)
    endif()

    conan_cmake_run(REQUIRES ${REQUIREMENTS} OPTIONS ${CONAN_OPTIONS} BASIC_SETUP CMAKE_TARGETS KEEP_RPATHS BUILD missing)
endmacro()