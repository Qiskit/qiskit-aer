# CMake config file to build the C++ Simulator
#
# For Linux and Mac, we can build both statically or dynamically. The latter is
# the default. If you want to build an static executable, you need to set
# STATIC_LINKING to True, example:
#     out$ cmake -DSTATIC_LINKING=True ..
#
# For Mac, you'll probably need to install static versions of the toolchain in
# order to make a static executable.
# Additionally, OpenMP support is only available in clang from

cmake_minimum_required(VERSION 3.6)
project(qasm_simulator LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Warning: Because of a bug on CMake's FindBLAS or (it's not clear whoes fault is)
# libopenblas.a for Ubuntu (maybe others) we need to copy the file:
# cmake/FindBLAS.cmake.fix-static-linking, to the directory were CMake Modules are
# installed in the system, but with the name: FindBLAS.cmake
option(STATIC_LINKING "Specify if we want statically link the executable (for
						redistribution mainly)" FALSE)
option(BUILD_TESTS "Specify whether we want to build tests or not" FALSE)

include(CTest)
include(compiler_utils)

set(AER_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(AER_SIMULATOR_CPP_MAIN
    "${PROJECT_SOURCE_DIR}/contrib/standalone/qasm_simulator.cpp")
set(AER_SIMULATOR_CPP_EXTERNAL_LIBS
    "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers"
	"${AER_SIMULATOR_CPP_SRC_DIR}/third-party/win64/lib"
	"${AER_SIMULATOR_CPP_SRC_DIR}/third-party/linux-x86_64/lib"
	"${AER_SIMULATOR_CPP_SRC_DIR}/third-party/macos/dylib"
	"${USER_LIB_PATH}")

# Adding support for CCache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Set default build type to Release with Debug Symbols
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: Debug Release"
		FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Target definition
add_executable(qasm_simulator ${AER_SIMULATOR_CPP_MAIN})

# Target properties: C++ program
set_target_properties(qasm_simulator PROPERTIES
	LINKER_LANGUAGE CXX
	CXX_STANDARD 14)

set_target_properties(qasm_simulator PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
	RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# In order to build for MacOSX 10.9 and above with Clang, we need to force the "deployment target" to 10.9
	# and force using libc++ instead of the default for this target: libstdc++ otherwise we could not
	# use C++11/14
	set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "" FORCE)
	enable_cxx_compiler_flag_if_supported("-stdlib=libc++")
endif()

if(STATIC_LINKING)
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(WARNING "Clang on MacOS doesn't support some -static-* flags. Switching to dyn compilation...")
		unset(STATIC_LINKING)
	else()
	    # MacOS compilers don't support -static flag either
	    if(NOT APPLE)
	        enable_cxx_compiler_flag_if_supported("-static")
	    endif()
	    # This is enough to build a semi-static executable on Mac
	    enable_cxx_compiler_flag_if_supported("-static-libgcc")
	    enable_cxx_compiler_flag_if_supported("-static-libstdc++")
	endif()
endif()

if(NOT MSVC)
	enable_cxx_compiler_flag_if_supported("-ffast-math")
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
		# PowerPC builds are not meant to be redistributable, we build them
		# in place, so we can have CPU = native.
		enable_cxx_compiler_flag_if_supported("-mcpu=native")
	endif()
	# Warnings and Errors
	enable_cxx_compiler_flag_if_supported("-pedantic")
	enable_cxx_compiler_flag_if_supported("-Wall")
	enable_cxx_compiler_flag_if_supported("-Wfloat-equal")
	enable_cxx_compiler_flag_if_supported("-Wundef")
	enable_cxx_compiler_flag_if_supported("-Wcast-align")
	enable_cxx_compiler_flag_if_supported("-Wwrite-strings")
	enable_cxx_compiler_flag_if_supported("-Wmissing-declarations")
	enable_cxx_compiler_flag_if_supported("-Wredundant-decls")
	enable_cxx_compiler_flag_if_supported("-Wshadow")
	enable_cxx_compiler_flag_if_supported("-Woverloaded-virtual")
endif()

target_include_directories(qasm_simulator PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR})

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

# Looking for external libraries
find_package(OpenMP QUIET)
# This is a hack for building with Apple's LLVM, which doesn't support OpenMP yet
# so we need to link with an external library: libomp.
# NOTE: CMake >= 3.12.0 doesn't need this hack. It will just find OpenMP in the
# first find_package(OpenMP) call
if(NOT "${OpenMP_FOUND}" OR NOT "${OpenMP_CXX_FOUND}")
	find_program(BREW NAMES brew)
	if(BREW)
		execute_process(COMMAND ${BREW} ls libomp RESULT_VARIABLE BREW_RESULT_CODE OUTPUT_QUIET ERROR_QUIET)
		if(BREW_RESULT_CODE)
			message(STATUS "This program supports OpenMP on Mac through Brew. Please run \"brew install libomp\"")
		else()
			execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
			set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
			set(OpenMP_CXX_LIB_NAMES "omp")
			if(STATIC_LINKING)
				message(STATUS "Using static library for OMP Threads: libomp.a")
				set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.a")
			else()
				set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
			endif()
			include_directories("${BREW_LIBOMP_PREFIX}/include")
			message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")
			set(OPENMP_FOUND TRUE)
		endif()
	else()
		message(STATUS "This program supports OpenMP on Mac through Homebrew, installing Homebrew recommended https://brew.sh")
	endif()
endif()

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	if(APPLE)
		# On Apple and clang, we do need to link against the library
		set(OPENMP_EXTERNAL_LIB "${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY}")
		message("Adding Clang: ${OPENMP_EXTERNAL_LIB}")
	endif()
	message("OpenMP found!")
endif()

set(NLOHMANN_JSON_PATH ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
find_package(nlohmann_json REQUIRED)

find_package(Threads)

if(STATIC_LINKING)
	message(STATUS "Setting BLA_STATIC")
	set(BLA_STATIC TRUE)
endif()

if(APPLE)
	message(STATUS "Looking for Apple BLAS library...")
    set(BLA_VENDOR "Apple")
else()
	message(STATUS "Looking for OpenBLAS library...")
    set(BLA_VENDOR "OpenBLAS")
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
	message(STATUS "OpenBLAS not found. Looking for any other BLAS library...")
	unset(BLA_VENDOR)
	find_package(BLAS REQUIRED)
endif()

message("BLAS: ${BLAS_LIBRARIES}")

# Set dependent libraries
set(AER_LIBRARIES
		${OPENMP_EXTERNAL_LIB}
	    ${BLAS_LIBRARIES}
	    nlohmann_json
	    Threads::Threads)

# Linter
# This will add the linter as part of the compiling build target
include(Linter)

target_link_libraries(qasm_simulator	PRIVATE ${AER_LIBRARIES})

# Tests
if(BUILD_TESTS)
	add_subdirectory(test)
endif()

# Cython build is only enabled if building through scikit-build.
if(SKBUILD)
	add_subdirectory(qiskit/providers/aer/backends/wrappers)
endif()
