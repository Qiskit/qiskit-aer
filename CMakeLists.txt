# CMake config file to build the C++ Simulator
#
# For Linux and Mac, we can build both statically or dynamically. The latter is
# the default. If you want to build an static executable, you need to set
# STATIC_LINKING to True, example:
#     out$ cmake -DSTATIC_LINKING=True ..
#
# For Mac, you'll probably need to install static versions of the toolchain in
# order to make a static executable.
# Additionaly, OpenMP support is only available in clang from
cmake_minimum_required(VERSION 3.11)
project(aer_simulator_cpp LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(STATIC_LINKING "Specify if we want statically link the executable (for 
						redistribution mainly)" FALSE)
option(BUILD_TESTS "Specify whether we want to build tests or not" TRUE)

include(CTest)
include(compiler_utils)

set(AER_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(AER_SIMULATOR_CPP_MAIN
    "${PROJECT_SOURCE_DIR}/legacy-build/aer_simulator.cpp")
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

# Target definition
add_executable(aer_simulator_cpp ${AER_SIMULATOR_CPP_MAIN})

# Target properties: C++ program
set_target_properties(aer_simulator_cpp PROPERTIES 
											LINKER_LANGUAGE CXX
											CXX_STANDARD 14)

if(STATIC_LINKING)
	# Hack: Seems like enable_cxx_compiler_flag_if_supported() is not properly
	# working on MacOS, when a flag is not supported, it cascades errors
	# to the rest of the flags being tested... and -static compilation on Mac
	# with gcc is failing
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	    message(WARNING "Clang on MacOS doesn't support some -static-* flags. Switching to dyn compilation...")
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
	# Compiler flags
	enable_cxx_compiler_flag_if_supported("-O3")
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

target_include_directories(aer_simulator_cpp PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR})

if(STATIC_LINKING)
	set(BLAS_STATIC True)
endif()

# Looking for external libraries

find_package(OpenMP QUIET)
# This is a hack for building with Apple's LLVM, which doesn't support OpenMP yet
# so we need to link with an external library: libomp
if(NOT "${OpenMP_FOUND}" OR NOT "${OpenMP_CXX_FOUND}")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7")
		message(STATUS "AppleClang >= 7.0 detected, adding OpenMP. Disable with -DAPPLE_OMP_AUTOADD=OFF")
		find_program(BREW NAMES brew)
		if(BREW)
			execute_process(COMMAND ${BREW} ls libomp RESULT_VARIABLE BREW_RESULT_CODE OUTPUT_QUIET ERROR_QUIET)
			if(BREW_RESULT_CODE)
				message(STATUS "This program supports OpenMP on Mac through Brew. Please run \"brew install libomp\"")
			else()
				execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
				set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
				set(OpenMP_CXX_LIB_NAMES "omp")
				set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
				include_directories("${BREW_LIBOMP_PREFIX}/include")
				message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")
			endif()
		else()
			message(STATUS "This program supports OpenMP on Mac through Homebrew, installing Homebrew recommmended https://brew.sh")
		endif()
	endif()
endif()

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	message(STATUS "OpenMP found!")
endif()

set(NLOHMANN_JSON_PATH ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
find_package(nlohmann_json REQUIRED)

message(STATUS "Lookgin for OpenBLAS library...")
set(BLA_VENDOR "OpenBLAS")
find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
	message(STATUS "OpenBLAS not found. Lookgin for any other BLAS library...")
	unset(BLA_VENDOR)
	find_package(BLAS REQUIRED)
endif()

set(LIBRARIES ${BLAS_LIBRARIES} nlohmann_json)

# Linking
target_link_libraries(aer_simulator_cpp PRIVATE ${LIBRARIES})
set(AER_SIMULATOR_CPP_OUTPUT_DIR $<TARGET_FILE_DIR:aer_simulator_cpp>
	CACHE INTERNAL "Output directory for building Qiskit C++ Simulator")

if(BUILD_TESTS)
	add_subdirectory(test)
endif()
