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
project(qasm_simulator_cpp LANGUAGES CXX)

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

set(QASM_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(QASM_SIMULATOR_CPP_SRC
    "${QASM_SIMULATOR_CPP_SRC_DIR}/main.cpp")
set(QASM_SIMULATOR_CPP_EXTERNAL_LIBS
    "${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/headers"
	"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/win64/lib"
	"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/linux-x86_64/lib"
	"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/macos/dylib"
	"${USER_LIB_PATH}")

# Target definition
add_executable(qasm_simulator_cpp ${QASM_SIMULATOR_CPP_SRC})

# Target properties: C++ program
set_target_properties(qasm_simulator_cpp PROPERTIES LINKER_LANGUAGE CXX)

# Toolchain options
set_property(TARGET qasm_simulator_cpp PROPERTY CXX_STANDARD 11)

if(STATIC_LINKING)
	# Hack: Seems like enable_cxx_compiler_flag_if_supported() is not properly
	# working on MacOS, when a flag is not supported, it cascades errors
	# to the rest of the flags being tested... and -static compilation on Mac
	# with gcc is failing...
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

target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR})
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/backends)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/engines)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/utilities)
target_include_directories(qasm_simulator_cpp PRIVATE ${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/headers)

# For header only libraries
SET(CMAKE_FIND_LIBRARY_PREFIXES "" ${CMAKE_FIND_LIBRARY_PREFIXES})
SET(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .hpp)

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

find_package(OpenMP QUIET)
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

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Looking for external libraries
# This is a header-only library so we don't want it to link it, but we still
# want CMake to find it.
find_library(LIB_JSON
	NAMES json nlohmann_json.hpp
	PATHS ${QASM_SIMULATOR_CPP_EXTERNAL_LIBS})
if(NOT LIB_JSON)
    message(FATAL_ERROR "JSON library not found!. Please provide with a USER_LIB_PATH to CMake so it can be searched there")
endif()

find_library(BLAS_LIB
	NAMES openblas mkl blas
	PATHS ${QASM_SIMULATOR_CPP_EXTERNAL_LIBS})
if(NOT BLAS_LIB)
	message(FATAL_ERROR "BLAS library not found!. Please provide with a USER_LIB_PATH to CMake so it can be searched there")
endif()

set(LIBRARIES PRIVATE ${BLAS_LIB})

# Linking
target_link_libraries(qasm_simulator_cpp ${LIBRARIES})

set(QASM_SIMULATOR_CPP_OUTPUT_DIR $<TARGET_FILE_DIR:qasm_simulator_cpp>
	CACHE INTERNAL "Output directory for building Qiskit C++ Simulator")

# TODO Windows 32 bits builds are not supported yet
set(ARCH_BITS "64")
if(ARCH32)
	set(ARCH_BITS "32")
endif()

# Copying necessary .dll files to output
# TODO look into "install()" CMake function
if(WIN32) # Windows 64 bits is also true here
	set(QASM_SIMULATOR_CPP_THIRD_PARTY_DLLS
		"${QASM_SIMULATOR_CPP_SRC_DIR}/third-party/win${ARCH_BITS}/dll/openblas.dll"
		CACHE INTERNAL "Third-party C++ Simulator DLLs")
	foreach(dll_file ${QASM_SIMULATOR_CPP_THIRD_PARTY_DLLS})
		add_custom_command(
			TARGET qasm_simulator_cpp
			POST_BUILD
			COMMAND ${CMAKE_COMMAND}
			ARGS -E copy ${dll_file} ${QASM_SIMULATOR_CPP_OUTPUT_DIR}/
		)
		# For 'make clean' target
		get_filename_component(FINAL_FILE ${dll_file} NAME)
		set_property(DIRECTORY APPEND PROPERTY
			ADDITIONAL_MAKE_CLEAN_FILES
				${QASM_SIMULATOR_CPP_OUTPUT_DIR}/${FINAL_FILE})
	endforeach()
endif()

# Tests
# TODO: Enable them when ready
#add_subdirectory(${QASM_SIMULATOR_CPP_DIR}/test)
