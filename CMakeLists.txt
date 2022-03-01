# CMake config file to build AER
#
# For Linux and Mac, we can build both statically or dynamically. The latter is
# the default. If you want to build a static executable/library, you need to set
# STATIC_LINKING to True, example:
#     out$ cmake -DSTATIC_LINKING=True ..
#
# For Mac, statically linking only happens with user libraries, system libraries cannot
# be linked statically per Apple's indications.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
file(STRINGS "qiskit/providers/aer/VERSION.txt" VERSION_NUM)

include(CheckLanguage)
project(qasm_simulator VERSION ${VERSION_NUM} LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT DEFINED AER_BLAS_LIB_PATH AND DEFINED ENV{AER_BLAS_LIB_PATH})
	set(AER_BLAS_LIB_PATH $ENV{AER_BLAS_LIB_PATH})
endif()

if(NOT DEFINED AER_THRUST_BACKEND AND DEFINED ENV{AER_THRUST_BACKEND})
	set(AER_THRUST_BACKEND $ENV{AER_THRUST_BACKEND})
endif()

if(AER_THRUST_BACKEND STREQUAL "CUDA")
  include(nvcc_add_compiler_options)
  set(CUDA_FOUND TRUE)
  #include(FindCUDA) # for cuda_select_nvcc_arch_flags, CUDA_FOUND
  include(FindCUDA/select_compute_arch)
  enable_language(CUDA)
else()
  # idiosyncrasy of CMake that it still creates a reference to this
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION "")
endif()


# Warning: Because of a bug on CMake's FindBLAS or (it's not clear who's fault is)
# libopenblas.a for Ubuntu (maybe others) we need to copy the file:
# cmake/FindBLAS.cmake.fix-static-linking, to the directory were CMake Modules are
# installed in the system, but with the name: FindBLAS.cmake
option(STATIC_LINKING "Specify if we want statically link the executable (for
						redistribution mainly)" FALSE)
option(BUILD_TESTS "Specify whether we want to build tests or not" FALSE)

# Allow disabling conan for downstream package managers. Requires all libraries to be present in path
# Default is value of environment variable if defined or ON
if(NOT DEFINED DISABLE_CONAN AND DEFINED ENV{DISABLE_CONAN})
	set(DISABLE_CONAN $ENV{DISABLE_CONAN})
endif()

include(CTest)
include(compiler_utils)
include(Linter)
include(findBLASInSpecificPath)
include(dependency_utils)

# Get version information
get_version(${VERSION_NUM})
configure_file("${PROJECT_SOURCE_DIR}/contrib/standalone/version.hpp.in"
               "${PROJECT_SOURCE_DIR}/contrib/standalone/version.hpp")

set(AER_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(AER_SIMULATOR_CPP_EXTERNAL_LIBS
	"${USER_LIB_PATH}")

set(AER_COMPILER_DEFINITIONS "")

# TODO: We may want to change the prefix path for all the environments
if(WIN32)
	set(CMAKE_PREFIX_PATH "${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${CMAKE_PREFIX_PATH}")
endif()

# Adding support for CCache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Set default build type to Release with Debug Symbols
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: Debug Release"
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	# In order to build for MacOSX 10.9 and above with Clang, we need to
	# force using libc++ instead of the default for this target: libstdc++
	# otherwise we could not use C++11/14
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
else("Windows general compiler flags")
	enable_cxx_compiler_flag_if_supported("/Oi") # Enable intrinsics instead of functions  (faster code)
	enable_cxx_compiler_flag_if_supported("/bigobj")
endif()

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

#
# Looking for external libraries
#
set(BACKEND_REDIST_DEPS "") # List of redistributable dependencies
setup_dependencies()

# If we do not set them with a space CMake fails afterwards if nothing is set for this vars!
set(AER_LINKER_FLAGS " ")
set(AER_COMPILER_FLAGS " ")
if(MSVC)
  set(AER_COMPILER_FLAGS " /bigobj")
endif ()

if(NOT OPENMP_FOUND) # Could already be setup for macos with conan
	message(STATUS "Looking for OpenMP support...")
	find_package(OpenMP QUIET)
	if(OPENMP_FOUND)
		set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
		if(APPLE)
			set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${OpenMP_CXX_INCLUDE_DIRS})
			# On Apple and clang, we do need to link against the library unless we are building
			# the Terra Addon, see issue: https://github.com/Qiskit/qiskit-aer/issues/1
			if(NOT SKBUILD)
				set(AER_LIBRARIES "${AER_LIBRARIES}" "${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY}")
				message(STATUS "Adding Clang: ${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY}")
			else()
				get_filename_component(OPENMP_LIB_TO_COPY ${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY} REALPATH) #Needed to follow symlinks
				set(BACKEND_REDIST_DEPS ${BACKEND_REDIST_DEPS} ${OPENMP_LIB_TO_COPY})
			endif()
		endif()
		message(STATUS "OpenMP found!")
		message(STATUS "OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
		message(STATUS "OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
	else()
		message(STATUS "WARNING: No OpenMP support found!")
	endif()
endif()

if(STATIC_LINKING)
	message(STATUS "Using static linking with Threads...")
	set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
	set(THREADS_PREFER_PTHREAD_FLAG True)
endif()
find_package(Threads)

if(STATIC_LINKING)
	message(STATUS "Setting BLA_STATIC")
	set(BLA_STATIC TRUE)
endif()

if(AER_BLAS_LIB_PATH)
	find_BLAS_in_specific_path(${AER_BLAS_LIB_PATH})
else()
	if(APPLE)
		message(STATUS  "Looking for Apple BLAS & Lapack library...")
		set(BLA_VENDOR "Apple")
	else()
		message(STATUS "Looking for OpenBLAS library...")
		set(BLA_VENDOR "OpenBLAS")
	endif()
	if(WIN32)
		message(STATUS "Uncompressing OpenBLAS static library...")
		set(WIN_ARCH "win64" )
		if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4") # Checking if win32 build
			set(WIN_ARCH "win32")
		endif()
		execute_process(COMMAND ${CMAKE_COMMAND} -E tar "xvfj" "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/openblas.7z" WORKING_DIRECTORY  "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/")
		set(BACKEND_REDIST_DEPS ${BACKEND_REDIST_DEPS} "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/libopenblas.dll")
		set(BLAS_LIBRARIES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/libopenblas.dll.a") # Seems CMake is unable to find it on its own
		set(BLAS_FOUND True)
	else()
		find_package(BLAS QUIET)
	endif()
	if(NOT BLAS_FOUND)
		message(STATUS "OpenBLAS not found. Looking for any other BLAS & Lapack libraries...")
		unset(BLA_VENDOR)
		find_package(BLAS REQUIRED)
		find_package(LAPACK REQUIRED)
		set(BLAS_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")
	endif()
endif()

message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
    if(APPLE OR UNIX)
        if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(SIMD_FLAGS_LIST "-mfma;-mavx2")
	        enable_cxx_compiler_flag_if_supported("-mpopcnt")
        endif()
	elseif(MSVC)
		set(SIMD_FLAGS_LIST "/arch:AVX2")
	endif()
endif()

set(AER_THRUST_SUPPORTED TRUE)
if(AER_THRUST_SUPPORTED)
	if(AER_THRUST_BACKEND STREQUAL "CUDA")
		message(STATUS "Thrust library: Looking for CUDA backend...")
		find_package(CUDA REQUIRED)
		message(STATUS "Thrust library: CUDA found!")
		if(NOT DEFINED AER_CUDA_ARCH)
			if(DEFINED ENV{AER_CUDA_ARCH})
				set(AER_CUDA_ARCH $ENV{AER_CUDA_ARCH})
			else()
				set(AER_CUDA_ARCH "Auto")
			endif()
		endif()
		cuda_select_nvcc_arch_flags(AER_CUDA_ARCH_FLAGS ${AER_CUDA_ARCH})

		string(REPLACE ";" " "  AER_CUDA_ARCH_FLAGS_EXPAND "${AER_CUDA_ARCH_FLAGS}")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${AER_CUDA_ARCH_FLAGS_EXPAND} -DAER_THRUST_CUDA -I${AER_SIMULATOR_CPP_SRC_DIR} -isystem ${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers -use_fast_math --expt-extended-lambda")

		set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
		set(THRUST_DEPENDENT_LIBS "")
	elseif(AER_THRUST_BACKEND STREQUAL "TBB")
		message(STATUS "TBB Support found!")
		set(THRUST_DEPENDENT_LIBS AER_DEPENDENCY_PKG::tbb)
		set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_CPU=TRUE)
	elseif(AER_THRUST_BACKEND STREQUAL "OMP")
		message(STATUS "Thrust library: Setting OMP backend")
		if(NOT OPENMP_FOUND)
			message(FATAL_ERROR "There's no OMP support. We cannot set Thrust backend to OMP!!")
		endif()
		set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_CPU=TRUE)
		# We don't need to add OMP because it's already an AER dependency
		set(THRUST_DEPENDENT_LIBS "")
	else()
		message(STATUS "No Thrust supported backend")
		set(AER_THRUST_SUPPORTED FALSE)
	endif()

	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
	endif()
endif()

if(AER_THRUST_SUPPORTED)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_SUPPORTED=TRUE)
else()
	message(STATUS "No Thrust support enabled")
endif()

if(AER_DEBUG)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_DEBUG)
endif()

if(TEST_JSON)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} TEST_JSON)
endif()

if(AER_MPI)
	find_package(MPI REQUIRED)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_MPI)
	set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${MPI_CXX_INCLUDE_PATH})
	set(MPI_DEPENDANT_LIBS ${MPI_CXX_LIBRARIES})
	if(AER_DISABLE_GDR)
	  set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_DISABLE_GDR)
	endif()
else()
	set(MPI_DEPENDANT_LIBS "")
endif()

# Set dependent libraries
set(AER_LIBRARIES
	${AER_LIBRARIES}
	${BLAS_LIBRARIES}
	AER_DEPENDENCY_PKG::nlohmann_json
	AER_DEPENDENCY_PKG::spdlog
	Threads::Threads
	${CMAKE_DL_LIBS}
	${THRUST_DEPENDANT_LIBS}
	${MPI_DEPENDANT_LIBS})

set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} ${CONAN_DEFINES})
if(SKBUILD) # Terra Addon build
	set(AER_LIBRARIES ${AER_LIBRARIES} AER_DEPENDENCY_PKG::muparserx)
	add_subdirectory(qiskit/providers/aer/backends/wrappers)
	add_subdirectory(src/open_pulse)
else() # Standalone build

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
      if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
          # We build SIMD filed separately, because they will be reached only if the
          # machine running the code has SIMD support
          set(SIMD_SOURCE_FILE "${PROJECT_SOURCE_DIR}/src/simulators/statevector/qv_avx2.cpp")
      endif()
  endif()

	set(AER_SIMULATOR_SOURCES "${PROJECT_SOURCE_DIR}/contrib/standalone/qasm_simulator.cpp")
	set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
	if(CUDA_FOUND AND AER_THRUST_BACKEND STREQUAL "CUDA")
		set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
		set_source_files_properties(${AER_SIMULATOR_SOURCES} PROPERTIES LANGUAGE CUDA)
		set_source_files_properties(${AER_SIMULATOR_SOURCES} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")
		if(DEFINED SIMD_FLAGS_LIST)
			nvcc_add_compiler_options_list("${SIMD_FLAGS_LIST}" SIMD_FLAGS)
			set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS} ${SIMD_FLAGS}")
		endif()
		add_executable(qasm_simulator ${AER_SIMULATOR_SOURCES} ${SIMD_SOURCE_FILE})
		target_link_libraries(qasm_simulator ${AER_LIBRARIES})
		string(STRIP ${AER_COMPILER_FLAGS} AER_COMPILER_FLAGS_STRIPPED)
                nvcc_add_compiler_options(${AER_COMPILER_FLAGS_STRIPPED} AER_COMPILER_FLAGS_OUT)

		set_target_properties(qasm_simulator PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14
			COMPILE_FLAGS ${AER_COMPILER_FLAGS_OUT}
			LINK_FLAGS ${AER_LINKER_FLAGS}
			RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
			RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)
	else()
  		string(REPLACE ";" " " SIMD_FLAGS "${SIMD_FLAGS_LIST}")
		set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
		add_executable(qasm_simulator ${AER_SIMULATOR_SOURCES} ${SIMD_SOURCE_FILE})
		target_link_libraries(qasm_simulator PRIVATE ${AER_LIBRARIES})
		set_target_properties(qasm_simulator PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14
			COMPILE_FLAGS ${AER_COMPILER_FLAGS}
			LINK_FLAGS ${AER_LINKER_FLAGS}
			RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
			RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)
	endif()
	target_include_directories(qasm_simulator
		PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
		PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
	target_compile_definitions(qasm_simulator
		PRIVATE ${AER_COMPILER_DEFINITIONS})
	if(WIN32 AND NOT AER_BLAS_LIB_PATH)
		add_custom_command(TARGET qasm_simulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${BACKEND_REDIST_DEPS}
				$<TARGET_FILE_DIR:qasm_simulator>)
		install(FILES ${BACKEND_REDIST_DEPS} DESTINATION bin)
	endif()

	install(TARGETS qasm_simulator DESTINATION bin)

	# Linter
	# This will add the linter as part of the compiling build target
	#add_linter(qasm_simulator)
endif()

# Tests
if(BUILD_TESTS)
	add_subdirectory(test)
endif()
