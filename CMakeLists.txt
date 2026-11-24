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
file(STRINGS "qiskit_aer/VERSION.txt" VERSION_NUM)

# For ROCm builds we need to make sure the CXX and HIP compilers match and are clang.
# We should do this before the project() call to make sure the compiler options are
# properly assessed.
if(AER_THRUST_BACKEND STREQUAL "ROCM")

  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
      
  if(NOT DEFINED CMAKE_HIP_COMPILER)
    if(DEFINED ENV{CMAKE_HIP_COMPILER})
      set(CMAKE_HIP_COMPILER "$ENV{CMAKE_HIP_COMPILER}")
    else()
      set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
    endif()
  endif()
  
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(DEFINED ENV{CMAKE_CXX_COMPILER})
      set(CMAKE_CXX_COMPILER "$ENV{CMAKE_CXX_COMPILER}")
    else()
      set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}")
    endif()
  endif()
endif()

# Add CUDA to the project if needed.
set(EXTRA_LANGUAGES "")
if(AER_THRUST_BACKEND STREQUAL "CUDA")
  list(APPEND EXTRA_LANGUAGES CUDA)
endif()

include(CheckLanguage)
project(qasm_simulator VERSION ${VERSION_NUM} LANGUAGES CXX C ${EXTRA_LANGUAGES} )


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
option(USE_BUNDLED_BLAS_WIN "Use the bundled openblas library on Windows when
							not using AER_BLAS_LIB_PATH" TRUE)

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
		if(NOT BLA_VENDOR)
		  set(BLA_VENDOR "OpenBLAS")
		endif()
	endif()
	if(WIN32 AND USE_BUNDLED_BLAS_WIN)
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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    if(APPLE OR UNIX)
        set(SIMD_FLAGS_LIST "-mfma;-mavx2")
        enable_cxx_compiler_flag_if_supported("-mpopcnt")
	elseif(MSVC)
		set(SIMD_FLAGS_LIST "/arch:AVX2")
	endif()
endif()

set(AER_THRUST_SUPPORTED TRUE)
if(AER_THRUST_SUPPORTED)
	if(AER_THRUST_BACKEND STREQUAL "CUDA")
		message(STATUS "Thrust library: Looking for CUDA backend...")
		# FindCUDAToolkit (CMake >= 3.17) replaces the FindCUDA module, which is
		# removed by policy CMP0146 in recent CMake versions. The CUDA language is
		# already enabled by the project() call above.
		if(CMAKE_VERSION VERSION_LESS 3.17)
			message(FATAL_ERROR "Building the CUDA backend requires CMake 3.17 or newer (found ${CMAKE_VERSION}).")
		endif()
		find_package(CUDAToolkit REQUIRED)
		message(STATUS "Thrust library: CUDA found!")
		if(NOT DEFINED AER_CUDA_ARCH)
			if(DEFINED ENV{AER_CUDA_ARCH})
				set(AER_CUDA_ARCH $ENV{AER_CUDA_ARCH})
			elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
				# Honour an explicit -DCMAKE_CUDA_ARCHITECTURES=80;90;120 by translating
				# it to the "N.N" notation used by AER_CUDA_ARCH ("120" -> "12.0").
				# Values CMake accepts but we cannot translate (native, all, 80-real...)
				# are ignored and architectures are detected as usual.
				set(AER_CUDA_ARCH "")
				set(_aer_cuda_arch_translatable TRUE)
				foreach(_cc ${CMAKE_CUDA_ARCHITECTURES})
					if(_cc MATCHES "^([0-9]+)([0-9])$")
						list(APPEND AER_CUDA_ARCH "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
					else()
						set(_aer_cuda_arch_translatable FALSE)
					endif()
				endforeach()
				if(NOT _aer_cuda_arch_translatable)
					set(AER_CUDA_ARCH "Auto")
				endif()
			else()
				set(AER_CUDA_ARCH "Auto")
			endif()
		endif()
		# Resolve the Auto, Common and All keywords. The architectures supported by
		# the installed nvcc are queried from nvcc itself, because the lists CMake
		# keeps for each toolkit version are only up to date for the CMake releases
		# that know the toolkit (older CMake offers Maxwell/Pascal/Volta targets that
		# CUDA 13 rejects, and misses Turing).
		#   Auto   -> GPUs present in this machine (CMAKE_CUDA_ARCHITECTURES_NATIVE,
		#             CMake >= 3.24) that nvcc supports, falling back to Common
		#   Common -> the lowest supported architecture of each major version, with
		#             PTX for the newest
		#   All    -> every architecture supported by nvcc, with PTX for the newest
		# The results are converted to the "N.N" notation of AER_CUDA_ARCH
		# ("120" -> "12.0").
		if(AER_CUDA_ARCH STREQUAL "Auto" OR AER_CUDA_ARCH STREQUAL "Common" OR AER_CUDA_ARCH STREQUAL "All")
			set(_aer_cuda_arch_keyword "${AER_CUDA_ARCH}")
			execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-code
				OUTPUT_VARIABLE _aer_nvcc_sm RESULT_VARIABLE _aer_nvcc_result
				OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
			string(REGEX MATCHALL "sm_[0-9]+" _aer_nvcc_sm "${_aer_nvcc_sm}")
			string(REPLACE "sm_" "" _aer_supported "${_aer_nvcc_sm}")
			list(REMOVE_DUPLICATES _aer_supported)
			if(NOT _aer_supported)
				message(FATAL_ERROR "Cannot list the architectures supported by ${CMAKE_CUDA_COMPILER}. "
					"Set AER_CUDA_ARCH to an explicit list such as \"7.5 8.0 9.0\".")
			endif()

			set(_aer_cuda_arch_sm "")
			if(_aer_cuda_arch_keyword STREQUAL "Auto")
				# Some CMake versions store the detection error message in this
				# variable when no GPU is found, so only keep architecture values.
				foreach(_cc ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
					if(_cc MATCHES "^([0-9]+)(-real|-virtual)?$")
						set(_aer_native_sm "${CMAKE_MATCH_1}")
						if(_aer_native_sm IN_LIST _aer_supported)
							list(APPEND _aer_cuda_arch_sm ${_aer_native_sm})
						else()
							message(STATUS "Ignoring GPU architecture ${_aer_native_sm}, not supported by this CUDA toolkit")
						endif()
					endif()
				endforeach()
				if(NOT _aer_cuda_arch_sm)
					message(STATUS "No supported CUDA GPU detected, building for the common architectures")
					set(_aer_cuda_arch_keyword "Common")
				endif()
			endif()
			if(_aer_cuda_arch_keyword STREQUAL "Common")
				set(_aer_cuda_arch_majors "")
				foreach(_cc ${_aer_supported})
					# A cubin runs on later minor versions of the same major version,
					# except that Volta (7.0, 7.2) cubins do not run on Turing (7.5).
					string(REGEX REPLACE "[0-9]$" "" _major "${_cc}")
					if(_cc EQUAL 75)
						set(_major 75)
					endif()
					list(APPEND _aer_cuda_arch_majors ${_major})
					if(NOT DEFINED _aer_cuda_arch_lowest_${_major} OR _cc LESS _aer_cuda_arch_lowest_${_major})
						set(_aer_cuda_arch_lowest_${_major} ${_cc})
					endif()
				endforeach()
				list(REMOVE_DUPLICATES _aer_cuda_arch_majors)
				foreach(_major ${_aer_cuda_arch_majors})
					list(APPEND _aer_cuda_arch_sm ${_aer_cuda_arch_lowest_${_major}})
				endforeach()
			elseif(_aer_cuda_arch_keyword STREQUAL "All")
				set(_aer_cuda_arch_sm ${_aer_supported})
			endif()

			set(_aer_cuda_arch_newest 0)
			foreach(_cc ${_aer_cuda_arch_sm})
				if(_cc GREATER _aer_cuda_arch_newest)
					set(_aer_cuda_arch_newest ${_cc})
				endif()
			endforeach()
			set(AER_CUDA_ARCH "")
			foreach(_cc ${_aer_cuda_arch_sm})
				if(_cc MATCHES "^([0-9]+)([0-9])$")
					set(_arch "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
					if(NOT _aer_cuda_arch_keyword STREQUAL "Auto" AND _cc EQUAL _aer_cuda_arch_newest)
						set(_arch "${_arch}+PTX")
					endif()
					list(APPEND AER_CUDA_ARCH "${_arch}")
				endif()
			endforeach()
			message(STATUS "AER_CUDA_ARCH = ${AER_CUDA_ARCH}")
		endif()

		# Build the nvcc flags from an explicit numeric list such as
		# "7.5 8.0 9.0+PTX 10.0".
		string(REGEX REPLACE "[ \t]+" ";" AER_CUDA_ARCH_LIST "${AER_CUDA_ARCH}")
		set(AER_CUDA_ARCH_IS_NUMERIC TRUE)
		foreach(_arch ${AER_CUDA_ARCH_LIST})
			if(NOT _arch MATCHES "^[0-9]+\\.[0-9]+(\\+PTX)?$")
				set(AER_CUDA_ARCH_IS_NUMERIC FALSE)
			endif()
		endforeach()
		if(AER_CUDA_ARCH_IS_NUMERIC)
			set(AER_CUDA_ARCH_FLAGS "")
			foreach(_arch ${AER_CUDA_ARCH_LIST})
				string(REGEX REPLACE "\\+PTX$" "" _arch_num "${_arch}")
				string(REPLACE "." "" _arch_sm "${_arch_num}")
				list(APPEND AER_CUDA_ARCH_FLAGS -gencode arch=compute_${_arch_sm},code=sm_${_arch_sm})
				if(_arch MATCHES "\\+PTX$")
					list(APPEND AER_CUDA_ARCH_FLAGS -gencode arch=compute_${_arch_sm},code=compute_${_arch_sm})
				endif()
			endforeach()
		else()
			message(FATAL_ERROR "Unsupported AER_CUDA_ARCH value \"${AER_CUDA_ARCH}\". "
				"Use Auto, Common, All or a list of compute capabilities such as \"7.5 8.0 9.0+PTX\".")
		endif()

		string(REPLACE ";" " "  AER_CUDA_ARCH_FLAGS_EXPAND "${AER_CUDA_ARCH_FLAGS}")

		string(REGEX MATCHALL "sm_[0-9]+" CUDA_SM "${AER_CUDA_ARCH_FLAGS}")
		string(REPLACE "sm_" ""  AER_CUDA_ARCHITECTURES "${CUDA_SM}")
		set(CMAKE_CUDA_ARCHITECTURES "${AER_CUDA_ARCHITECTURES}")
		message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

		if(NOT DEFINED AER_PYTHON_CUDA_ROOT AND DEFINED ENV{AER_PYTHON_CUDA_ROOT})
			set(AER_PYTHON_CUDA_ROOT $ENV{AER_PYTHON_CUDA_ROOT})
		endif()
		if(NOT DEFINED AER_CIBUILD AND DEFINED ENV{AER_CIBUILD})
			set(AER_CIBUILD $ENV{AER_CIBUILD})
		endif()
		if(AER_PYTHON_CUDA_ROOT)
			find_package(Python)
			if(AER_CIBUILD)
				# cibuildwheel installs into the base interpreter so use its site-packages
				# layout instead of a virtual environment's site-packages
				set(PYTHON_SITE_PATH ${Python_STDLIB}/site-packages)
				set(STR_ORIGIN "\$$ORIGIN")
			else()
				set(PYTHON_SITE_PATH ${AER_PYTHON_CUDA_ROOT}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)
				set(STR_ORIGIN "$ORIGIN")
			endif()

			message(STATUS "PYTHON_SITE_PATH = ${PYTHON_SITE_PATH}")

			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC AER_CUTENSORNET)

			set(CMAKE_CUDA_RUNTIME_LIBRARY None)

			# Layout and sonames of the NVIDIA pip wheels depend on the CUDA major version.
			# CUDA 13 wheels install all libraries under nvidia/cu13/lib and bump the
			# cuSOLVER/cuSPARSE sonames to 12, while CUDA 11 and 12 use one directory per
			# library, cuSOLVER 11 and cuSPARSE matching the CUDA major version.
			if(CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 13)
				set(AER_CUSOLVER_SOVERSION 12)
				set(AER_CUSPARSE_SOVERSION 12)
			else()
				set(AER_CUSOLVER_SOVERSION 11)
				set(AER_CUSPARSE_SOVERSION ${CUDAToolkit_VERSION_MAJOR})
			endif()

			set(CUDA_NVCC_FLAGS "${AER_CUDA_ARCH_FLAGS_EXPAND} -DAER_THRUST_GPU -DAER_THRUST_CUDA -I${AER_SIMULATOR_CPP_SRC_DIR} -isystem ${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers -use_fast_math --expt-extended-lambda")
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${PYTHON_SITE_PATH}/cuquantum/include")
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${PYTHON_SITE_PATH}/cutensor/include")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--disable-new-dtags")
			if(CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 13)
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cu13/lib'")
			else()
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cublas/lib'")
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cusolver/lib'")
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cusparse/lib'")
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cuda_runtime/lib'")
			endif()
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../cutensor/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../cuquantum/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcusolver.so.${AER_CUSOLVER_SOVERSION}")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcusparse.so.${AER_CUSPARSE_SOVERSION}")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${PYTHON_SITE_PATH}/cuquantum/lib")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcustatevec.so.1")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcutensornet.so.2")


			if(CUDAToolkit_VERSION_MAJOR STREQUAL "11")
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcudart.so.${CUDAToolkit_VERSION_MAJOR}.0")
			else()
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcudart.so.${CUDAToolkit_VERSION_MAJOR},-as-needed")
			endif()
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--as-needed")

			string(STRIP ${CUDA_NVCC_FLAGS} CUDA_NVCC_FLAGS)
			string(STRIP ${THRUST_DEPENDANT_LIBS} THRUST_DEPENDANT_LIBS)
		else()
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${AER_CUDA_ARCH_FLAGS_EXPAND} -DAER_THRUST_GPU -DAER_THRUST_CUDA -I${AER_SIMULATOR_CPP_SRC_DIR} -isystem ${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers -use_fast_math --expt-extended-lambda")

			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
			set(THRUST_DEPENDANT_LIBS "-L${CUDAToolkit_LIBRARY_DIR}")
			if(NOT DEFINED CUQUANTUM_ROOT)
				if(DEFINED ENV{CUQUANTUM_ROOT})
					set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
				endif()
			endif()
			if(NOT DEFINED CUTENSOR_ROOT)
				if(DEFINED ENV{CUTENSOR_ROOT})
					set(CUTENSOR_ROOT $ENV{CUTENSOR_ROOT})
				endif()
			endif()
			if(NOT DEFINED AER_ENABLE_CUQUANTUM)
				if(DEFINED ENV{AER_ENABLE_CUQUANTUM})
					set(AER_ENABLE_CUQUANTUM $ENV{AER_ENABLE_CUQUANTUM})
				endif()
			endif()

			if(AER_ENABLE_CUQUANTUM)
				set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC AER_CUTENSORNET)
				if(DEFINED CUQUANTUM_ROOT)
					set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUQUANTUM_ROOT}/include")
					# pip installed packages keep the libraries directly in lib/, SDK tarballs use lib/<CUDA major>
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUQUANTUM_ROOT}/lib -L${CUQUANTUM_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}")
				endif()
				if(DEFINED CUTENSOR_ROOT)
					set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUTENSOR_ROOT}/include")
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUTENSOR_ROOT}/lib -L${CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}")
				endif()
       	     if(CUQUANTUM_STATIC)
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -lcustatevec_static -lcutensornet_static  -lcutensor_static -lmetis_static -lcusolver_static -lcusparse_static -lcusolver_lapack_static -lcublas_static -lcublasLt_static -lculibos")
				else()
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -lcustatevec -lcutensornet -lcutensor")
				endif()
			elseif(CUSTATEVEC_ROOT)
				#TODO this is remained for backward compatibility, use CUQUANTUM_ROOT instead
				set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC)
				set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUSTATEVEC_ROOT}/include")
        	    if(CUSTATEVEC_STATIC)
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUSTATEVEC_ROOT}/lib -L${CUSTATEVEC_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR} -lcustatevec_static -lcusolver_static -lcusparse_static -lcusolver_lapack_static -lcublas_static -lcublasLt_static -lculibos")
				else()
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUSTATEVEC_ROOT}/lib -L${CUSTATEVEC_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR} -lcustatevec")
				endif()
			endif()
		endif()
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
	elseif(AER_THRUST_BACKEND STREQUAL "ROCM")
    #
    # Build with GPU support with ROCm
    #
    
    # Assert that the C++ and compilers are Clang to enable ROCm builds.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(FATAL_ERROR 
      "The compiler for ROCm builds must be Clang. Set CMAKE_CXX_COMPILER to <ROCm path>/llvm/bin/clang++")
    endif()
    
    # GDB debug information is what is needed for runs enabled with ROCm.
    set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ggdb")
    
    # Leverage AER_ROCM_ARCH to specify the relevant targets and send the ROCm default ones to 
    # the background by marking them as advanced. We need to set the architectures in advance 
    # of attemting to find HIP to leverage the package machinery.
    
    string(REPLACE " " ";" AER_ROCM_ARCH_LIST ${AER_ROCM_ARCH})
    set(GPU_TARGETS ${AER_ROCM_ARCH_LIST} CACHE INTERNAL "GPU targets to compile for")
    set(AMDGPU_TARGETS ${AER_ROCM_ARCH_LIST} CACHE INTERNAL "AMD GPU targets to compile for")
    set(CMAKE_HIP_ARCHITECTURES ${AER_ROCM_ARCH_LIST})
    
    mark_as_advanced(GPU_TARGETS)
    mark_as_advanced(AMDGPU_TARGETS)
    mark_as_advanced(CMAKE_HIP_ARCHITECTURES)
    message(STATUS "ROCm build targeting GPU Architectures: ${GPU_TARGETS}")
    
    message(STATUS "ROCm assumed path: ${ROCM_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake ${ROCM_PATH})
    
    include(CheckLanguage)
    check_language(HIP)
    
    # Find HIP in config mode as the module mode may not provide the hip:: targets. We can use module mode
    # if we had hip libraries as they invoke the config package.
    find_package(HIP CONFIG)
    if(HIP_FOUND)
      message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
      message(FATAL_ERROR "Could not find HIP.")
    endif()

    list(APPEND AER_LIBRARIES hip::device)
    
    # Add definitions so that dependencies are properly determined. 
    # TODO: investigate the need for THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    list(APPEND ROCM_EXTRA_DEFS AER_THRUST_GPU AER_THRUST_ROCM THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    
    # Add -D prefix to all defs as that is what ROCM_EXTRA_* expect to be set to.
    list(TRANSFORM ROCM_EXTRA_DEFS PREPEND -D)
    add_definitions(${ROCM_EXTRA_DEFS})
    list(APPEND ROCM_EXTRA_FLAGS ${ROCM_EXTRA_DEFS})
    list(APPEND ROCM_EXTRA_FLAGS -isystem${ROCM_PATH}/include; -I${AER_SIMULATOR_CPP_SRC_DIR} ; -isystem${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers; -ffast-math; -fPIC)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND ROCM_EXTRA_FLAGS -O0)
    else()
      list(APPEND ROCM_EXTRA_FLAGS -O3)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      list(APPEND ROCM_EXTRA_FLAGS -g; -ggdb)
    endif()
    
    # Add some warning flags to allow existing code to go through with clang.
    list(APPEND ROCM_EXTRA_FLAGS -ferror-limit=3
                                 -Wno-unused-lambda-capture
                                 -Wno-bitwise-instead-of-logical
                                 -Wno-inconsistent-missing-override
                                 -Wno-cast-align
                                 -Wno-float-equal
                                 -Wno-unused-variable
                                 -Wno-unused-but-set-variable
                                 -Wno-switch
                                 -Wno-writable-strings
                                 -Wno-shadow
                                 -Wno-delete-non-abstract-non-virtual-dtor
                                 -Wno-pessimizing-move
                                 -Wno-return-type-c-linkage
                                 -Wno-overloaded-virtual
                                 -Wno-braced-scalar-init)
    
    
    if(AER_ENABLE_CUQUANTUM)
      message(WARNING "Implementation of cuQuantum is not available for ROCm builds.")
    endif()
	
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
	set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -g")
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
	${CMAKE_DL_LIBS})

set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} ${CONAN_DEFINES})

add_subdirectory(qiskit_aer/backends/wrappers)

