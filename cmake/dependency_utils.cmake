# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Fetch Aer's C++ dependencies via CMake FetchContent. By default, if a system
# install satisfies the version constraint (FIND_PACKAGE_ARGS), it is used in
# preference to a fresh download. If AER_USE_SYSTEM_LIBS is set, downloads are
# disabled entirely and missing system libraries are fatal — this matches what
# downstream packagers (e.g. conda-forge) need.

include(FetchContent)

macro(setup_dependencies)
	if(AER_USE_SYSTEM_LIBS)
		# Force find_package only; never download.
		set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)
		set(FETCHCONTENT_FULLY_DISCONNECTED TRUE)
	endif()

	# nlohmann_json: 3.10.3+ is required for the explicit JSON-conversion path
	# wired up in the noise-model loader.
	FetchContent_Declare(nlohmann_json
		GIT_REPOSITORY https://github.com/nlohmann/json.git
		GIT_TAG v3.11.3
		GIT_SHALLOW TRUE
		EXCLUDE_FROM_ALL
		FIND_PACKAGE_ARGS 3.10.3
	)

	FetchContent_Declare(spdlog
		GIT_REPOSITORY https://github.com/gabime/spdlog.git
		GIT_TAG v1.14.1
		GIT_SHALLOW TRUE
		EXCLUDE_FROM_ALL
		FIND_PACKAGE_ARGS 1.10
	)

	FetchContent_MakeAvailable(nlohmann_json spdlog)

	# Expose each fetched package under the AER_DEPENDENCY_PKG::<pkg> alias
	# that the Aer source tree links against.
	if(NOT TARGET AER_DEPENDENCY_PKG::nlohmann_json)
		add_library(AER_DEPENDENCY_PKG::nlohmann_json INTERFACE IMPORTED)
		target_link_libraries(AER_DEPENDENCY_PKG::nlohmann_json INTERFACE nlohmann_json::nlohmann_json)
	endif()
	if(NOT TARGET AER_DEPENDENCY_PKG::spdlog)
		add_library(AER_DEPENDENCY_PKG::spdlog INTERFACE IMPORTED)
		target_link_libraries(AER_DEPENDENCY_PKG::spdlog INTERFACE spdlog::spdlog)
	endif()

	if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA" AND NOT AER_THRUST_BACKEND STREQUAL "ROCM")
		# For OMP/TBB Thrust backends we use Thrust as a header-only dep. The
		# CUDA backend uses the Thrust shipped with the CUDA toolkit; ROCm uses
		# its own HIP-Thrust.
		set(THRUST_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
		set(THRUST_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
		set(THRUST_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
		set(THRUST_ENABLE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
		FetchContent_Declare(thrust
			GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
			GIT_TAG 1.17.2
			GIT_SHALLOW TRUE
			EXCLUDE_FROM_ALL
			FIND_PACKAGE_ARGS 1.17
		)
		FetchContent_MakeAvailable(thrust)
		if(NOT TARGET AER_DEPENDENCY_PKG::thrust)
			add_library(AER_DEPENDENCY_PKG::thrust INTERFACE IMPORTED)
			# Use the source dir as the include path; the upstream Thrust
			# CMake export uses thrust_create_target() which we deliberately
			# avoid here — Aer only needs the headers.
			if(thrust_SOURCE_DIR)
				target_include_directories(AER_DEPENDENCY_PKG::thrust INTERFACE ${thrust_SOURCE_DIR})
			else()
				target_link_libraries(AER_DEPENDENCY_PKG::thrust INTERFACE Thrust::Thrust)
			endif()
		endif()
		# Preserve the legacy variable used by CMakeLists.txt's CUDA-fallback path.
		if(thrust_SOURCE_DIR)
			set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${thrust_SOURCE_DIR})
		endif()
	endif()

	if(BUILD_TESTS)
		FetchContent_Declare(Catch2
			GIT_REPOSITORY https://github.com/catchorg/Catch2.git
			GIT_TAG v3.7.1
			GIT_SHALLOW TRUE
			EXCLUDE_FROM_ALL
			FIND_PACKAGE_ARGS 3.0
		)
		FetchContent_MakeAvailable(Catch2)
	endif()

	if(APPLE)
		# Allow the BLAS/LAPACK find paths to pick up Homebrew/MacPorts installs.
		link_directories(/usr/local/lib)        # brew (Intel)
		link_directories(/opt/homebrew/lib)     # brew (arm64)
		link_directories(/opt/local/lib)        # MacPorts
	endif()
endmacro()
