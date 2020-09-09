# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


macro(setup_dependencies)
	# Defines AER_DEPENDENCY_PKG alias which refers to either conan-provided or system libraries.
	if(USE_CONAN)
		include(conan_utils)
		setup_conan()

		# NOTE: this assumes CONAN_PKG is static.
		# Might need to change to import CONAN_PKG if they need changed
		add_library(AER_DEPENDENCY_PKG ALIAS CONAN_PKG)
	else()
		# Use system libraries
		_import_aer_system_dependency(nlohmann_json 3.1.1)
		_import_aer_system_dependency(spdlog 1.5.0)

		if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
			_import_aer_system_dependency(llvm-openmp 8.0.1)
		endif()

		if(SKBUILD)
			_import_aer_system_dependency(muparserx 4.0.8)
		endif()

		if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA")
			string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
			_import_aer_system_dependency(thrust 1.9.5)
		endif()

		if(BUILD_TESTS)
			_import_aer_system_dependency(Catch2 2.12.1)
		endif()
	endif()
endmacro()

macro(_import_aer_system_dependency package version)
	# Arguments:
		# package: name of package to search for using find_package()
		# version: version of package to search for

	find_package(${package} ${version} REQUIRED)
	add_library(AER_DEPENDENCY_PKG::${package} INTERFACE IMPORTED)
	target_link_libraries(AER_DEPENDENCY_PKG::${package} PUBLIC INTERFACE ${package})
	message(STATUS "Using system-provided ${package} library")
endmacro()
