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
	if(DISABLE_CONAN)
		_use_system_libraries()
	else()
		include(conan_utils)
		setup_conan()
	endif()
endmacro()

macro(_use_system_libraries)
	# Use system libraries
	_import_aer_system_dependency(nlohmann_json 3.1.1)
	_import_aer_system_dependency(spdlog 1.5.0)

	if(SKBUILD)
		_import_aer_system_dependency(muparserx 4.0.8)
	endif()

	if(AER_THRUST_BACKEND AND NOT AER_THRUST_BACKEND STREQUAL "CUDA")
		string(TOLOWER ${AER_THRUST_BACKEND} THRUST_BACKEND)
		_import_aer_system_dependency(Thrust 1.9.5)
	endif()

	if(BUILD_TESTS)
		_import_aer_system_dependency(Catch2 2.13.6)
	endif()

	if(APPLE)
		# Fix linking. See https://stackoverflow.com/questions/54068035
		link_directories(/usr/local/lib) #brew
		link_directories(/opt/local/lib) #ports
	endif()
endmacro()

macro(_import_aer_system_dependency package version)
	# Arguments:
		# package: name of package to search for using find_package()
		# version: version of package to search for

	find_package(${package} ${version} EXACT QUIET)
	if(NOT ${package}_FOUND)
		message(STATUS "${package} ${version} NOT found! Looking for any other version available.")
		find_package(${package} REQUIRED)
		message(STATUS "${package} version found: ${${package}_VERSION}. WARNING: This version may not work!!!")
	endif()
	string(TOLOWER ${package} PACKAGE_LOWER) # Conan use lowercase for every lib
	add_library(AER_DEPENDENCY_PKG::${PACKAGE_LOWER} INTERFACE IMPORTED)
	target_link_libraries(AER_DEPENDENCY_PKG::${PACKAGE_LOWER} PUBLIC INTERFACE ${package})
	message(STATUS "Using system-provided ${PACKAGE_LOWER} library")
endmacro()
