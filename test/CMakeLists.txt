
macro(add_test_executable target_name)
	add_executable(${target_name} ${ARGN})
	set_target_properties(${target_name} PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14)
	target_include_directories(${target_name}
			PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
			PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
	target_link_libraries(${target_name}
			PRIVATE AER_DEPENDENCY_PKG::catch2
			PRIVATE ${AER_LIBRARIES})
	add_test(${target_name} ${target_name})
	if(WIN32 AND NOT BLAS_LIB_PATH)
		add_custom_command(TARGET test_linalg POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${BACKEND_REDIST_DEPS}
				$<TARGET_FILE_DIR:${target_name}>)
	endif()
endmacro()

add_test_executable(test_linalg "src/test_linalg.cpp")

# Don't forget to add your test target here
add_custom_target(build_tests
		test_linalg)
