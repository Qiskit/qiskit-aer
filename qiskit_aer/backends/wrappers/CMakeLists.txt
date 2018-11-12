find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

add_cython_target(qv_wrapper qv_wrapper.pyx CXX)
add_library(qv_wrapper MODULE ${qv_wrapper})
set_target_properties(qv_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(qv_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})

if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Pyhton library, they will be
        solved at runtime by the loader")
    set_target_properties(qv_wrapper PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    unset(PYTHON_LIBRARIES)
endif()

target_link_libraries(qv_wrapper
    ${AER_LIBRARIES}
    ${PYTHON_LIBRARIES})

python_extension_module(qv_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})

set(SITE_PACKAGES "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
install(TARGETS qv_wrapper LIBRARY DESTINATION ${SITE_PACKAGES})