find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

add_cython_target(aer_qv_wrapper aer_qv_wrapper.pyx CXX)
add_library(aer_qv_wrapper MODULE ${aer_qv_wrapper})
set_target_properties(aer_qv_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
target_include_directories(aer_qv_wrapper
    PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS}
    PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(aer_qv_wrapper
    ${PYTHON_LIBRARIES}
    ${AER_LIBRARIES})
python_extension_module(aer_qv_wrapper
    LINKED_MODULES_VAR linked_module_list
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

# application executable -- generated header file + other source files
python_modules_header(modules
                      FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})

set(SITE_PACKAGES "${PYTHON_RELATIVE_SITE_PACKAGES_DIR}")
install(TARGETS aer_qv_wrapper LIBRARY DESTINATION ${SITE_PACKAGES})