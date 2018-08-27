set(Catch2_PATH "third-party/")
find_package(Catch2 REQUIRED)

add_executable(test_snapshot "src/test_snapshot.cpp")
set_target_properties(test_snapshot PROPERTIES 
										LINKER_LANGUAGE CXX
										CXX_STANDARD 14)
target_include_directories(test_snapshot
                            PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
                            PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
target_link_libraries(test_snapshot 
                        PRIVATE Catch2::Catch
                        PRIVATE ${LIBRARIES})
add_test(test_snapshot test_snapshot)



# Don't forget to add your test target here
add_custom_target(build_tests test_snapshot)