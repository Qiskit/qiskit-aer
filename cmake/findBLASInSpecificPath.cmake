function(find_blas_in_specific_path BLAS_LIB_PATH)

    function(check_blas_lib_found BLAS_LIB_PATH BLAS_LIBS BLAS_FOUND)
        # This function is intented to be called only from find_blas_in_specific_path
        # Check if the lib has been found and if it was in a sub of the provided path.
        # FindBLAS.cmake always search some standard paths so, if no lib is found
        # below provided BLAS_LIB_PATH, it could have found the BLAS lib at some other place
        # and not where the user specified
        if(NOT BLAS_FOUND)
            return()
        endif()
        list(GET BLAS_LIBS 0 FIRST_BLAS_PATH)
        # Need to add final separator in order to check if BLAS_LIB_PATH is a
        # parent directory for BLAS_LIBS
        string(APPEND BLAS_LIB_PATH "/") # already in CMake format
        file(TO_CMAKE_PATH ${FIRST_BLAS_PATH} FIRST_BLAS_PATH)
        string(APPEND FIRST_BLAS_PATH "/") # already in CMake format
        string(FIND ${FIRST_BLAS_PATH} ${BLAS_LIB_PATH} BLAS_DIR_MATCH)
        if(NOT BLAS_FOUND OR NOT BLAS_DIR_MATCH STREQUAL "0")
            set(BLAS_FOUND FALSE PARENT_SCOPE)
        endif()
    endfunction()

    get_filename_component(BLAS_LIB_PATH ${BLAS_LIB_PATH} ABSOLUTE)
    message(STATUS "Looking for BLAS library in user defined dir: ${BLAS_LIB_PATH}")
    file(TO_CMAKE_PATH ${BLAS_LIB_PATH} BLAS_LIB_PATH)
    if(NOT IS_DIRECTORY ${BLAS_LIB_PATH})
        message(FATAL_ERROR "${BLAS_LIB_PATH} is not a valid directory")
    endif()

    # Modify CMAKE_PREFIX_PATH locally to only search in provided dir
    # (though FindBLAS ALWAYS search certain system dirs)
    set(CMAKE_PREFIX_PATH ${BLAS_LIB_PATH})

    find_package(BLAS QUIET)
    # Check if BLAS libs are under provided DIR
    check_blas_lib_found(${BLAS_LIB_PATH} ${BLAS_LIBRARIES} ${BLAS_FOUND})

    if(NOT BLAS_FOUND AND APPLE)
        # We may need to search for specific APPLE framework
        # Try again setting the BLA_VENDOR to APPLE
        set(BLA_VENDOR "Apple")
        find_package(BLAS QUIET)
        check_blas_lib_found(${BLAS_LIB_PATH} ${BLAS_LIBRARIES} ${BLAS_FOUND})
    endif()

    if(NOT BLAS_FOUND)
        message(FATAL_ERROR "BLAS library not found in dir: ${BLAS_LIB_PATH}")
    endif()

    set(BLAS_LIBRARIES ${BLAS_LIBRARIES} PARENT_SCOPE)
    set(BLAS_FOUND ${BLAS_FOUND} PARENT_SCOPE)
    set(BLAS_LINKER_FLAGS ${BLAS_LINKER_FLAGS} PARENT_SCOPE)
endfunction()
