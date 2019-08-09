function(add_linter target)
    find_program(CLANG_TIDY_EXE "clang-tidy")
    if(NOT CLANG_TIDY_EXE)
        message(WARNING "Bypassing C++ linter because 'clang-tidy' was not found.")
    else()
        # TODO: Once final checks are decided, add "-warnings-as-errors=*".
        set(CLANG_TIDY_PROPERTIES "${CLANG_TIDY_EXE}"
            "-format-style=google"
            "-header-filter=\"${AER_SIMULATOR_CPP_SRC_DIR}\""
            "-quiet")
        # This will add the linter as part of the build process
        set_target_properties(${target} PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_PROPERTIES}")
        message("Clang Tidy linter will be passed at build time")

        # We create two more custom targets, so we can invoque both linter and
        # format from the command line.

        # Get all project files
        file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.hpp)
        add_custom_target(
            linter
            COMMAND ${CLANG_TIDY_EXE}
            ${ALL_SOURCE_FILES}
            ${CLANG_TIDY_PROPERTIES}
            ${INCLUDE_DIRECTORIES}
        )

        find_program(CLANG_TIDY_FORMAT "clang-format")
        if(NOT CLANG_TIDY_FORMAT)
            message(WARNING "Bypassing C++ auto-format because 'clang-format' was not found.")
        else()
            add_custom_target(
                format
                COMMAND ${CLANG_TIDY_FORMAT}
                -style=file
                -i
                ${ALL_SOURCE_FILES}
            )
        endif()
    endif()
endfunction()
