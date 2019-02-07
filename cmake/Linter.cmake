function(add_linter target)
    find_program(CLANG_TIDY_EXE "clang-tidy")
    if(NOT CLANG_TIDY_EXE)
        message(WARNING "The 'lint' target will not be available: 'clang-tidy' was not found.")
    else()
        # TODO: Once final checks are decided, add "-warnings-as-errors=*".
        set(CLANG_TIDY_PROPERTIES "${CLANG_TIDY_EXE}"
            "-checks=*"
            "-format-style=google"
            "-header-filter=\"${AER_SIMULATOR_CPP_SRC_DIR}\""
            "-quiet")
        # This will add the linter as part of the build process
        set_target_properties(${target} PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_PROPERTIES}")
        message("Clang Tidy linter will be passed at build time")
    endif()
endfunction()
