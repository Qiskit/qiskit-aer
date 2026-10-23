# Patch script to disable compiler check in generated conanbuildinfo.cmake
# This is called after Conan generates the file but before it's loaded

function(patch_conan_buildinfo BUILDINFO_FILE)
    if(EXISTS "${BUILDINFO_FILE}")
        message(STATUS "Patching ${BUILDINFO_FILE} to disable compiler check")
        
        # Read the file
        file(READ "${BUILDINFO_FILE}" BUILDINFO_CONTENT)
        
        # Replace the compiler check function to always return early
        # Find: function(conan_check_compiler)
        # Replace with: function(conan_check_compiler)\n    return()
        string(REPLACE 
            "function(conan_check_compiler)"
            "function(conan_check_compiler)\n    message(STATUS \"WARN: Conan compiler check disabled by patch\")\n    return()"
            BUILDINFO_CONTENT 
            "${BUILDINFO_CONTENT}")
        
        # Write back
        file(WRITE "${BUILDINFO_FILE}" "${BUILDINFO_CONTENT}")
        message(STATUS "Successfully patched conanbuildinfo.cmake")
    else()
        message(WARNING "Could not find ${BUILDINFO_FILE} to patch")
    endif()
endfunction()
