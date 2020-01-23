# This is an undocumented feature of CMake. With this PreLoad.cmake file we
# can set the Generator without using the command line argument -G.
# This is needed in order to make automatic wheel generator works in our CIs
# Please, don't rely on this functionality.

if (DEFINED ENV{FORCE_GENERATOR})
    set(CMAKE_GENERATOR ENV{FORCE_GENERATOR} CACHE INTERNAL "" FORCE)
endif()
