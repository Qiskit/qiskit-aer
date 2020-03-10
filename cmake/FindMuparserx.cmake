# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#.rst:
# FindMuparserx
# =============
# This module finds a muparserx library installed in the system (see https://beltoforion.de/article.php?a=muparserx
# and https://github.com/beltoforion/muparserx).
#
# This module defines the following target:
#
# ::
#
#   PkgConfig::muparserx
#
# and sets the typical package variables:
#
# ::
#
#   muparserx_FOUND
#   muparserx_INCLUDE_DIRS
#   muparserx_LIBRARIES
#   muparserx_VERSION
#   ....
#

find_package(PkgConfig QUIET)
pkg_check_modules(muparserx QUIET IMPORTED_TARGET muparserx)

if(NOT muparserx_FIND_QUIETLY)
    if(NOT muparserx_FOUND)
        message(FATAL_ERROR "muparserx library not found!")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(muparserx REQUIRED_VARS muparserx_INCLUDE_DIRS muparserx_LIBRARIES
        VERSION_VAR muparserx_VERSION)
