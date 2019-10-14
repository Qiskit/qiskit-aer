# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Init for openpulse"""

# pylint: disable=import-error
import distutils.sysconfig  # noqa
import numpy as np
#from .qutip_lite.cy import pyxbuilder as pbldr

# Remove -Wstrict-prototypes from cflags
CFG_VARS = distutils.sysconfig.get_config_vars()
if "CFLAGS" in CFG_VARS:
    CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")

# Setup pyximport
# pylint: disable=no-member
#pbldr.install(setup_args={'include_dirs': [np.get_include()]})
#del pbldr
