# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================================
Pulse System Models (:mod:`qiskit.providers.aer.pulse`)
=======================================================

.. currentmodule:: qiskit.providers.aer.pulse

This module contains classes and functions to build a pulse system model
for simulating a Qiskit pulse schedule.


Classes
=======

.. autosummary::
    :toctree: ../stubs/

    PulseSystemModel


Functions
=========

These functions can be used to generate a pulse system model for certain types
of systems.

.. autosummary::
    :toctree: ../stubs/

    duffing_system_model
"""

# pylint: disable=import-error
import distutils.sysconfig  # noqa
import numpy as np
from .qutip_extra_lite.cy import pyxbuilder as pbldr

from .system_models.duffing_model_generators import duffing_system_model
from .system_models.pulse_system_model import PulseSystemModel

# Remove -Wstrict-prototypes from cflags
CFG_VARS = distutils.sysconfig.get_config_vars()
if "CFLAGS" in CFG_VARS:
    CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")

# Setup pyximport
# pylint: disable=no-member
pbldr.install(setup_args={'include_dirs': [np.get_include()]})
del pbldr
