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

"""
================================================
Noise Models (:mod:`qiskit.providers.aer.noise`)
================================================

.. currentmodule:: qiskit.providers.aer.noise

This module contains classes and functions to build a noise model for
simulating a Qiskit quantum circuit in the presence of errors.


Classes
=======

.. autosummary::
    :toctree: ../stubs/

    NoiseModel


Submodules
==========

Errors for Noise Models
-----------------------

The :mod:`qiskit.providers.aer.noise.errors` submodule contains classes
and functions for constructing generating errors for custom noise models.


Device Noise Models
-------------------

The :mod:`qiskit.providers.aer.noise.device` submodule contains functions
for generating approximate noise models for a hardware device.


Noise Utilities
---------------

The :mod:`qiskit.providers.aer.noise.utils` submodule contains utilities
for remapping and approximating noise models, and inserting noise into
quantum circuits.
"""

from .noise_model import NoiseModel
from . import errors
from . import device
from . import utils
