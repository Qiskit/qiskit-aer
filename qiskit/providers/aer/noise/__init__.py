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

The following are the classes used to represented noise and error terms.

.. autosummary::
    :toctree: ../stubs/

    NoiseModel
    QuantumError
    ReadoutError


Quantum Error Functions
=======================

The following functions can be used to generate many common types of
:class:`QuantumError` objects for inclusion in a :class:`NoiseModel`.

.. autosummary::
    :toctree: ../stubs/

    pauli_error
    depolarizing_error
    pauli_error
    mixed_unitary_error
    coherent_unitary_error
    reset_error
    amplitude_damping_error
    phase_damping_error
    phase_amplitude_damping_error
    thermal_relaxation_error
    kraus_error


Noise Model Functions
=====================

The following functions can be used to generate approximate noise models for
IBMQ hardware devices based on the parameters in their backend properties.

.. autosummary::
    :toctree: ../stubs/

    basic_device_noise_model
    basic_device_readout_errors
    basic_device_gate_errors


Noise Model Utilities
=====================

The :mod:`qiskit.providers.aer.noise.utils` submodule contains utilities
for remapping and approximating noise models, and inserting noise into
quantum circuits.
"""

# Noise and Error classes
from .noise_model import NoiseModel
from .errors import QuantumError
from .errors import ReadoutError

# Error generating functions
from .errors import kraus_error
from .errors import mixed_unitary_error
from .errors import coherent_unitary_error
from .errors import pauli_error
from .errors import depolarizing_error
from .errors import reset_error
from .errors import thermal_relaxation_error
from .errors import phase_amplitude_damping_error
from .errors import amplitude_damping_error
from .errors import phase_damping_error

# Noise model generating functions
from .device.models import basic_device_noise_model
from .device.models import basic_device_readout_errors
from .device.models import basic_device_gate_errors

# Submodules
from . import errors
from . import device
from . import utils
