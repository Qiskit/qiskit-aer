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
==================================================================
Errors for Noise Models (:mod:`qiskit.providers.aer.noise.errors`)
==================================================================

.. currentmodule:: qiskit.providers.aer.noise.errors


Classes
=======

The following are the base classes used to represented error terms in a
Qiskit Aer :class:`NoiseModel`.

.. autosummary::
    :toctree: ../stubs/

    QuantumError
    ReadoutError


Generator Functions
===================

The following functions can be used to generate many common types of
:class:`QuantumError` objects for inclusion in a :class:`~qiskit.providers.aer.noise.NoiseModel`.

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
"""

from .readout_error import ReadoutError
from .quantum_error import QuantumError
from .standard_errors import kraus_error
from .standard_errors import mixed_unitary_error
from .standard_errors import coherent_unitary_error
from .standard_errors import pauli_error
from .standard_errors import depolarizing_error
from .standard_errors import reset_error
from .standard_errors import thermal_relaxation_error
from .standard_errors import phase_amplitude_damping_error
from .standard_errors import amplitude_damping_error
from .standard_errors import phase_damping_error
