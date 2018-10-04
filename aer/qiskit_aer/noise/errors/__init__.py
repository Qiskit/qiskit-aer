# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Standard error module for Qiskit Aer.

This module contains functions to generate QuantumError objects for
standard noise channels in quantum information science.

errors (package)
|
|-- mixed_unitary_error
|-- coherent_unitary_error
|-- pauli_channel_error
|-- depolarizing_channel_error
|-- thermal_relaxation_error
|-- phase_amplitude_damping_error
|-- amplitude_damping_error
|-- phase_damping_error
"""

from .standard_errors import mixed_unitary_error
from .standard_errors import coherent_unitary_error
from .standard_errors import pauli_channel_error
from .standard_errors import depolarizing_channel_error
from .standard_errors import thermal_relaxation_error
from .standard_errors import phase_amplitude_damping_error
from .standard_errors import amplitude_damping_error
from .standard_errors import phase_damping_error
