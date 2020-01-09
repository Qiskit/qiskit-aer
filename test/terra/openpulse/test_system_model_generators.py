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
Tests for pulse system generator functions
"""

import unittest
from test.terra.common import QiskitAerTestCase
from qiskit.providers.aer.openpulse.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.openpulse.hamiltonian_model import HamiltonianModel
from qiskit.providers.aer.openpulse import pulse_model_generators as model_gen
