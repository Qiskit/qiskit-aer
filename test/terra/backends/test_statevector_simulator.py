# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
StatevectorSimulator Integration Tests
"""

import unittest
from test.terra import common
from test.terra.decorators import requires_method
# Basic circuit instruction tests
from test.terra.backends.statevector_simulator.statevector_basics import StatevectorSimulatorTests


class TestStatevectorSimulator(common.QiskitAerTestCase,
                               StatevectorSimulatorTests):
    """StatevectorSimulator automatic method tests."""

    BACKEND_OPTS = {"seed_simulator": 10598}


@requires_method("statevector_simulator", "statevector_gpu")
class TestStatevectorSimulatorThrustGPU(common.QiskitAerTestCase,
                                        StatevectorSimulatorTests):
    """StatevectorSimulator automatic method tests."""

    BACKEND_OPTS = {"seed_simulator": 10598, "method": "statevector_gpu"}


@requires_method("statevector_simulator", "statevector_thrust")
class TestStatevectorSimulatorThrustCPU(common.QiskitAerTestCase,
                                        StatevectorSimulatorTests):
    """StatevectorSimulator automatic method tests."""

    BACKEND_OPTS = {"seed_simulator": 10598, "method": "statevector_thrust"}


if __name__ == '__main__':
    unittest.main()
