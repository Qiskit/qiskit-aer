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
UnitarySimulator Integration Tests
"""

import unittest
from test.terra import common
from test.terra.decorators import requires_method
# Basic circuit instruction tests
from test.terra.backends.unitary_simulator.unitary_basics import UnitarySimulatorTests
from test.terra.backends.unitary_simulator.unitary_snapshot import UnitarySnapshotTests

class TestUnitarySimulator(common.QiskitAerTestCase, UnitarySimulatorTests, UnitarySnapshotTests):
    """UnitarySimulator automatic method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113}


@requires_method("unitary_simulator", "unitary_gpu")
class TestUnitarySimulatorThrustGPU(common.QiskitAerTestCase,
                                    UnitarySimulatorTests):
    """UnitarySimulator unitary_gpu method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113, "method": "unitary_gpu"}


@requires_method("unitary_simulator", "unitary_thrust")
class TestUnitarySimulatorThrustCPU(common.QiskitAerTestCase,
                                    UnitarySimulatorTests):
    """UnitarySimulator unitary_thrust method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113, "method": "unitary_thrust"}


if __name__ == '__main__':
    unittest.main()
