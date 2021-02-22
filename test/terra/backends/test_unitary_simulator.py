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
from qiskit.providers.aer import UnitarySimulator
from qiskit.providers.aer import AerError
from test.terra import common
from test.terra.decorators import requires_method

from test.terra.backends.unitary_simulator.unitary_basics import UnitarySimulatorTests
from test.terra.backends.unitary_simulator.unitary_snapshot import UnitarySnapshotTests
from test.terra.backends.unitary_simulator.unitary_fusion import UnitaryFusionTests
from test.terra.backends.unitary_simulator.unitary_gates import UnitaryGateTests
from test.terra.backends.unitary_simulator.unitary_save import UnitarySaveUnitaryTests


class TestUnitarySimulator(common.QiskitAerTestCase,
                           UnitaryGateTests,
                           UnitarySimulatorTests,
                           UnitarySnapshotTests,
                           UnitaryFusionTests,
                           UnitarySaveUnitaryTests):
    """UnitarySimulator automatic method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113}
    SIMULATOR = UnitarySimulator(**BACKEND_OPTS)


@requires_method("unitary_simulator", "unitary_gpu")
class TestUnitarySimulatorThrustGPU(common.QiskitAerTestCase,
                                    UnitaryGateTests,
                                    UnitarySimulatorTests,
                                    UnitaryFusionTests,
                                    UnitarySaveUnitaryTests):
    """UnitarySimulator unitary_gpu method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113, "method": "unitary_gpu"}
    try:
        SIMULATOR = UnitarySimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


@requires_method("unitary_simulator", "unitary_thrust")
class TestUnitarySimulatorThrustCPU(common.QiskitAerTestCase,
                                    UnitaryGateTests,
                                    UnitarySimulatorTests,
                                    UnitaryFusionTests,
                                    UnitarySaveUnitaryTests):
    """UnitarySimulator unitary_thrust method tests."""

    BACKEND_OPTS = {"seed_simulator": 2113, "method": "unitary_thrust"}
    try:
        SIMULATOR = UnitarySimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


if __name__ == '__main__':
    unittest.main()
