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
QasmSimulator Integration Tests
"""

import unittest
from test.terra import common

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTests
# Conditional instruction tests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalGateTests
# Algorithm circuit tests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
# Noise model simulation tests
from test.terra.backends.qasm_simulator.qasm_noise import QasmReadoutNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmPauliNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmResetNoiseTests
# Snapshot tests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStatevectorTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotDensityMatrixTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStabilizerTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotProbabilitiesTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValPauliTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValMatrixTests
# Other tests
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests


class TestQasmStabilizerSimulator(common.QiskitAerTestCase,
                                  QasmMethodTests,
                                  QasmMeasureTests,
                                  QasmMultiQubitMeasureTests,
                                  QasmResetTests,
                                  QasmConditionalGateTests,
                                  QasmCliffordTests,
                                  QasmReadoutNoiseTests,
                                  QasmResetNoiseTests,
                                  QasmPauliNoiseTests,
                                  QasmSnapshotStatevectorTests,
                                  QasmSnapshotDensityMatrixTests,
                                  QasmSnapshotProbabilitiesTests,
                                  QasmSnapshotExpValPauliTests,
                                  QasmSnapshotExpValMatrixTests,
                                  QasmSnapshotStabilizerTests):
    """QasmSimulator stabilizer method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 1337,
        "method": "stabilizer"
    }


if __name__ == '__main__':
    unittest.main()
