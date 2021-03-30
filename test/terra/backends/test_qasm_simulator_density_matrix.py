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
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import AerError
from test.terra import common
from test.terra.decorators import requires_method

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmDiagonalGateTests
from test.terra.backends.qasm_simulator.qasm_standard_gates import QasmStandardGateDensityMatrixTests
from test.terra.backends.qasm_simulator.qasm_delay_gate import QasmDelayGateTests
# Conditional instruction tests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalGateTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalUnitaryTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalKrausTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalSuperOpTests
# Algorithm circuit tests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
# Noise model simulation tests
from test.terra.backends.qasm_simulator.qasm_noise import QasmReadoutNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmPauliNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmResetNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmKrausNoiseTests
# Other tests
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests
from test.terra.backends.qasm_simulator.qasm_fusion import QasmFusionTests
# Save data tests
from test.terra.backends.qasm_simulator.qasm_save import QasmSaveDataTests
from test.terra.backends.qasm_simulator.qasm_set_state import QasmSetStateTests
# Snapshot tests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStatevectorTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotDensityMatrixTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStabilizerTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotProbabilitiesTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValPauliTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValPauliNCTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValMatrixTests


class DensityMatrixTests(
        QasmMethodTests, QasmFusionTests, QasmMeasureTests, QasmMultiQubitMeasureTests,
        QasmResetTests, QasmConditionalGateTests, QasmConditionalUnitaryTests,
        QasmConditionalKrausTests, QasmConditionalSuperOpTests,
        QasmAlgorithmTests, QasmAlgorithmTestsWaltzBasis,
        QasmAlgorithmTestsMinimalBasis, QasmUnitaryGateTests, QasmDiagonalGateTests,
        QasmReadoutNoiseTests, QasmPauliNoiseTests, QasmResetNoiseTests,
        QasmKrausNoiseTests, QasmSaveDataTests, QasmSetStateTests,
        QasmSnapshotStatevectorTests,
        QasmSnapshotDensityMatrixTests, QasmSnapshotProbabilitiesTests,
        QasmSnapshotExpValPauliTests, QasmSnapshotExpValPauliNCTests,
        QasmSnapshotExpValMatrixTests, QasmSnapshotStabilizerTests,
        QasmStandardGateDensityMatrixTests, QasmDelayGateTests):
    """Container class of density_matrix method tests."""
    pass


class TestQasmSimulatorDensityMatrix(common.QiskitAerTestCase,
                                     DensityMatrixTests):
    """QasmSimulator density_matrix method tests."""
    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix",
        "max_parallel_threads": 1
    }
    SIMULATOR = QasmSimulator(**BACKEND_OPTS)


@requires_method("qasm_simulator", "density_matrix_gpu")
class TestQasmSimulatorDensityMatrixThrustGPU(common.QiskitAerTestCase,
                                              DensityMatrixTests):
    """QasmSimulator density_matrix_gpu method tests."""
    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix_gpu",
        "max_parallel_threads": 1
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


@requires_method("qasm_simulator", "density_matrix_thrust")
class TestQasmSimulatorDensityMatrixThrustCPU(common.QiskitAerTestCase,
                                              DensityMatrixTests):
    """QasmSimulator density_matrix_thrust method tests."""
    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix_thrust",
        "max_parallel_threads": 1
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


if __name__ == '__main__':
    unittest.main()
