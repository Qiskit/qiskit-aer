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
from qiskit.providers.aer import QasmSimulator

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmDiagonalGateTests
from test.terra.backends.qasm_simulator.qasm_initialize import QasmInitializeTests
from test.terra.backends.qasm_simulator.qasm_multiplexer import QasmMultiplexerTests
from test.terra.backends.qasm_simulator.qasm_standard_gates import QasmStandardGateStatevectorTests
from test.terra.backends.qasm_simulator.qasm_standard_gates import QasmStandardGateDensityMatrixTests
from test.terra.backends.qasm_simulator.qasm_delay_gate import QasmDelayGateTests
# Conditional instruction tests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalGateTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalUnitaryTests
from test.terra.backends.qasm_simulator.qasm_conditional import QasmConditionalKrausTests
# Algorithm circuit tests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTests
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_algorithms import QasmAlgorithmTestsMinimalBasis
# Noise model simulation tests
from test.terra.backends.qasm_simulator.qasm_noise import QasmReadoutNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmPauliNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmResetNoiseTests
from test.terra.backends.qasm_simulator.qasm_noise import QasmKrausNoiseTests
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
# Other tests
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests
from test.terra.backends.qasm_simulator.qasm_thread_management import QasmThreadManagementTests
from test.terra.backends.qasm_simulator.qasm_fusion import QasmFusionTests
from test.terra.backends.qasm_simulator.qasm_delay_measure import QasmDelayMeasureTests
from test.terra.backends.qasm_simulator.qasm_truncate import QasmQubitsTruncateTests
from test.terra.backends.qasm_simulator.qasm_basics import QasmBasicsTests


class TestQasmSimulator(common.QiskitAerTestCase,
                        QasmMethodTests,
                        QasmMeasureTests,
                        QasmMultiQubitMeasureTests,
                        QasmResetTests,
                        QasmInitializeTests,
                        QasmConditionalGateTests,
                        QasmConditionalUnitaryTests,
                        QasmConditionalKrausTests,
                        QasmMultiplexerTests,
                        QasmAlgorithmTests,
                        QasmAlgorithmTestsWaltzBasis,
                        QasmAlgorithmTestsMinimalBasis,
                        QasmUnitaryGateTests,
                        QasmDiagonalGateTests,
                        QasmReadoutNoiseTests,
                        QasmPauliNoiseTests,
                        QasmThreadManagementTests,
                        QasmFusionTests,
                        QasmDelayMeasureTests,
                        QasmQubitsTruncateTests,
                        QasmResetNoiseTests,
                        QasmKrausNoiseTests,
                        QasmBasicsTests,
                        QasmSaveDataTests,
                        QasmSetStateTests,
                        QasmStandardGateStatevectorTests,
                        QasmStandardGateDensityMatrixTests,
                        QasmDelayGateTests,
                        QasmSnapshotStatevectorTests,
                        QasmSnapshotDensityMatrixTests,
                        QasmSnapshotProbabilitiesTests,
                        QasmSnapshotExpValPauliTests,
                        QasmSnapshotExpValPauliNCTests,
                        QasmSnapshotExpValMatrixTests,
                        QasmSnapshotStabilizerTests
                        ):
    """QasmSimulator automatic method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 2113,
        "method": "automatic",
        "max_parallel_threads": 1
    }
    SIMULATOR = QasmSimulator(**BACKEND_OPTS)


if __name__ == '__main__':
    unittest.main()
