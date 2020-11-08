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
QasmSimulator matrix product state method integration tests
"""

import os
import unittest
from test.terra import common

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsTGate
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsCCXGate
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmDiagonalGateTests
from test.terra.backends.qasm_simulator.qasm_standard_gates import QasmStandardGateStatevectorTests
# from test.terra.backends.qasm_simulator.qasm_initialize import QasmInitializeTests
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
# Snapshot tests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStatevectorTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotDensityMatrixTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotStabilizerTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotProbabilitiesTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValPauliTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValPauliNCTests
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValMatrixTests


class TestQasmMatrixProductStateSimulator(
        common.QiskitAerTestCase,
        QasmMeasureTests,
        QasmMultiQubitMeasureTests,
        QasmResetTests,
        QasmConditionalGateTests,
        QasmConditionalUnitaryTests,
        QasmConditionalKrausTests,
        QasmCliffordTests,
        QasmCliffordTestsWaltzBasis,
        QasmCliffordTestsMinimalBasis,
        QasmNonCliffordTestsTGate,
        QasmNonCliffordTestsCCXGate,
        QasmNonCliffordTestsWaltzBasis,
        QasmNonCliffordTestsMinimalBasis,
        QasmAlgorithmTests,
        QasmAlgorithmTestsWaltzBasis,
        QasmAlgorithmTestsMinimalBasis,
        QasmUnitaryGateTests,
        QasmDiagonalGateTests,
        # QasmInitializeTests,  # THROWS: partial initialize not supported
        QasmReadoutNoiseTests,
        QasmPauliNoiseTests,
        QasmResetNoiseTests,
        QasmKrausNoiseTests,
        QasmSnapshotStatevectorTests,
        QasmSnapshotDensityMatrixTests,
        QasmSnapshotProbabilitiesTests,
        QasmSnapshotStabilizerTests,
        QasmSnapshotExpValPauliTests,
        QasmSnapshotExpValPauliNCTests,
        QasmSnapshotExpValMatrixTests,
        QasmStandardGateStatevectorTests
):
    """QasmSimulator matrix product state method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "matrix_product_state",
        "max_parallel_threads": 1
    }


if __name__ == '__main__':
    unittest.main()
