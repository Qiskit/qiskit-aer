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
QasmSimulator Statevector GPU method integration tests
"""

import unittest
from test.terra import common
from test.terra.decorators import requires_gpu

# Basic circuit instruction tests
from test.terra.backends.qasm_simulator.qasm_reset import QasmResetTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMeasureTests
from test.terra.backends.qasm_simulator.qasm_measure import QasmMultiQubitMeasureTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTests
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_cliffords import QasmCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTests
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsWaltzBasis
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsMinimalBasis
from test.terra.backends.qasm_simulator.qasm_unitary_gate import QasmUnitaryGateTests
from test.terra.backends.qasm_simulator.qasm_initialize import QasmInitializeTests
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
from test.terra.backends.qasm_simulator.qasm_snapshot import QasmSnapshotExpValMatrixTests
<<<<<<< HEAD
# Other tests
from test.terra.backends.qasm_simulator.qasm_method import QasmMethodTests
=======
>>>>>>> upstream/pr/544


@requires_gpu
class TestQasmStatevectorSimulator(common.QiskitAerTestCase,
<<<<<<< HEAD
#                                   QasmMethodTests,
=======
>>>>>>> upstream/pr/544
                                   QasmMeasureTests,
                                   QasmMultiQubitMeasureTests,
                                   QasmResetTests,
                                   QasmConditionalGateTests,
                                   QasmConditionalUnitaryTests,
                                   QasmConditionalKrausTests,
                                   QasmCliffordTests,
                                   QasmCliffordTestsWaltzBasis,
                                   QasmCliffordTestsMinimalBasis,
                                   QasmNonCliffordTests,
                                   QasmNonCliffordTestsWaltzBasis,
                                   QasmNonCliffordTestsMinimalBasis,
                                   QasmAlgorithmTests,
                                   QasmAlgorithmTestsWaltzBasis,
                                   QasmAlgorithmTestsMinimalBasis,
                                   QasmUnitaryGateTests,
                                   QasmInitializeTests,
                                   QasmReadoutNoiseTests,
                                   QasmPauliNoiseTests,
                                   QasmResetNoiseTests,
                                   QasmKrausNoiseTests,
                                   QasmSnapshotStatevectorTests,
                                   QasmSnapshotDensityMatrixTests,
                                   QasmSnapshotProbabilitiesTests,
                                   QasmSnapshotExpValPauliTests,
                                   QasmSnapshotExpValMatrixTests,
                                   QasmSnapshotStabilizerTests):
    """QasmSimulator statevector_gpu method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 54321,
        "method": "statevector_gpu"
    }


if __name__ == '__main__':
    unittest.main()
