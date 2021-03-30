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
from test.terra.decorators import requires_method

# Save data tests
from test.terra.backends.qasm_simulator.qasm_save import QasmSaveDataTests
# chunk tests
from test.terra.backends.qasm_simulator.qasm_chunk import QasmChunkTests

class DensityMatrixChunkTests(
        QasmSaveDataTests,
        QasmChunkTests
        ):
    """Container class of density matrix method tests."""
    pass


class TestQasmSimulatorDensityMatrixChunk(common.QiskitAerTestCase, DensityMatrixChunkTests):
    """QasmSimulator density_matrix method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }


@requires_method("qasm_simulator", "density_matrix_gpu")
class TestQasmSimulatorDensityMatrixChunkThrustGPU(common.QiskitAerTestCase,
                                            DensityMatrixChunkTests):
    """QasmSimulator density_matrix_gpu method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix_gpu",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }


@requires_method("qasm_simulator", "density_matrix_thrust")
class TestQasmSimulatorDensityMatrixChunkThrustCPU(common.QiskitAerTestCase,
                                            DensityMatrixChunkTests):
    """QasmSimulator density_matrix_thrust method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 314159,
        "method": "density_matrix_thrust",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }


if __name__ == '__main__':
    unittest.main()
