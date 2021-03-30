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

# Save data tests
from test.terra.backends.qasm_simulator.qasm_save import QasmSaveDataTests
# chunk tests
from test.terra.backends.qasm_simulator.qasm_chunk import QasmChunkTests

class StatevectorChunkTests(
        QasmSaveDataTests,
        QasmChunkTests
        ):
    """Container class of statevector method tests."""
    pass


class TestQasmSimulatorStatevectorChunk(common.QiskitAerTestCase, StatevectorChunkTests):
    """QasmSimulator statevector method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "statevector",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }
    SIMULATOR = QasmSimulator(**BACKEND_OPTS)


@requires_method("qasm_simulator", "statevector_gpu")
class TestQasmSimulatorStatevectorChunkThrustGPU(common.QiskitAerTestCase,
                                            StatevectorChunkTests):
    """QasmSimulator statevector_gpu method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "statevector_gpu",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


@requires_method("qasm_simulator", "statevector_thrust")
class TestQasmSimulatorStatevectorChunkThrustCPU(common.QiskitAerTestCase,
                                            StatevectorChunkTests):
    """QasmSimulator statevector_thrust method tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "statevector_thrust",
        "max_parallel_threads": 1,
        "blocking_enable" : True,
        "blocking_qubits" : 2
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


if __name__ == '__main__':
    unittest.main()
