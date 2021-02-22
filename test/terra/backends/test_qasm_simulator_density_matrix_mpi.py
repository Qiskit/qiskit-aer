# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021.
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

from test.terra.backends.qasm_simulator.qasm_mpi import QasmMPITests


class DensityMatrixMPITests(QasmMPITests):
    """Container class of statevector method tests."""
    pass


class TestQasmSimulatorDensityMatrixMPI(common.QiskitAerTestCase, DensityMatrixMPITests):
    """QasmSimulator density_matrix method MPI tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "density_matrix",
        "blocking_enable": True,
        "blocking_qubits": 6,
        "max_parallel_threads": 1
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


@requires_method("qasm_simulator", "density_matrix_gpu")
class TestQasmSimulatorDensityMatrixMPIThrustGPU(common.QiskitAerTestCase,
                                                 DensityMatrixMPITests):
    """QasmSimulator density_matrix_gpu method MPI tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "density_matrix_gpu",
        "blocking_enable": True,
        "blocking_qubits": 6,
        "blocking_ignore_diagonal" : True,
        "max_parallel_threads": 1
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


@requires_method("qasm_simulator", "density_matrix_thrust")
class TestQasmSimulatorDensityMatrixMPIThrustCPU(common.QiskitAerTestCase,
                                                 DensityMatrixMPITests):
    """QasmSimulator density_matrix_thrust method MPI tests."""

    BACKEND_OPTS = {
        "seed_simulator": 271828,
        "method": "density_matrix_thrust",
        "blocking_enable": True,
        "blocking_qubits": 6,
        "max_parallel_threads": 1
    }
    try:
        SIMULATOR = QasmSimulator(**BACKEND_OPTS)
    except AerError:
        SIMULATOR = None


if __name__ == '__main__':
    unittest.main()
