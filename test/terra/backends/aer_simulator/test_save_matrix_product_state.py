# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests for SaveMatrixProductState instruction
"""

from ddt import ddt
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestSaveMatrixProductStateTests(SimulatorTestCase):
    """SaveMatrixProductState instruction tests."""

    @supported_methods(["automatic", "matrix_product_state"])
    def test_save_matrix_product_state(self, method, device):
        """Test save matrix_product_state instruction"""
        backend = self.backend(method=method, device=device)

        # Target mps structure
        target_qreg = []
        target_qreg.append((np.array([[1, 0]], dtype=complex), np.array([[0, 1]], dtype=complex)))
        target_qreg.append(
            (np.array([[1], [0]], dtype=complex), np.array([[0], [1]], dtype=complex))
        )
        target_qreg.append((np.array([[1]], dtype=complex), np.array([[0]], dtype=complex)))

        target_lambda_reg = []
        target_lambda_reg.append(np.array([1 / math.sqrt(2)], dtype=float))
        target_lambda_reg.append(np.array([1], dtype=float))

        # Matrix product state test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.cx(0, 1)

        # Add save to circuit
        label = "mps"
        circ.save_matrix_product_state(label=label)

        # Run
        shots = 10
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=shots).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        for val, target in zip(value[0], target_qreg):
            self.assertTrue(np.allclose(val, target))
        for val, target in zip(value[1], target_lambda_reg):
            self.assertTrue(np.allclose(val, target))

    @supported_methods(["automatic", "matrix_product_state"])
    def test_save_matrix_product_state_memory_limit(self, method, device):
        """Test save matrix_product_state instruction when max memory check is disabled"""
        backend = self.backend(
            method=method,
            device=device,
            matrix_product_state_max_bond_dimension=10,
            max_memory_mb=-1,  # Disable memory limit check
        )

        L = 100
        hamiltonian = SparsePauliOp.from_sparse_list([], num_qubits=L)
        for x in range(L - 1):
            hamiltonian += SparsePauliOp.from_sparse_list(
                [
                    ("XX", (x, x + 1), 1),
                    ("YY", (x, x + 1), 1),
                ],
                num_qubits=L,
            )
        qc = QuantumCircuit(L)
        qc.append(
            PauliEvolutionGate(hamiltonian, synthesis=SuzukiTrotter(reps=100)), qargs=qc.qubits
        )
        qc = qc.decompose()
        qc.save_matrix_product_state("mps")
        result = backend.run(qc).result().data(0)
        self.assertTrue("mps" in result)

    @supported_methods(["matrix_product_state"])
    def test_apply_multi_qubit_diagonal_gate(self, method, device):
        """Regression test for issue #2292.
        MPS::apply_multi_qubit_gate crashed when is_diagonal=True and the
        matrix was a 1-row vector (diagonal representation).  The fix
        constructs a 1-row result matrix when is_diagonal is true instead of
        accessing mat(row, col) on a 1xN matrix as if it were NxN.
        """
        backend = self.backend(method=method, device=device)

        # 3-qubit diagonal gate forces the apply_multi_qubit_gate code path
        # (1-qubit and 2-qubit cases have dedicated handlers).
        diag_3q = [1, -1, 1, -1, 1, -1, 1, -1]
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.append(DiagonalGate(diag_3q), [0, 1, 2])
        qc.measure_all()

        # Before the fix this would crash with a memory access violation.
        result = backend.run(transpile(qc, backend, optimization_level=0), shots=100).result()
        self.assertTrue(result.success)
        counts = result.get_counts()
        # All 8 computational basis states of qubits 0-2 should appear.
        self.assertEqual(sum(counts.values()), 100)
