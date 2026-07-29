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
Integration Tests for SaveExpval instruction
"""

from ddt import ddt
from numpy import allclose
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.compiler import transpile

PAULI2 = [
    "II",
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]


@ddt
class TestSaveExpectationValueTests(SimulatorTestCase):
    """Test SaveExpectationValue instruction."""

    def _test_save_expval(self, circuit, oper, qubits, variance, **options):
        """Test Pauli expval for stabilizer circuit"""
        backend = self.backend(**options)
        label = "expval"

        # Format operator and target value
        circ = circuit.copy()
        oper = qi.Operator(oper)
        state = qi.DensityMatrix(circ)
        expval = state.expectation_value(oper, qubits).real

        if variance:
            var = state.expectation_value(oper**2, qubits).real - expval**2
            target = [expval, var]
            circ.save_expectation_value_variance(oper, qubits, label=label)
        else:
            target = expval
            circ.save_expectation_value(oper, qubits, label=label)

        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(result.success)
        simdata = result.data(0)
        self.assertIn(label, simdata)
        value = simdata[label]
        if variance:
            self.assertTrue(allclose(value, target))
        else:
            self.assertAlmostEqual(value, target)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],
        PAULI2,
    )
    def test_save_expval_bell_pauli(self, method, device, pauli):
        """Test Pauli expval for Bell state circuit"""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],
        PAULI2,
    )
    def test_save_expval_stabilizer_pauli(self, method, device, pauli):
        """Test Pauli expval for stabilizer circuit"""
        SEED = 5832
        circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],
        PAULI2,
    )
    def test_save_expval_var_stabilizer_pauli(self, method, device, pauli):
        """Test Pauli expval_var for stabilizer circuit"""
        SEED = 5832
        circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, True, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
    )
    def test_save_expval_stabilizer_hermitian(self, method, device, qubits):
        """Test expval for stabilizer circuit and Hermitian operator"""
        SEED = 7123
        circ = qi.random_clifford(3, seed=SEED).to_circuit()
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],  # , 'extended_stabilizer'],
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
    )
    def test_save_expval_var_stabilizer_hermitian(self, method, device, qubits):
        """Test expval_var for stabilizer circuit and Hermitian operator"""
        SEED = 7123
        circ = qi.random_clifford(3, seed=SEED).to_circuit()
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        self._test_save_expval(circ, oper, qubits, True, method=method, device=device)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"],
        PAULI2,
    )
    def test_save_expval_nonstabilizer_pauli(self, method, device, pauli):
        """Test Pauli expval for non-stabilizer circuit"""
        SEED = 7382
        circ = QuantumVolume(2, 1, seed=SEED)
        oper = qi.Operator(qi.Pauli(pauli))
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"],
        PAULI2,
    )
    def test_save_expval_var_nonstabilizer_pauli(self, method, device, pauli):
        """Test Pauli expval_var for non-stabilizer circuit"""
        SEED = 7382
        circ = QuantumVolume(2, 1, seed=SEED)
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, True, method=method, device=device)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"],
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
    )
    def test_save_expval_nonstabilizer_hermitian(self, method, device, qubits):
        """Test expval for non-stabilizer circuit and Hermitian operator"""
        SEED = 8124
        circ = QuantumVolume(3, 1, seed=SEED)
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"],
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
    )
    def test_save_expval_var_nonstabilizer_hermitian(self, method, device, qubits):
        """Test expval_var for non-stabilizer circuit and Hermitian operator"""
        SEED = 8124
        circ = QuantumVolume(3, 1, seed=SEED)
        oper = qi.random_hermitian(4, traceless=True, seed=SEED)
        self._test_save_expval(circ, oper, qubits, True, method=method, device=device)

    @supported_methods(["density_matrix", "tensor_network"], PAULI2)
    def test_save_expval_cptp_pauli(self, method, device, pauli):
        """Test Pauli expval for stabilizer circuit"""
        SEED = 5832
        oper = qi.Operator(qi.Pauli(pauli))
        channel = qi.random_quantum_channel(4, seed=SEED)
        circ = QuantumCircuit(2)
        circ.append(channel, range(2))
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(["density_matrix", "tensor_network"], PAULI2)
    def test_save_expval_var_cptp_pauli(self, method, device, pauli):
        """Test Pauli expval_var for stabilizer circuit"""
        SEED = 5832
        oper = qi.Operator(qi.Operator(qi.Pauli(pauli)))
        channel = qi.random_quantum_channel(4, seed=SEED)
        circ = QuantumCircuit(2)
        circ.append(channel, range(2))
        qubits = [0, 1]
        self._test_save_expval(circ, oper, qubits, True, method=method, device=device)

    @supported_methods(["statevector", "density_matrix"], PAULI2)
    def test_save_expval_stabilizer_pauli_cache_blocking(self, method, device, pauli):
        """Test Pauli expval for stabilizer circuit"""
        SEED = 5832
        circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(
            circ,
            oper,
            qubits,
            False,
            method=method,
            device=device,
            blocking_qubits=2,
            max_parallel_threads=1,
        )

    @supported_methods(["statevector", "density_matrix"], PAULI2)
    def test_save_expval_var_stabilizer_pauli_cache_blocking(self, method, device, pauli):
        """Test Pauli expval_var for stabilizer circuit"""
        SEED = 5832
        circ = qi.random_clifford(2, seed=SEED).to_circuit()
        oper = qi.Pauli(pauli)
        qubits = [0, 1]
        self._test_save_expval(
            circ,
            oper,
            qubits,
            True,
            method=method,
            device=device,
            blocking_qubits=2,
            max_parallel_threads=1,
        )

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        ],
    )
    def test_save_expval_truncation(self, method, device):
        """Test save expval when qubits are not in use and are truncated"""
        circ = QuantumCircuit(3)
        circ.x(1)
        oper = qi.Pauli("ZZ")
        qubits = [1, 2]  # qubit 0 will be truncated
        self._test_save_expval(circ, oper, qubits, False, method=method, device=device)

    @supported_methods(["statevector"])
    def test_save_expval_truncation_padding(self, method, device):
        """Verify truncation pads Pauli strings with 'I' for removed qubits"""
        # 4-qubit circuit but only qubit 2 is used -> truncation should remove unused qubits
        circ = QuantumCircuit(4)
        circ.x(2)

        # Operator defined over all 4 qubits (will be truncated/padded)
        full_op = qi.SparsePauliOp("ZZZZ")
        qubits = [0, 1, 2, 3]

        base = circ.copy()
        # Expected: equivalent to applying 'ZZ' on qubits [2,3] (unused qubits act as identity)
        expected = qi.Statevector(base).expectation_value(qi.SparsePauliOp("ZZ"), [2, 3]).real

        circ.save_expectation_value(full_op, qubits, label="padded")

        backend = self.backend(method=method, device=device, enable_truncation=True)
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(getattr(result, "success", False))
        simdata = result.data(0)
        self.assertIn("padded", simdata)
        val = simdata["padded"].real

        self.assertAlmostEqual(val, expected)

    @supported_methods(["statevector"])
    def test_save_expval_suffix_truncation(self, method, device):
        """Test that truncation pads/remaps pauli strings (non-suffix kept qubits)."""
        # 5-qubit circuit but only qubits 0 and 2 are used -> truncation should
        # remove unused qubits and remap pauli string accordingly. A naive
        # suffix-trim would take the last 2 chars, which is incorrect here.
        circ = QuantumCircuit(5)
        circ.x(0)
        circ.x(2)

        # Operator defined over all 5 qubits with non-identity on qubits 0 and 2
        full_op = qi.SparsePauliOp("ZIZII")
        qubits = [0, 1, 2, 3, 4]

        # Reference expectation computed directly on the full state
        expected = qi.Statevector(circ).expectation_value(full_op, qubits).real

        circ.save_expectation_value(full_op, qubits, label="suffix_test")

        backend = self.backend(method=method, device=device, enable_truncation=True)
        result = backend.run(transpile(circ, backend, optimization_level=0), shots=1).result()
        self.assertTrue(getattr(result, "success", False))
        simdata = result.data(0)
        self.assertIn("suffix_test", simdata)
        val = simdata["suffix_test"].real

        self.assertAlmostEqual(val, expected)

    @supported_methods(["statevector"])
    def test_save_expval_subsystem_consistency(self, method, device):
        """Reproduction test issue 2442: compare Statevector expectation_value with save_expectation_value on a subsystem"""

        circ = QuantumCircuit(12)
        circ.h(0)
        for t in [3, 5, 4, 8, 9, 10]:
            circ.cx(0, t)
        circ.cx(5, 11)
        circ.cx(10, 11)

        # Reference using Statevector
        base = circ.copy()
        truth = qi.Statevector(base).expectation_value(qi.SparsePauliOp("Z"), [11]).real

        # Use backend save_expectation_value on subsystem (label 'sub')
        sub = base.copy()
        sub.save_expectation_value(qi.SparsePauliOp("Z"), [11], label="sub")

        backend = self.backend(method=method, device=device)
        result = backend.run(transpile(sub, backend, optimization_level=0), shots=1).result()
        self.assertTrue(getattr(result, "success", False))
        simdata = result.data(0)
        self.assertIn("sub", simdata)
        sub_ev = simdata["sub"].real

        self.assertAlmostEqual(sub_ev, truth)
