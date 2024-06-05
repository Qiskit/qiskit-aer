# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for AerStatevector
"""

import unittest
import logging
from itertools import permutations
from ddt import ddt, data

import numpy as np
from numpy.testing import assert_allclose

from qiskit import transpile
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.random import random_unitary, random_statevector, random_pauli
from qiskit.quantum_info.states import Statevector
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Kraus
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.visualization.state_visualization import state_to_latex
from qiskit.circuit.library import QFT, HGate
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from qiskit_aer.noise import pauli_error
from qiskit_aer.quantum_info.states import AerStatevector

from test.terra import common


logger = logging.getLogger(__name__)


@ddt
class TestAerStatevector(common.QiskitAerTestCase):
    """AerState tests"""

    def test_qv(self):
        """Test generation of Aer's StateVector with QV"""
        circ = QuantumVolume(5, seed=1111)
        state = AerStatevector(circ)
        expected = Statevector(circ)

        for e, s in zip(expected, state):
            self.assertAlmostEqual(e, s)

    def test_sample_randomness(self):
        """Test randomness of results of sample_counts"""
        circ = QuantumVolume(5, seed=1111)

        state = AerStatevector(circ, seed_simulator=1)

        shots = 1024
        counts0 = state.sample_counts(shots, qargs=range(5))
        counts1 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts0, counts1)

        state = AerStatevector(circ, seed_simulator=10)

        shots = 1024
        counts2 = state.sample_counts(shots, qargs=range(5))
        counts3 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts2, counts3)

        self.assertNotEqual(counts0, counts2)
        self.assertNotEqual(counts1, counts2)

    def test_sample_with_same_seed(self):
        """Test randomness of results of sample_counts"""
        circ = QuantumVolume(5, seed=1111)

        state = AerStatevector(circ, seed_simulator=1)

        shots = 1024
        counts0 = state.sample_counts(shots, qargs=range(5))
        counts1 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts0, counts1)

        state = AerStatevector(circ, seed_simulator=1)

        shots = 1024
        counts2 = state.sample_counts(shots, qargs=range(5))
        counts3 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts2, counts3)
        self.assertEqual(counts0, counts2)
        self.assertEqual(counts1, counts3)

        shots = 1024
        state.seed(1)
        counts4 = state.sample_counts(shots, qargs=range(5))
        counts5 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts4, counts5)
        self.assertEqual(counts0, counts4)
        self.assertEqual(counts1, counts5)

    def test_method_and_device_properties(self):
        """Test method and device properties"""
        circ = QuantumVolume(5, seed=1111)
        state1 = AerStatevector(circ)

        self.assertEqual("statevector", state1.metadata()["method"])
        self.assertEqual("CPU", state1.metadata()["device"])

        state2 = AerStatevector(circ, method="matrix_product_state")
        self.assertEqual("matrix_product_state", state2.metadata()["method"])
        self.assertEqual("CPU", state2.metadata()["device"])

        self.assertEqual(state1, state2)

    def test_GHZ(self):
        """Test each method can process ghz"""
        ghz = QuantumCircuit(4)
        ghz.h(0)
        ghz.cx(0, 1)
        ghz.cx(1, 2)
        ghz.cx(2, 3)

        for method in ["statevector", "matrix_product_state"]:
            sv = AerStatevector(ghz, method=method)
            counts = sv.sample_counts(shots=1024)
            self.assertEqual(2, len(counts))
            self.assertTrue("0000" in counts)
            self.assertTrue("1111" in counts)

    def test_QFT(self):
        """Test each method can process qft"""
        qft = QuantumCircuit(4)
        qft.h(range(4))
        qft.compose(QFT(4), inplace=True)

        for method in ["statevector", "matrix_product_state"]:
            sv = AerStatevector(qft, method=method)
            counts = sv.sample_counts(shots=1024)
            self.assertEqual(1, len(counts))
            self.assertTrue("0000" in counts)

    def test_two_qubit_QV(self):
        """Test two qubit QuantumVolume"""
        state = AerStatevector(QuantumVolume(2))
        counts = state.sample_counts(shots=1024)
        self.assertEqual(sum(counts.values()), 1024)

    def test_evolve(self):
        """Test method and device properties"""
        circ1 = QuantumVolume(5, seed=1111)
        circ2 = circ1.compose(circ1)
        circ3 = circ2.compose(circ1)

        state1 = AerStatevector(circ1)
        state2 = AerStatevector(circ2)
        state3 = AerStatevector(circ3)

        self.assertEqual(state1.evolve(circ1), state2)
        self.assertEqual(state1.evolve(circ1).evolve(circ1), state3)

    def test_decompose(self):
        """Test basic gates can be decomposed correctly"""
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.x(1)
        circ.ry(np.pi / 2, 2)
        state1 = AerStatevector(circ)

    def test_ry(self):
        # Test ry
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        psi = AerStatevector.from_instruction(circuit)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        self.assertEqual(target, psi)

    def test_u(self):
        # Test u
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.u(0.1, 0.1, 0.1, 0)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_cu(self):
        # Test cu
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.cu(0.1, 0.1, 0.1, 0.1, 0, 1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_h(self):
        # Test h
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_x(self):
        # Test x
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_cx(self):
        # Test cx
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_y(self):
        # Test y
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.y(1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_cy(self):
        # Test cy
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cy(0, 1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_z(self):
        # Test z
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.z(0)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_cz(self):
        # Test cz
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cz(0, 1)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_unitary(self):
        # Test unitary
        circuit = QuantumCircuit(3)
        mat = random_unitary(8).data
        circuit.unitary(mat, range(3))
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_diagonal(self):
        # Test diagonal
        circuit = QuantumCircuit(3)
        circuit.h(range(3))
        diagonal = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
        circuit.append(DiagonalGate(diagonal), list(range(3)))
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_kraus(self):
        """Test kraus"""

        def same_1qubit_statevectors_up_to_global_phase(sv1, sv2):
            if not isinstance(sv1, np.ndarray):
                sv1 = sv1.data
            if not isinstance(sv2, np.ndarray):
                sv2 = sv2.data
            return np.isclose(np.cross(sv1, sv2), 0)

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.y(0)
        p_error = 0.5
        error = pauli_error([("Y", p_error), ("I", 1 - p_error)])
        circuit.append(Kraus(error).to_instruction(), [0])

        circuit_no_noise = QuantumCircuit(1)
        circuit_no_noise.h(0)
        circuit_no_noise.y(0)
        target0 = AerStatevector.from_label("0").evolve(Operator(circuit_no_noise))
        circuit_no_noise.y(0)
        target1 = AerStatevector.from_label("0").evolve(Operator(circuit_no_noise))

        for i in range(100):
            psi = AerStatevector(circuit, seed_simulator=i)
            self.assertTrue(
                same_1qubit_statevectors_up_to_global_phase(psi, target0)
                or same_1qubit_statevectors_up_to_global_phase(psi, target1)
            )

    def test_deepcopy(self):
        """Test deep copy"""
        import copy

        circ1 = QuantumVolume(5, seed=1111)

        state1 = AerStatevector(circ1)
        state2 = copy.deepcopy(state1)

        for pa1, pa2 in zip(state1, state2):
            self.assertAlmostEqual(pa1, pa2)

        self.assertNotEqual(id(state1._data), id(state2._data))

    def test_initialize_with_ndarray(self):
        """Test ndarray initialization"""
        circ = QuantumVolume(5, seed=1111)
        expected = Statevector(circ)
        state = AerStatevector(expected.data)

        for e, s in zip(expected, state):
            self.assertAlmostEqual(e, s)

    def test_initialize_with_non_contiguous_ndarray(self):
        """Test ndarray initialization"""

        for n_qubits in [2, 4, 8, 16]:
            u = random_unitary(n_qubits, seed=1111).data
            vec = u[0, :]
            state = AerStatevector(vec)

            self.assertTrue(np.allclose(state, vec, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT))

    def test_initialize_with_terra_statevector(self):
        """Test Statevector initialization"""
        circ = QuantumVolume(5, seed=1111)
        expected = Statevector(circ)
        state = AerStatevector(expected)

        for e, s in zip(expected, state):
            self.assertAlmostEqual(e, s)

    def test_initialize_with_aer_statevector(self):
        """Test AerStatevector initialization"""
        circ = QuantumVolume(5, seed=1111)
        expected = AerStatevector(circ)
        state = AerStatevector(expected)

        for e, s in zip(expected, state):
            self.assertAlmostEqual(e, s)

    ####                                          ####
    ###   Copy from test_statevector.py in terra   ###
    ####                                          ####
    @classmethod
    def rand_vec(cls, n, normalize=False):
        """Return complex vector or statevector"""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_vec default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        vec = rng.random(n) + 1j * rng.random(n)
        if normalize:
            vec /= np.sqrt(np.dot(vec, np.conj(vec)))
        return vec

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        vec = self.rand_vec(8)
        for dims in [None, 8]:
            state = AerStatevector(vec, dims=dims)
            assert_allclose(state.data, vec)
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            self.assertEqual(state.num_qubits, 3)

    def test_init_circuit(self):
        """Test initialization from circuit."""
        circuit = QuantumCircuit(3)
        circuit.x(0)
        state = AerStatevector(circuit)

        self.assertEqual(state.dim, 8)
        self.assertEqual(state.dims(), (2, 2, 2))
        self.assertTrue(all(state.data == np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)))
        self.assertEqual(state.num_qubits, 3)

    def test_init_array_except(self):
        """Test initialization exception from array."""
        vec = self.rand_vec(4)
        self.assertRaises(QiskitError, AerStatevector, vec, dims=[4, 2])
        self.assertRaises(QiskitError, AerStatevector, vec, dims=[2, 4])
        self.assertRaises(QiskitError, AerStatevector, vec, dims=5)

    def test_init_statevector(self):
        """Test initialization from AerStatevector."""
        vec1 = AerStatevector(self.rand_vec(4))
        vec2 = AerStatevector(vec1)
        self.assertEqual(vec1, vec2)

    def test_from_circuit(self):
        """Test initialization from a circuit."""
        # random unitaries
        u0 = random_unitary(2).data
        u1 = random_unitary(2).data
        # add to circuit
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.unitary(u0, [qr[0]])
        circ.unitary(u1, [qr[1]])
        target = AerStatevector(np.kron(u1, u0).dot([1, 0, 0, 0]))
        vec = AerStatevector.from_instruction(circ)
        self.assertEqual(vec, target)

        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cp(lam, 0, 1)
        target = AerStatevector.from_label("00").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circ.x(0)
        circuit.ch(0, 1)
        target = AerStatevector.from_label("00").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test custom controlled gate
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        gate = qc.to_gate()
        gate_ctrl = gate.control()

        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.append(gate_ctrl, range(3))
        target = AerStatevector.from_label("000").evolve(Operator(circuit))
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test initialize instruction
        target = AerStatevector([1, 0, 0, 1j]) / np.sqrt(2)
        circuit = QuantumCircuit(2)
        circuit.initialize(target.data, [0, 1])
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test reset instruction
        target = AerStatevector([1, 0])
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.reset(0)
        psi = AerStatevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_from_instruction(self):
        """Test initialization from an instruction."""
        target = np.dot(HGate().to_matrix(), [1, 0])
        vec = AerStatevector.from_instruction(HGate()).data
        global_phase_equivalent = matrix_equal(vec, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_label(self):
        """Test initialization from a label"""
        x_p = AerStatevector(np.array([1, 1]) / np.sqrt(2))
        x_m = AerStatevector(np.array([1, -1]) / np.sqrt(2))
        y_p = AerStatevector(np.array([1, 1j]) / np.sqrt(2))
        y_m = AerStatevector(np.array([1, -1j]) / np.sqrt(2))
        z_p = AerStatevector(np.array([1, 0]))
        z_m = AerStatevector(np.array([0, 1]))

        label = "01"
        target = z_p.tensor(z_m)
        self.assertEqual(target, AerStatevector.from_label(label))

        label = "+-"
        target = x_p.tensor(x_m)
        self.assertEqual(target, AerStatevector.from_label(label))

        label = "rl"
        target = y_p.tensor(y_m)
        self.assertEqual(target, AerStatevector.from_label(label))

    def test_equal(self):
        """Test __eq__ method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            self.assertEqual(AerStatevector(vec), AerStatevector(vec.tolist()))

    def test_getitem(self):
        """Test __getitem__ method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = AerStatevector(vec)
            for i in range(4):
                self.assertEqual(state[i], vec[i])
                self.assertEqual(state[format(i, "b")], vec[i])

    def test_getitem_except(self):
        """Test __getitem__ method raises exceptions."""
        for i in range(1, 4):
            state = AerStatevector(self.rand_vec(2**i))
            self.assertRaises(QiskitError, state.__getitem__, 2**i)
            self.assertRaises(QiskitError, state.__getitem__, -1)

    def test_copy(self):
        """Test AerStatevector copy method"""
        for _ in range(5):
            vec = self.rand_vec(4)
            orig = AerStatevector(vec)
            cpy = orig.copy()
            cpy._data[0] += 1.0
            self.assertFalse(cpy == orig)

    def test_is_valid(self):
        """Test is_valid method."""
        state = AerStatevector([1, 1])
        self.assertFalse(state.is_valid())
        for _ in range(10):
            state = AerStatevector(self.rand_vec(4, normalize=True))
            self.assertTrue(state.is_valid())

    def test_to_operator(self):
        """Test to_operator method for returning projector."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = Operator(np.outer(vec, np.conj(vec)))
            op = AerStatevector(vec).to_operator()
            self.assertEqual(op, target)

    def test_evolve(self):
        """Test _evolve method."""
        for _ in range(10):
            op = random_unitary(4)
            vec = self.rand_vec(4)
            target = AerStatevector(np.dot(op.data, vec))
            evolved = AerStatevector(vec).evolve(op)
            self.assertEqual(target, evolved)

    def test_evolve_subsystem(self):
        """Test subsystem _evolve method."""
        # Test evolving single-qubit of 3-qubit system
        for _ in range(5):
            vec = self.rand_vec(8)
            state = AerStatevector(vec)
            op0 = random_unitary(2)
            op1 = random_unitary(2)
            op2 = random_unitary(2)

            # Test evolve on 1-qubit
            op = op0
            op_full = Operator(np.eye(4)).tensor(op)
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0]), target)

            # Evolve on qubit 1
            op_full = Operator(np.eye(2)).tensor(op).tensor(np.eye(2))
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[1]), target)

            # Evolve on qubit 2
            op_full = op.tensor(np.eye(4))
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2]), target)

            # Test evolve on 2-qubits
            op = op1.tensor(op0)

            # Evolve on qubits [0, 2]
            op_full = op1.tensor(np.eye(2)).tensor(op0)
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 2]), target)

            # Evolve on qubits [2, 0]
            op_full = op0.tensor(np.eye(2)).tensor(op1)
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 0]), target)

            # Test evolve on 3-qubits
            op = op2.tensor(op1).tensor(op0)

            # Evolve on qubits [0, 1, 2]
            op_full = op
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 1, 2]), target)

            # Evolve on qubits [2, 1, 0]
            op_full = op0.tensor(op1).tensor(op2)
            target = AerStatevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 1, 0]), target)

    def test_evolve_global_phase(self):
        """Test evolve circuit with global phase."""
        state_i = AerStatevector([1, 0])
        qr = QuantumRegister(2)
        phase = np.pi / 4
        circ = QuantumCircuit(qr, global_phase=phase)
        circ.x(0)
        state_f = state_i.evolve(circ, qargs=[0])
        target = AerStatevector([0, 1]) * np.exp(1j * phase)

        self.assertEqual(state_f, target)

    def test_conjugate(self):
        """Test conjugate method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = AerStatevector(np.conj(vec))
            state = AerStatevector(vec).conjugate()
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(4)
            target = np.kron(vec1, vec0)
            state = AerStatevector(vec0).expand(AerStatevector(vec1))
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            assert_allclose(state.data, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(4)
            target = np.kron(vec0, vec1)
            state = AerStatevector(vec0).tensor(AerStatevector(vec1))
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            assert_allclose(state.data, target)

    def test_inner(self):
        """Test inner method."""
        for _ in range(10):
            vec0 = AerStatevector(self.rand_vec(4))
            vec1 = AerStatevector(self.rand_vec(4))
            target = np.vdot(vec0.data, vec1.data)
            result = vec0.inner(vec1)
            self.assertAlmostEqual(result, target)

    def test_add(self):
        """Test add method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = AerStatevector(vec0)
            state1 = AerStatevector(vec1)
            self.assertEqual(state0 + state1, AerStatevector(vec0 + vec1))

    def test_add_except(self):
        """Test add method raises exceptions."""
        state1 = AerStatevector(self.rand_vec(2))
        state2 = AerStatevector(self.rand_vec(4))
        self.assertRaises(QiskitError, state1.__add__, state2)

    def test_subtract(self):
        """Test subtract method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = AerStatevector(vec0)
            state1 = AerStatevector(vec1)
            self.assertEqual(state0 - state1, AerStatevector(vec0 - vec1))

    def test_multiply(self):
        """Test multiply method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = AerStatevector(vec)
            val = np.random.rand() + 1j * np.random.rand()
            self.assertEqual(val * state, AerStatevector(val * state))

    def test_negate(self):
        """Test negate method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = AerStatevector(vec)
            self.assertEqual(-state, AerStatevector(-1 * vec))

    def test_equiv(self):
        """Test equiv method"""
        vec = np.array([1, 0, 0, -1j]) / np.sqrt(2)
        phase = np.exp(-1j * np.pi / 4)
        statevec = AerStatevector(vec)
        self.assertTrue(statevec.equiv(phase * vec))
        self.assertTrue(statevec.equiv(AerStatevector(phase * vec)))
        self.assertFalse(statevec.equiv(2 * vec))

    def test_equiv_on_circuit(self):
        """Test the equiv method on different types of input."""
        statevec = AerStatevector([1, 0])

        qc = QuantumCircuit(1)
        self.assertTrue(statevec.equiv(qc))
        qc.x(0)
        self.assertFalse(statevec.equiv(qc))

    def test_to_dict(self):
        """Test to_dict method"""

        with self.subTest(msg="dims = (2, 3)"):
            with self.assertRaises(Exception):
                vec = AerStatevector(np.arange(1, 7), dims=(2, 3))

        with self.subTest(msg="dims = (11, )"):
            with self.assertRaises(Exception):
                vec = AerStatevector(np.arange(1, 12), dims=(11,))

        with self.subTest(msg="dims = (2, 11)"):
            with self.assertRaises(Exception):
                vec = AerStatevector(np.arange(1, 23), dims=(2, 11))

    def test_probabilities_product(self):
        """Test probabilities method for product state"""

        state = AerStatevector.from_label("+0")

        # 2-qubit qargs
        with self.subTest(msg="P(None)"):
            probs = state.probabilities()
            target = np.array([0.5, 0, 0.5, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([0, 1])"):
            probs = state.probabilities([0, 1])
            target = np.array([0.5, 0, 0.5, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([1, 0]"):
            probs = state.probabilities([1, 0])
            target = np.array([0.5, 0.5, 0, 0])
            self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        with self.subTest(msg="P([0])"):
            probs = state.probabilities([0])
            target = np.array([1, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([1])"):
            probs = state.probabilities([1])
            target = np.array([0.5, 0.5])
            self.assertTrue(np.allclose(probs, target))

    def test_probabilities_ghz(self):
        """Test probabilities method for GHZ state"""

        state = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)

        # 3-qubit qargs
        target = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 2-qubit qargs
        target = np.array([0.5, 0, 0, 0.5])
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        target = np.array([0.5, 0.5])
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

    def test_probabilities_w(self):
        """Test probabilities method with W state"""

        state = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)

        # 3-qubit qargs
        target = np.array([0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0])
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 2-qubit qargs
        target = np.array([1 / 3, 1 / 3, 1 / 3, 0])
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        target = np.array([2 / 3, 1 / 3])
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

    def test_probabilities_dict_product(self):
        """Test probabilities_dict method for product state"""

        state = AerStatevector.from_label("+0")

        # 2-qubit qargs
        with self.subTest(msg="P(None)"):
            probs = state.probabilities_dict()
            target = {"00": 0.5, "10": 0.5}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([0, 1])"):
            probs = state.probabilities_dict([0, 1])
            target = {"00": 0.5, "10": 0.5}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([1, 0]"):
            probs = state.probabilities_dict([1, 0])
            target = {"00": 0.5, "01": 0.5}
            self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        with self.subTest(msg="P([0])"):
            probs = state.probabilities_dict([0])
            target = {"0": 1}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([1])"):
            probs = state.probabilities_dict([1])
            target = {"0": 0.5, "1": 0.5}
            self.assertDictAlmostEqual(probs, target)

    def test_probabilities_dict_ghz(self):
        """Test probabilities_dict method for GHZ state"""

        state = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)

        # 3-qubit qargs
        target = {"000": 0.5, "111": 0.5}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 2-qubit qargs
        target = {"00": 0.5, "11": 0.5}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        target = {"0": 0.5, "1": 0.5}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

    def test_probabilities_dict_w(self):
        """Test probabilities_dict method with W state"""

        state = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)

        # 3-qubit qargs
        target = np.array([0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0])
        target = {"001": 1 / 3, "010": 1 / 3, "100": 1 / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 2-qubit qargs
        target = {"00": 1 / 3, "01": 1 / 3, "10": 1 / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        target = {"0": 2 / 3, "1": 1 / 3}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

    def test_sample_counts_ghz(self):
        """Test sample_counts method for GHZ state"""

        shots = 2000
        threshold = 0.02 * shots
        state = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        state.seed(100)

        # 3-qubit qargs
        target = {"000": shots / 2, "111": shots / 2}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 2-qubit qargs
        target = {"00": shots / 2, "11": shots / 2}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 1-qubit qargs
        target = {"0": shots / 2, "1": shots / 2}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

    def test_sample_counts_w(self):
        """Test sample_counts method for W state"""
        shots = 6000
        threshold = 0.02 * shots
        state = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)
        state.seed(100)

        target = {"001": shots / 3, "010": shots / 3, "100": shots / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 2-qubit qargs
        target = {"00": shots / 3, "01": shots / 3, "10": shots / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 1-qubit qargs
        target = {"0": 2 * shots / 3, "1": shots / 3}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

    def test_sample_memory_ghz(self):
        """Test sample_memory method for GHZ state"""

        shots = 5000
        state = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        state.seed(100)

        # 3-qubit qargs
        target = {"000": shots / 2, "111": shots / 2}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {"00": shots / 2, "11": shots / 2}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": shots / 2, "1": shots / 2}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

    def test_sample_memory_w(self):
        """Test sample_memory method for W state"""
        shots = 3000
        state = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)
        state.seed(100)

        target = {"001": shots / 3, "010": shots / 3, "100": shots / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {"00": shots / 3, "01": shots / 3, "10": shots / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": 2 * shots / 3, "1": shots / 3}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

    def test_reset_2qubit(self):
        """Test reset method for 2-qubit state"""

        state = AerStatevector(np.array([1, 0, 0, 1]) / np.sqrt(2))
        state.seed(100)

        with self.subTest(msg="reset"):
            psi = state.copy()
            value = psi.reset()
            target = AerStatevector(np.array([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset"):
            psi = state.copy()
            value = psi.reset([0, 1])
            target = AerStatevector(np.array([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset [0]"):
            psi = state.copy()
            value = psi.reset([0])
            targets = [
                AerStatevector(np.array([1, 0, 0, 0])),
                AerStatevector(np.array([0, 0, 1, 0])),
            ]
            self.assertIn(value, targets)

        with self.subTest(msg="reset [0]"):
            psi = state.copy()
            value = psi.reset([1])
            targets = [
                AerStatevector(np.array([1, 0, 0, 0])),
                AerStatevector(np.array([0, 1, 0, 0])),
            ]
            self.assertIn(value, targets)

    def test_measure_2qubit(self):
        """Test measure method for 2-qubit state"""

        state = AerStatevector.from_label("+0")
        seed = 200
        shots = 100

        with self.subTest(msg="measure"):
            for i in range(shots):
                psi = state.copy()
                psi.seed(seed + i)
                outcome, value = psi.measure()
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = AerStatevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerStatevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [0, 1]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([0, 1])
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = AerStatevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerStatevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [1, 0]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([1, 0])
                self.assertIn(outcome, ["00", "01"])
                if outcome == "00":
                    target = AerStatevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerStatevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [0]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([0])
                self.assertEqual(outcome, "0")
                target = AerStatevector(np.array([1, 0, 1, 0]) / np.sqrt(2))
                self.assertEqual(value, target)

        with self.subTest(msg="measure [1]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([1])
                self.assertIn(outcome, ["0", "1"])
                if outcome == "0":
                    target = AerStatevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerStatevector.from_label("10")
                    self.assertEqual(value, target)

    def test_from_int(self):
        """Test from_int method"""

        with self.subTest(msg="from_int(0, 4)"):
            target = AerStatevector([1, 0, 0, 0])
            value = AerStatevector.from_int(0, 4)
            self.assertEqual(target, value)

        with self.subTest(msg="from_int(3, 4)"):
            target = AerStatevector([0, 0, 0, 1])
            value = AerStatevector.from_int(3, 4)
            self.assertEqual(target, value)

    def test_expval(self):
        """Test expectation_value method"""

        psi = AerStatevector([1, 0, 0, 1]) / np.sqrt(2)
        for label, target in [
            ("II", 1),
            ("XX", 1),
            ("YY", -1),
            ("ZZ", 1),
            ("IX", 0),
            ("YZ", 0),
            ("ZX", 0),
            ("YI", 0),
        ]:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = psi.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        psi = AerStatevector([np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 + 1j]) / 2
        for label, target in [
            ("XXX", np.sqrt(2) / 2),
            ("YYY", -np.sqrt(2) / 2),
            ("ZZZ", 0),
            ("XYZ", 0),
            ("YIY", 0),
        ]:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = psi.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        labels = ["XXX", "IXI", "YYY", "III"]
        coeffs = [3.0, 5.5, -1j, 23]
        spp_op = SparsePauliOp.from_list(list(zip(labels, coeffs)))
        expval = psi.expectation_value(spp_op)
        target = 25.121320343559642 + 0.7071067811865476j
        self.assertAlmostEqual(expval, target)

    @data(
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
        "-II",
        "-IX",
        "-IY",
        "-IZ",
        "-XI",
        "-XX",
        "-XY",
        "-XZ",
        "-YI",
        "-YX",
        "-YY",
        "-YZ",
        "-ZI",
        "-ZX",
        "-ZY",
        "-ZZ",
        "iII",
        "iIX",
        "iIY",
        "iIZ",
        "iXI",
        "iXX",
        "iXY",
        "iXZ",
        "iYI",
        "iYX",
        "iYY",
        "iYZ",
        "iZI",
        "iZX",
        "iZY",
        "iZZ",
        "-iII",
        "-iIX",
        "-iIY",
        "-iIZ",
        "-iXI",
        "-iXX",
        "-iXY",
        "-iXZ",
        "-iYI",
        "-iYX",
        "-iYY",
        "-iYZ",
        "-iZI",
        "-iZX",
        "-iZY",
        "-iZZ",
    )
    def test_expval_pauli(self, pauli):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = Pauli(pauli)
        state = random_statevector(2**op.num_qubits, seed=seed)
        state = AerStatevector(state.data)
        target = state.expectation_value(op.to_matrix())
        expval = state.expectation_value(op)
        self.assertAlmostEqual(expval, target)

    @data([0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1])
    def test_expval_pauli_qargs(self, qubits):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = random_pauli(2, seed=seed)
        state = random_statevector(2**3, seed=seed)
        state = AerStatevector(state.data)
        target = state.expectation_value(op.to_matrix(), qubits)
        expval = state.expectation_value(op, qubits)
        self.assertAlmostEqual(expval, target)

    @data(*(qargs for i in range(4) for qargs in permutations(range(4), r=i + 1)))
    def test_probabilities_qargs(self, qargs):
        """Test probabilities method with qargs"""
        # Get initial state
        nq = 4
        nc = len(qargs)
        state_circ = QuantumCircuit(nq, nc)
        for i in range(nq):
            state_circ.ry((i + 1) * np.pi / (nq + 1), i)

        # Get probabilities
        state = AerStatevector(state_circ)
        probs = state.probabilities(qargs)

        # Estimate target probs from simulator measurement
        terra_state = Statevector(state_circ)
        terra_probs = terra_state.probabilities(qargs)
        assert_allclose(probs, terra_probs)

    def test_global_phase(self):
        """Test global phase is handled correctly when evolving statevector."""

        qc = QuantumCircuit(1)
        qc.rz(0.5, 0)
        qc2 = transpile(qc, basis_gates=["p"])
        sv = AerStatevector.from_instruction(qc2)
        expected = np.array([0.96891242 - 0.24740396j, 0])
        self.assertEqual(float(qc2.global_phase), 2 * np.pi - 0.25)
        self.assertEqual(sv, AerStatevector(expected))

    def test_reverse_qargs(self):
        """Test reverse_qargs method"""
        circ1 = QFT(5)
        circ2 = circ1.reverse_bits()

        state1 = AerStatevector.from_instruction(circ1)
        state2 = AerStatevector.from_instruction(circ2)
        self.assertEqual(state1.reverse_qargs(), state2)

    def test_drawings(self):
        """Test draw method"""
        qc1 = QFT(5)
        sv = AerStatevector.from_instruction(qc1)
        with self.subTest(msg="str(statevector)"):
            str(sv)
        for drawtype in ["repr", "text", "latex", "latex_source", "qsphere", "hinton", "bloch"]:
            with self.subTest(msg=f"draw('{drawtype}')"):
                sv.draw(drawtype)
        with self.subTest(msg=" draw('latex', convention='vector')"):
            sv.draw("latex", convention="vector")

    def test_state_to_latex_for_large_statevector(self):
        """Test conversion of large dense state vector"""
        sv = AerStatevector(np.ones((2**15, 1)))
        latex_representation = state_to_latex(sv)
        self.assertEqual(
            latex_representation,
            " |000000000000000\\rangle+ |000000000000001\\rangle+ |000000000000010\\rangle+"
            " |000000000000011\\rangle+ |000000000000100\\rangle+ |000000000000101\\rangle +"
            " \\ldots + |111111111111011\\rangle+ |111111111111100\\rangle+"
            " |111111111111101\\rangle+ |111111111111110\\rangle+ |111111111111111\\rangle",
        )

    def test_state_to_latex_for_large_sparse_statevector(self):
        """Test conversion of large sparse state vector"""
        sv = AerStatevector(np.eye(2**15, 1))
        latex_representation = state_to_latex(sv)
        self.assertEqual(latex_representation, " |000000000000000\\rangle")

    def test_statevector_draw_latex_regression(self):
        """Test numerical rounding errors are not printed"""
        sv = AerStatevector(np.array([1 - 8e-17, 8.32667268e-17j]))
        latex_string = sv.draw(output="latex_source")
        self.assertTrue(latex_string.startswith(" |0\\rangle"))
        self.assertNotIn("|1\\rangle", latex_string)

    def test_statevector_len(self):
        """Test state vector length"""
        empty_vector = []
        with self.assertRaises(Exception):
            empty_sv = AerStatevector([])


if __name__ == "__main__":
    unittest.main()
