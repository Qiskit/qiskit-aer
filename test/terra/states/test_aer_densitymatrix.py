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
Integration Tests for AerDensityMatrix
"""

import unittest
import numpy as np
import logging
from ddt import ddt, data
from numpy.testing import assert_allclose

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.random import random_unitary, random_density_matrix, random_pauli
from qiskit.quantum_info.states import DensityMatrix, Statevector
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Kraus
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.circuit.library import QFT, HGate

from test.terra import common
from qiskit_aer.aererror import AerError
from qiskit_aer.noise import pauli_error
from qiskit_aer.quantum_info.states import AerDensityMatrix, AerStatevector


logger = logging.getLogger(__name__)


@ddt
class TestAerDensityMatrix(common.QiskitAerTestCase):
    """Tests for AerDensityMatrix class."""

    def test_qv(self):
        """Test generation of Aer's DensityMatrix with QV"""
        circ = QuantumVolume(5, seed=1111)
        state = AerDensityMatrix(circ)
        expected = DensityMatrix(circ)

        self.assertEqual(expected.data.shape, state.data.shape)
        for e, s in zip(expected.data.ravel(), state.data.ravel()):
            self.assertAlmostEqual(e, s)

    def test_sample_randomness(self):
        """Test randomness of results of sample_counts"""
        circ = QuantumVolume(5, seed=1111)

        state = AerDensityMatrix(circ, seed_simulator=1)

        shots = 1024
        counts0 = state.sample_counts(shots, qargs=range(5))
        counts1 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts0, counts1)

        state = AerDensityMatrix(circ, seed_simulator=10)

        shots = 1024
        counts2 = state.sample_counts(shots, qargs=range(5))
        counts3 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts2, counts3)

        self.assertNotEqual(counts0, counts2)
        self.assertNotEqual(counts1, counts2)

    def test_sample_with_same_seed(self):
        """Test randomness of results of sample_counts"""
        circ = QuantumVolume(5, seed=1111)

        state = AerDensityMatrix(circ, seed_simulator=1)

        shots = 1024
        counts0 = state.sample_counts(shots, qargs=range(5))
        counts1 = state.sample_counts(shots, qargs=range(5))

        self.assertNotEqual(counts0, counts1)

        state = AerDensityMatrix(circ, seed_simulator=1)

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

        state1 = AerDensityMatrix(circ)
        self.assertEqual("density_matrix", state1.metadata()["method"])
        self.assertEqual("CPU", state1.metadata()["device"])

        state2 = AerDensityMatrix(circ, method="density_matrix")
        self.assertEqual("density_matrix", state2.metadata()["method"])
        self.assertEqual("CPU", state2.metadata()["device"])

        self.assertEqual(state1, state2)

    def test_GHZ(self):
        """Test each method can process ghz"""
        ghz = QuantumCircuit(4)
        ghz.h(0)
        ghz.cx(0, 1)
        ghz.cx(1, 2)
        ghz.cx(2, 3)

        dm = AerDensityMatrix(ghz)
        counts = dm.sample_counts(shots=1024)
        self.assertEqual(2, len(counts))
        self.assertTrue("0000" in counts)
        self.assertTrue("1111" in counts)

    def test_QFT(self):
        """Test each method can process qft"""
        qft = QuantumCircuit(4)
        qft.h(range(4))
        qft.compose(QFT(4), inplace=True)

        dm = AerDensityMatrix(qft)
        counts = dm.sample_counts(shots=1024)
        self.assertEqual(1, len(counts))
        self.assertTrue("0000" in counts)

    def test_two_qubit_QV(self):
        """Test single qubit QuantumVolume"""
        state = AerDensityMatrix(QuantumVolume(2, seed=1111))
        state.seed(1111)
        counts = state.sample_counts(shots=1024)
        self.assertEqual(sum(counts.values()), 1024)

    def test_evolve(self):
        """Test evolve method for circuits"""
        circ1 = QuantumVolume(5, seed=1111)
        circ2 = circ1.compose(circ1)
        circ3 = circ2.compose(circ1)

        state1 = AerDensityMatrix(circ1)
        state2 = AerDensityMatrix(circ2)
        state3 = AerDensityMatrix(circ3)

        self.assertEqual(state1.evolve(circ1), state2)
        self.assertEqual(state1.evolve(circ1).evolve(circ1), state3)

    def test_decompose(self):
        """Test basic gates can be decomposed correctly"""
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.x(1)
        circ.ry(np.pi / 2, 2)
        state1 = AerDensityMatrix(circ)

    def test_ry(self):
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        psi = AerDensityMatrix.from_instruction(circuit)
        target = AerDensityMatrix.from_label("000").evolve(Operator(circuit))
        self.assertEqual(target, psi)

    def test_h(self):
        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        target = AerDensityMatrix.from_label("000").evolve(Operator(circuit))
        psi = AerDensityMatrix.from_instruction(circuit)
        self.assertEqual(psi, target)

    @data(*list(np.linspace(0, 1, 101)))
    def test_kraus(self, p_error):
        """Test kraus"""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.y(0)
        error = pauli_error([("Y", p_error), ("I", 1 - p_error)])
        circuit.append(Kraus(error).to_instruction(), [0])

        circuit_no_noise = QuantumCircuit(1)
        circuit_no_noise.h(0)
        circuit_no_noise.y(0)
        rho0 = AerDensityMatrix.from_label("0").evolve(Operator(circuit_no_noise))
        circuit_no_noise.y(0)
        rho1 = AerDensityMatrix.from_label("0").evolve(Operator(circuit_no_noise))
        # (1-p_error)|+><+| + p_error|-><-|
        target = AerDensityMatrix(rho0.data * (1 - p_error) + rho1.data * p_error)

        psi = AerDensityMatrix.from_instruction(circuit)
        self.assertTrue(psi == target)

    def test_deepcopy(self):
        """Test deep copy"""
        import copy

        circ1 = QuantumVolume(5, seed=1111)

        state1 = AerDensityMatrix(circ1)
        state2 = copy.deepcopy(state1)

        self.assertEqual(state1.data.shape, state2.data.shape)
        for pa1, pa2 in zip(state1.data.ravel(), state2.data.ravel()):
            self.assertAlmostEqual(pa1, pa2)

        self.assertNotEqual(id(state1._data), id(state2._data))

    def test_initialize_with_ndarray(self):
        """Test ndarray initialization"""
        circ = QuantumVolume(5, seed=1111)
        expected = DensityMatrix(circ)
        state = AerDensityMatrix(expected.data)

        self.assertEqual(expected.data.shape, state.data.shape)
        for e, s in zip(expected.data.ravel(), state.data.ravel()):
            self.assertAlmostEqual(e, s)

    def test_initialize_with_terra_statevector(self):
        """Test Statevector initialization"""
        circ = QuantumVolume(5, seed=1111)
        sv = Statevector(circ)
        expected = np.outer(sv, np.conjugate(sv))
        state = AerDensityMatrix(sv)

        self.assertEqual(expected.shape, state.data.shape)
        for e, s in zip(expected.ravel(), state.data.ravel()):
            self.assertAlmostEqual(e, s)

    def test_initialize_with_statevector(self):
        """Test AerStatevector initialization"""
        circ = QuantumVolume(5, seed=1111)
        sv = AerStatevector(circ)
        expected = np.outer(sv, np.conjugate(sv))
        state = AerDensityMatrix(sv)

        self.assertEqual(expected.shape, state.data.shape)
        for e, s in zip(expected.ravel(), state.data.ravel()):
            self.assertAlmostEqual(e, s)

    def test_initialize_with_densitymatrix(self):
        """Test DensityMatrix initialization"""
        circ = QuantumVolume(5, seed=1111)
        expected = DensityMatrix(circ)
        state = AerDensityMatrix(expected)

        self.assertAlmostEqual(expected.data.shape, state.data.shape)
        for e, s in zip(expected.data.ravel(), state.data.ravel()):
            self.assertAlmostEqual(e, s)

    def test_init_array_qudit(self):
        """Test initialization from array."""
        rho = self.rand_rho(3)
        # qudit is not currently supported
        self.assertRaises(AerError, AerDensityMatrix, rho)

        rho = self.rand_rho(2 * 3 * 4)
        # qudit is not currently supported
        self.assertRaises(AerError, AerDensityMatrix, rho, dims=[2, 3, 4])

    ####                                            ####
    ###   Copy from test_densitymatrix.py in terra   ###
    ####                                            ####
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

    @classmethod
    def rand_rho(cls, n):
        """Return random pure state density matrix"""
        rho = cls.rand_vec(n, normalize=True)
        return np.outer(rho, np.conjugate(rho))

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        rho = self.rand_rho(8)
        for dims in [None, 8]:
            state = AerDensityMatrix(rho, dims=dims)
            assert_allclose(state.data, rho)
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            self.assertEqual(state.num_qubits, 3)

    def test_init_array(self):
        """Test initialization from array."""
        rho = self.rand_rho(4)
        state = AerDensityMatrix(rho)
        assert_allclose(state.data, rho)
        self.assertEqual(state.dim, 4)
        self.assertEqual(state.dims(), (2, 2))
        self.assertEqual(state.num_qubits, 2)

        rho = self.rand_rho(2 * 4 * 8)
        state = AerDensityMatrix(rho, dims=[2, 4, 8])
        assert_allclose(state.data, rho)
        self.assertEqual(state.dim, 2 * 4 * 8)
        self.assertEqual(state.dims(), (2, 4, 8))
        self.assertIsNone(state.num_qubits)

    def test_init_array_except(self):
        """Test initialization exception from array."""
        rho = self.rand_rho(4)
        self.assertRaises(QiskitError, AerDensityMatrix, rho, dims=[4, 2])
        self.assertRaises(QiskitError, AerDensityMatrix, rho, dims=[2, 4])
        self.assertRaises(QiskitError, AerDensityMatrix, rho, dims=5)

    def test_init_densitymatrix(self):
        """Test initialization from AerDensityMatrix."""
        rho1 = AerDensityMatrix(self.rand_rho(4))
        rho2 = AerDensityMatrix(rho1)
        self.assertEqual(rho1, rho2)

    def test_init_statevector(self):
        """Test initialization from AerDensityMatrix."""
        vec = self.rand_vec(4)
        target = AerDensityMatrix(np.outer(vec, np.conjugate(vec)))
        rho = AerDensityMatrix(AerStatevector(vec))
        self.assertEqual(rho, target)

    def test_init_circuit(self):
        """Test initialization from a circuit."""
        # random unitaries
        u0 = random_unitary(2).data
        u1 = random_unitary(2).data
        # add to circuit
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.unitary(u0, [qr[0]])
        circ.unitary(u1, [qr[1]])
        target_vec = AerStatevector(np.kron(u1, u0).dot([1, 0, 0, 0]))
        target = AerDensityMatrix(target_vec)
        rho = AerDensityMatrix(circ)
        self.assertEqual(rho, target)

        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        target = AerDensityMatrix.from_label("000").evolve(Operator(circuit))
        rho = AerDensityMatrix(circuit)
        self.assertEqual(rho, target)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cp(lam, 0, 1)
        target = AerDensityMatrix.from_label("00").evolve(Operator(circuit))
        rho = AerDensityMatrix(circuit)
        self.assertEqual(rho, target)

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

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circ.x(0)
        circuit.ch(0, 1)
        target = AerDensityMatrix.from_label("00").evolve(Operator(circuit))
        rho = AerDensityMatrix.from_instruction(circuit)
        self.assertEqual(rho, target)

        # Test initialize instruction
        init = AerStatevector([1, 0, 0, 1j]) / np.sqrt(2)
        target = AerDensityMatrix(init)
        circuit = QuantumCircuit(2)
        circuit.initialize(init.data, [0, 1])
        rho = AerDensityMatrix.from_instruction(circuit)
        self.assertEqual(rho, target)

        # Test reset instruction
        target = AerDensityMatrix([1, 0])
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.reset(0)
        rho = AerDensityMatrix.from_instruction(circuit)
        self.assertEqual(rho, target)

    def test_from_instruction(self):
        """Test initialization from an instruction."""
        target_vec = AerStatevector(np.dot(HGate().to_matrix(), [1, 0]))
        target = AerDensityMatrix(target_vec)
        rho = AerDensityMatrix.from_instruction(HGate())
        self.assertEqual(rho, target)

    def test_from_label(self):
        """Test initialization from a label"""
        x_p = AerDensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
        x_m = AerDensityMatrix(np.array([[0.5, -0.5], [-0.5, 0.5]]))
        y_p = AerDensityMatrix(np.array([[0.5, -0.5j], [0.5j, 0.5]]))
        y_m = AerDensityMatrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]]))
        z_p = AerDensityMatrix(np.diag([1, 0]))
        z_m = AerDensityMatrix(np.diag([0, 1]))

        label = "0+r"
        target = z_p.tensor(x_p).tensor(y_p)
        self.assertEqual(target, AerDensityMatrix.from_label(label))

        label = "-l1"
        target = x_m.tensor(y_m).tensor(z_m)
        self.assertEqual(target, AerDensityMatrix.from_label(label))

    def test_equal(self):
        """Test __eq__ method"""
        for _ in range(10):
            rho = self.rand_rho(4)
            self.assertEqual(AerDensityMatrix(rho), AerDensityMatrix(rho.tolist()))

    def test_copy(self):
        """Test AerDensityMatrix copy method"""
        for _ in range(5):
            rho = self.rand_rho(4)
            orig = AerDensityMatrix(rho)
            cpy = orig.copy()
            cpy._data[0] += 1.0
            self.assertFalse(cpy == orig)

    def test_is_valid(self):
        """Test is_valid method."""
        state = AerDensityMatrix(np.eye(2))
        self.assertFalse(state.is_valid())
        for _ in range(10):
            state = AerDensityMatrix(self.rand_rho(4))
            self.assertTrue(state.is_valid())

    def test_to_operator(self):
        """Test to_operator method for returning projector."""
        for _ in range(10):
            rho = self.rand_rho(4)
            target = Operator(rho)
            op = AerDensityMatrix(rho).to_operator()
            self.assertEqual(op, target)

    def test_evolve(self):
        """Test evolve method for operators."""
        for _ in range(10):
            op = random_unitary(4)
            rho = self.rand_rho(4)
            target = AerDensityMatrix(np.dot(op.data, rho).dot(op.adjoint().data))
            evolved = AerDensityMatrix(rho).evolve(op)
            self.assertEqual(target, evolved)

    def test_evolve_subsystem(self):
        """Test subsystem evolve method for operators."""
        # Test evolving single-qubit of 3-qubit system
        for _ in range(5):
            rho = self.rand_rho(8)
            state = AerDensityMatrix(rho)
            op0 = random_unitary(2)
            op1 = random_unitary(2)
            op2 = random_unitary(2)

            # Test evolve on 1-qubit
            op = op0
            op_full = Operator(np.eye(4)).tensor(op)
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[0]), target)

            # Evolve on qubit 1
            op_full = Operator(np.eye(2)).tensor(op).tensor(np.eye(2))
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[1]), target)

            # Evolve on qubit 2
            op_full = op.tensor(np.eye(4))
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[2]), target)

            # Test evolve on 2-qubits
            op = op1.tensor(op0)

            # Evolve on qubits [0, 2]
            op_full = op1.tensor(np.eye(2)).tensor(op0)
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[0, 2]), target)

            # Evolve on qubits [2, 0]
            op_full = op0.tensor(np.eye(2)).tensor(op1)
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[2, 0]), target)

            # Test evolve on 3-qubits
            op = op2.tensor(op1).tensor(op0)

            # Evolve on qubits [0, 1, 2]
            op_full = op
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[0, 1, 2]), target)

            # Evolve on qubits [2, 1, 0]
            op_full = op0.tensor(op1).tensor(op2)
            target = AerDensityMatrix(np.dot(op_full.data, rho).dot(op_full.adjoint().data))
            self.assertEqual(state.evolve(op, qargs=[2, 1, 0]), target)

    # omit test_evolve_qudit_subsystems since qudit is currently not supported

    def test_conjugate(self):
        """Test conjugate method."""
        for _ in range(10):
            rho = self.rand_rho(4)
            target = AerDensityMatrix(np.conj(rho))
            state = AerDensityMatrix(rho).conjugate()
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(10):
            rho0 = self.rand_rho(2)
            rho1 = self.rand_rho(4)
            target = np.kron(rho1, rho0)
            state = AerDensityMatrix(rho0).expand(AerDensityMatrix(rho1))
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            assert_allclose(state.data, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(10):
            rho0 = self.rand_rho(2)
            rho1 = self.rand_rho(4)
            target = np.kron(rho0, rho1)
            state = AerDensityMatrix(rho0).tensor(AerDensityMatrix(rho1))
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            assert_allclose(state.data, target)

    def test_add(self):
        """Test add method."""
        for _ in range(10):
            rho0 = self.rand_rho(4)
            rho1 = self.rand_rho(4)
            state0 = AerDensityMatrix(rho0)
            state1 = AerDensityMatrix(rho1)
            self.assertEqual(state0 + state1, AerDensityMatrix(rho0 + rho1))

    def test_add_except(self):
        """Test add method raises exceptions."""
        state1 = AerDensityMatrix(self.rand_rho(2))
        state2 = AerDensityMatrix(self.rand_rho(4))
        self.assertRaises(QiskitError, state1.__add__, state2)

    def test_subtract(self):
        """Test subtract method."""
        for _ in range(10):
            rho0 = self.rand_rho(4)
            rho1 = self.rand_rho(4)
            state0 = AerDensityMatrix(rho0)
            state1 = AerDensityMatrix(rho1)
            self.assertEqual(state0 - state1, AerDensityMatrix(rho0 - rho1))

    def test_multiply(self):
        """Test multiply method."""
        for _ in range(10):
            rho = self.rand_rho(4)
            state = AerDensityMatrix(rho)
            val = np.random.rand() + 1j * np.random.rand()
            self.assertEqual(val * state, AerDensityMatrix(val * state))

    def test_negate(self):
        """Test negate method"""
        for _ in range(10):
            rho = self.rand_rho(4)
            state = AerDensityMatrix(rho)
            self.assertEqual(-state, AerDensityMatrix(-1 * rho))

    def test_to_dict(self):
        """Test to_dict method"""

        with self.subTest(msg="dims = (2, 2)"):
            rho = AerDensityMatrix(np.arange(1, 17).reshape(4, 4))
            target = {
                "00|00": 1,
                "01|00": 2,
                "10|00": 3,
                "11|00": 4,
                "00|01": 5,
                "01|01": 6,
                "10|01": 7,
                "11|01": 8,
                "00|10": 9,
                "01|10": 10,
                "10|10": 11,
                "11|10": 12,
                "00|11": 13,
                "01|11": 14,
                "10|11": 15,
                "11|11": 16,
            }
            self.assertDictAlmostEqual(target, rho.to_dict())

        with self.subTest(msg="dims = (2, 4)"):
            rho = AerDensityMatrix(np.diag(np.arange(1, 9)), dims=(2, 4))
            target = {}
            for i in range(2):
                for j in range(4):
                    key = "{1}{0}|{1}{0}".format(i, j)
                    target[key] = 2 * j + i + 1
            self.assertDictAlmostEqual(target, rho.to_dict())

        with self.subTest(msg="dims = (2, 16)"):
            vec = AerDensityMatrix(np.diag(np.arange(1, 33)), dims=(2, 16))
            target = {}
            for i in range(2):
                for j in range(16):
                    key = "{1},{0}|{1},{0}".format(i, j)
                    target[key] = 2 * j + i + 1
            self.assertDictAlmostEqual(target, vec.to_dict())

    def test_densitymatrix_to_statevector_pure(self):
        """Test converting a pure density matrix to statevector."""
        state = 1 / np.sqrt(2) * (np.array([1, 0, 0, 0, 0, 0, 0, 1]))
        psi = AerStatevector(state)
        rho = AerDensityMatrix(psi)
        phi = rho.to_statevector()
        self.assertTrue(psi.equiv(phi))

    def test_densitymatrix_to_statevector_mixed(self):
        """Test converting a pure density matrix to statevector."""
        state_1 = 1 / np.sqrt(2) * (np.array([1, 0, 0, 0, 0, 0, 0, 1]))
        state_2 = 1 / np.sqrt(2) * (np.array([0, 0, 0, 0, 0, 0, 1, 1]))
        psi = 0.5 * (AerStatevector(state_1) + AerStatevector(state_2))
        rho = AerDensityMatrix(psi)
        self.assertRaises(QiskitError, rho.to_statevector)

    def test_probabilities_product(self):
        """Test probabilities method for product state"""

        state = AerDensityMatrix.from_label("+0")

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

        psi = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        state = AerDensityMatrix(psi)

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

        psi = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)
        state = AerDensityMatrix(psi)

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

        state = AerDensityMatrix.from_label("+0")

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

        psi = (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        state = AerDensityMatrix(psi)

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

        psi = (
            AerStatevector.from_label("001")
            + AerStatevector.from_label("010")
            + AerStatevector.from_label("100")
        ) / np.sqrt(3)
        state = AerDensityMatrix(psi)

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

        shots = 5000
        threshold = 0.02 * shots
        state = AerDensityMatrix(
            (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        )
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
        state = AerDensityMatrix(
            (
                AerStatevector.from_label("001")
                + AerStatevector.from_label("010")
                + AerStatevector.from_label("100")
            )
            / np.sqrt(3)
        )
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

    # omit test_sample_counts_qutrit since qutrit is currently not supported

    def test_sample_memory_ghz(self):
        """Test sample_memory method for GHZ state"""

        shots = 2000
        state = AerDensityMatrix(
            (AerStatevector.from_label("000") + AerStatevector.from_label("111")) / np.sqrt(2)
        )
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
        state = AerDensityMatrix(
            (
                AerStatevector.from_label("001")
                + AerStatevector.from_label("010")
                + AerStatevector.from_label("100")
            )
            / np.sqrt(3)
        )
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

    # omit test_sample_memory_qutrit since qutrit is currently not supported

    def test_reset_2qubit(self):
        """Test reset method for 2-qubit state"""

        state = AerDensityMatrix(np.diag([0.5, 0, 0, 0.5]))

        with self.subTest(msg="reset"):
            rho = state.copy()
            value = rho.reset()
            target = AerDensityMatrix(np.diag([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset"):
            rho = state.copy()
            value = rho.reset([0, 1])
            target = AerDensityMatrix(np.diag([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset [0]"):
            rho = state.copy()
            value = rho.reset([0])
            target = AerDensityMatrix(np.diag([0.5, 0, 0.5, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset [0]"):
            rho = state.copy()
            value = rho.reset([1])
            target = AerDensityMatrix(np.diag([0.5, 0.5, 0, 0]))
            self.assertEqual(value, target)

    # omit test_reset_qutrit since qutrit is currently not supported

    def test_measure_2qubit(self):
        """Test measure method for 2-qubit state"""

        state = AerDensityMatrix.from_label("+0")
        seed = 200
        shots = 100

        with self.subTest(msg="measure"):
            for i in range(shots):
                rho = state.copy()
                rho.seed(seed + i)
                outcome, value = rho.measure()
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = AerDensityMatrix.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerDensityMatrix.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [0, 1]"):
            for i in range(shots):
                rho = state.copy()
                outcome, value = rho.measure([0, 1])
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = AerDensityMatrix.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerDensityMatrix.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [1, 0]"):
            for i in range(shots):
                rho = state.copy()
                outcome, value = rho.measure([1, 0])
                self.assertIn(outcome, ["00", "01"])
                if outcome == "00":
                    target = AerDensityMatrix.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerDensityMatrix.from_label("10")
                    self.assertEqual(value, target)
        with self.subTest(msg="measure [0]"):
            for i in range(shots):
                rho = state.copy()
                outcome, value = rho.measure([0])
                self.assertEqual(outcome, "0")
                target = AerDensityMatrix.from_label("+0")
                self.assertEqual(value, target)

        with self.subTest(msg="measure [1]"):
            for i in range(shots):
                rho = state.copy()
                outcome, value = rho.measure([1])
                self.assertIn(outcome, ["0", "1"])
                if outcome == "0":
                    target = AerDensityMatrix.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = AerDensityMatrix.from_label("10")
                    self.assertEqual(value, target)

    # omit test_measure_qutrit since qutrit is currently not supported

    def test_from_int(self):
        """Test from_int method"""

        with self.subTest(msg="from_int(0, 4)"):
            target = AerDensityMatrix([1, 0, 0, 0])
            value = AerDensityMatrix.from_int(0, 4)
            self.assertEqual(target, value)

        with self.subTest(msg="from_int(3, 4)"):
            target = AerDensityMatrix([0, 0, 0, 1])
            value = AerDensityMatrix.from_int(3, 4)
            self.assertEqual(target, value)

        with self.subTest(msg="from_int(15, (4, 4))"):
            target = AerDensityMatrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dims=(4, 4))
            value = AerDensityMatrix.from_int(15, (4, 4))
            self.assertEqual(target, value)

    def test_expval(self):
        """Test expectation_value method"""

        psi = AerStatevector([1, 0, 0, 1]) / np.sqrt(2)
        rho = AerDensityMatrix(psi)
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
                expval = rho.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        psi = AerStatevector([np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 + 1j]) / 2
        rho = AerDensityMatrix(psi)
        for label, target in [
            ("XXX", np.sqrt(2) / 2),
            ("YYY", -np.sqrt(2) / 2),
            ("ZZZ", 0),
            ("XYZ", 0),
            ("YIY", 0),
        ]:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = rho.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        labels = ["XXX", "IXI", "YYY", "III"]
        coeffs = [3.0, 5.5, -1j, 23]
        spp_op = SparsePauliOp.from_list(list(zip(labels, coeffs)))
        expval = rho.expectation_value(spp_op)
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
    def test_expval_pauli_f_contiguous(self, pauli):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = Pauli(pauli)
        rho = random_density_matrix(2**op.num_qubits, seed=seed)
        rho._data = np.reshape(rho.data.flatten(order="F"), rho.data.shape, order="F")
        target = rho.expectation_value(op.to_matrix())
        expval = rho.expectation_value(op)
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
    def test_expval_pauli_c_contiguous(self, pauli):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = Pauli(pauli)
        rho = random_density_matrix(2**op.num_qubits, seed=seed)
        rho._data = np.reshape(rho.data.flatten(order="C"), rho.data.shape, order="C")
        target = rho.expectation_value(op.to_matrix())
        expval = rho.expectation_value(op)
        self.assertAlmostEqual(expval, target)

    @data([0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1])
    def test_expval_pauli_qargs(self, qubits):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = random_pauli(2, seed=seed)
        state = random_density_matrix(2**3, seed=seed)
        target = state.expectation_value(op.to_matrix(), qubits)
        expval = state.expectation_value(op, qubits)
        self.assertAlmostEqual(expval, target)

    def test_reverse_qargs(self):
        """Test reverse_qargs method"""
        circ1 = QFT(5)
        circ2 = circ1.reverse_bits()

        state1 = AerDensityMatrix.from_instruction(circ1)
        state2 = AerDensityMatrix.from_instruction(circ2)
        self.assertEqual(state1.reverse_qargs(), state2)

    def test_drawings(self):
        """Test draw method"""
        qc1 = QFT(5)
        dm = AerDensityMatrix.from_instruction(qc1)
        with self.subTest(msg="str(density_matrix)"):
            str(dm)
        for drawtype in ["repr", "text", "latex", "latex_source", "qsphere", "hinton", "bloch"]:
            with self.subTest(msg=f"draw('{drawtype}')"):
                dm.draw(drawtype)


if __name__ == "__main__":
    unittest.main()
