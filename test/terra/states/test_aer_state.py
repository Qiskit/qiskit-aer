# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for AerState
"""

import unittest
from math import pi
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import random_statevector, random_density_matrix
from qiskit.circuit.library import DiagonalGate

from qiskit_aer import AerSimulator

from test.terra import common
from qiskit_aer.aererror import AerError
from qiskit_aer.backends.controller_wrappers import AerStateWrapper
from qiskit_aer.quantum_info.states.aer_state import AerState


class TestAerState(common.QiskitAerTestCase):
    """AerState tests"""

    def test_generate_aer_state(self):
        """Test generation of AerState"""
        state = AerState()

    def test_move_from_aer_state(self):
        """Test move of aer state to python"""
        state = AerState(method="statevector")
        state.allocate_qubits(4)
        state.initialize()
        sv = state.move_to_ndarray()
        state.close()
        self.assertEqual(len(sv), 2**4)

    def test_error_reuse_aer_state(self):
        """Test reuse AerState after move of aer state to python"""
        state = AerState(method="statevector")
        state.allocate_qubits(4)
        state.initialize()
        sv = state.move_to_ndarray()

        with self.assertRaises(Exception):
            state.allocate_qubits(4)

        state.close()

    def test_initialize_statevector(self):
        """Test initialization of AerState with statevector"""
        state1 = AerState(method="statevector")
        state1.allocate_qubits(4)
        state1.initialize()
        sv1 = state1.move_to_ndarray()
        state1.close()

        self.assertEqual(sv1[0], complex(1.0, 0.0))
        for idx in range(1, len(sv1)):
            self.assertEqual(sv1[idx], complex(0.0, 0.0))

        sv1[0] = complex(0.0, 0.0)
        sv1[len(sv1) - 1] = complex(1.0, 0.0)

        state2 = AerState(method="statevector")
        state2.initialize(sv1)
        state2.flush()
        sv2 = state2.move_to_ndarray()
        state2.close()

        for idx in range(len(sv2) - 2):
            self.assertEqual(sv2[idx], complex(0.0, 0.0))
        self.assertEqual(sv2[len(sv2) - 1], complex(1.0, 0.0))

    def test_initialize_statevector_cache_blocking(self):
        """Test initialization of AerState with statevector"""
        state1 = AerState(blocking_qubits=2, method="statevector")
        state1.allocate_qubits(4)
        state1.initialize()
        sv1 = state1.move_to_ndarray()
        state1.close()

        self.assertEqual(sv1[0], complex(1.0, 0.0))
        for idx in range(1, len(sv1)):
            self.assertEqual(sv1[idx], complex(0.0, 0.0))

        sv1[0] = complex(0.0, 0.0)
        sv1[len(sv1) - 1] = complex(1.0, 0.0)

        state2 = AerState(blocking_qubits=2, method="statevector")
        state2.initialize(sv1)
        state2.flush()
        sv2 = state2.move_to_ndarray()
        state2.close()

        for idx in range(len(sv2) - 2):
            self.assertEqual(sv2[idx], complex(0.0, 0.0))
        self.assertEqual(sv2[len(sv2) - 1], complex(1.0, 0.0))

    def test_initialize_density_matrix(self):
        """Test initialization of AerState with densitymatrix"""
        target_c = random_density_matrix(2**4, seed=1111).data
        target_f = np.array(target_c, order="F")

        state1 = AerState(method="density_matrix")
        state1.initialize(target_f, True)  # copy
        dm1 = state1.move_to_ndarray()
        state1.close()

        self.assertTrue(np.isfortran(dm1))
        self.assertEqual((16, 16), dm1.shape)
        for row in range(dm1.shape[0]):
            for col in range(dm1.shape[1]):
                self.assertAlmostEqual(target_f[row][col], dm1[row][col])

        state2 = AerState(method="density_matrix")
        state2.initialize(dm1, False)
        state2.flush()
        dm2 = state2.move_to_ndarray()
        state2.close()

        self.assertTrue(np.isfortran(dm2))
        self.assertEqual((16, 16), dm2.shape)

        for row in range(dm1.shape[0]):
            for col in range(dm1.shape[1]):
                self.assertAlmostEqual(target_f[row][col], dm2[row][col])

        state3 = AerState(method="density_matrix")
        state3.initialize(target_c, True)  # copy
        dm3 = state3.move_to_ndarray()
        state3.close()

        self.assertTrue(np.isfortran(dm3))
        self.assertEqual((16, 16), dm3.shape)
        for row in range(dm3.shape[0]):
            for col in range(dm3.shape[1]):
                self.assertAlmostEqual(target_f[row][col], dm3[row][col])

    def test_map_statevector(self):
        """Test initialization of AerState with statevector"""
        init_state = random_statevector(2**5, seed=111)
        state1 = AerState(method="statevector", seed_simulator=2222)
        state1.allocate_qubits(4)
        state1.initialize(init_state.data, copy=True)
        sample1 = state1.sample_counts()
        sv1 = state1.move_to_ndarray()

        state2 = AerState(method="statevector", seed_simulator=2222)
        state2.initialize(sv1, copy=False)
        sample2 = state2.sample_counts()
        sv2 = state2.move_to_ndarray()
        state2.close()

        self.assertIs(sv1, sv2)
        self.assertEqual(sample1, sample2)

    def test_map_density_matrix(self):
        """Test initialization of AerState with densitymatrix"""
        init_state = random_statevector(4**4, seed=111).data.reshape(16, 16)
        state1 = AerState(method="density_matrix", seed_simulator=2222)
        state1.initialize(init_state, copy=True)
        sample1 = state1.sample_counts()
        dm1 = state1.move_to_ndarray()

        state2 = AerState(method="density_matrix", seed_simulator=2222)
        state2.initialize(dm1, copy=False)
        sample2 = state2.sample_counts()
        dm2 = state2.move_to_ndarray()
        state2.close()

        self.assertIs(dm1, dm2)
        self.assertEqual(sample1, sample2)

    def test_map_statevector_repeated(self):
        """Test initialization of AerState with statevector"""
        state1 = AerState(method="statevector")
        state1.allocate_qubits(4)
        state1.initialize()
        sv1 = state1.move_to_ndarray()
        sv1[0] = complex(0.0, 0.0)
        sv1[len(sv1) - 1] = complex(1.0, 0.0)
        state1.close()

        for _ in range(100):
            state2 = AerState(method="statevector")
            state2.initialize(sv1, copy=False)
            sv2 = state2.move_to_ndarray()
            state2.close()

            for idx in range(len(sv2) - 2):
                self.assertEqual(sv2[idx], complex(0.0, 0.0))
            self.assertEqual(sv2[len(sv2) - 1], complex(1.0, 0.0))

    def test_map_density_matrix_repeated(self):
        """Test initialization of AerState with densitymatrix"""
        state1 = AerState(method="density_matrix")
        state1.allocate_qubits(4)
        state1.initialize()

        dm1 = state1.move_to_ndarray()
        dm1[0][0] = complex(0.0, 0.0)
        dm1[len(dm1) - 1][len(dm1) - 1] = complex(1.0, 0.0)
        state1.close()

        for _ in range(100):
            state2 = AerState(method="density_matrix")
            state2.initialize(dm1, copy=False)
            dm2 = state2.move_to_ndarray()
            state2.close()

            for row in range(dm2.shape[0]):
                for col in range(dm2.shape[1]):
                    if row == len(dm2) - 1 and col == len(dm2) - 1:
                        self.assertEqual(dm2[row][col], complex(1.0, 0.0))
                    else:
                        self.assertEqual(dm2[row][col], complex(0.0, 0.0))

    def test_initialize_with_normal_ndarray(self):
        """Test initialization of AerState with normal ndarray"""
        sv1 = np.zeros((2**4), dtype=np.complex128)
        sv1[len(sv1) - 1] = 1.0

        state1 = AerState(method="statevector")
        state1.initialize(sv1)

        sv2 = state1.move_to_ndarray()
        self.assertIsNot(sv1, sv2)
        self.assertEqual(len(sv1), len(sv2))
        self.assertEqual(sv1[len(sv1) - 1], sv2[len(sv2) - 1])

        state1.close()

    def test_initialize_with_normal_ndarray_with_map(self):
        """Test initialization of AerState by mapping a normal ndarray"""
        sv1 = np.zeros((2**4), dtype=np.complex128)
        sv1[len(sv1) - 1] = 1.0

        state1 = AerState(method="statevector")
        state1.initialize(sv1, copy=False)

        sv2 = state1.move_to_ndarray()
        self.assertIs(sv1, sv2)

        state1.close()

    def test_initialize_with_non_contiguous_ndarray(self):
        """Test initialization of AerState with not-contiguous statevector"""

        sv1 = np.arange(2**4 * 2, dtype=np.complex128)[::2]
        for idx in range(0, len(sv1) - 1):
            sv1[idx] = complex(0.0, 0.0)
        sv1[len(sv1) - 1] = complex(1.0, 0.0)

        state2 = AerState(method="statevector")
        state2.initialize(sv1)
        state2.flush()
        sv2 = state2.move_to_ndarray()
        state2.close()

        self.assertIsNot(sv1, sv2)
        self.assertEqual(len(sv1), len(sv2))
        self.assertEqual(sv1[len(sv1) - 1], sv2[len(sv2) - 1])

    def test_initialize_density_matrix_with_non_contiguous_ndarray(self):
        """Test initialization of AerState with not-contiguous density matrix"""

        sv1 = np.arange(4**4 * 2, dtype=np.complex128).reshape(16, 32)[:, ::2]
        for row in range(len(sv1)):
            for col in range(len(sv1)):
                sv1[row, col] = complex(0.0, 0.0)
        sv1[len(sv1) - 1, len(sv1) - 1] = complex(1.0, 0.0)

        state2 = AerState(method="density_matrix")
        state2.initialize(sv1)
        state2.flush()
        sv2 = state2.move_to_ndarray()
        state2.close()

        self.assertIsNot(sv1, sv2)
        self.assertEqual(len(sv1), len(sv2))
        self.assertEqual(sv1[len(sv1) - 1, len(sv1) - 1], sv2[len(sv2) - 1, len(sv2) - 1])

    def test_appply_unitary(self):
        """Test applying a unitary matrix"""
        unitary_1 = random_unitary(2, seed=1111)
        unitary_2 = random_unitary(4, seed=2222)
        unitary_3 = random_unitary(8, seed=3333)

        circuit = QuantumCircuit(5)
        circuit.unitary(unitary_1, [0])
        circuit.unitary(unitary_2, [1, 2])
        circuit.unitary(unitary_3, [3, 4, 0])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize()

        state.apply_unitary([0], unitary_1)
        state.apply_unitary([1, 2], unitary_2)
        state.apply_unitary([3, 4, 0], unitary_3)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_multiplexer(self):
        """Test applying a multiplexer operation"""

        class CustomMultiplexer(Gate):
            def validate_parameter(self, param):
                return param

        def multiplexer_multi_controlled_x(num_control):
            identity = np.array(np.array([[1, 0], [0, 1]], dtype=complex))
            x_gate = np.array(np.array([[0, 1], [1, 0]], dtype=complex))
            num_qubits = num_control + 1
            multiplexer = CustomMultiplexer(
                "multiplexer",
                num_qubits,
                (2**num_control - 1) * [identity] + [x_gate],
            )
            return multiplexer

        multiplexr_1 = multiplexer_multi_controlled_x(1)
        multiplexr_2 = multiplexer_multi_controlled_x(2)
        multiplexr_3 = multiplexer_multi_controlled_x(3)

        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.append(multiplexr_1, [0, 1])
        circuit.append(multiplexr_2, [1, 2, 3])
        circuit.append(multiplexr_3, [3, 4, 0, 1])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_multiplexer([1], [0], multiplexr_1.params)
        state.apply_multiplexer([2, 3], [1], multiplexr_2.params)
        state.apply_multiplexer([4, 0, 1], [3], multiplexr_3.params)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_diagonal(self):
        """Test applying a diagonal gate"""
        diag_1 = [1, -1]
        diag_2 = [1, -1, -1, 1]
        diag_3 = [1, -1, 1, -1, 1, -1, 1, -1]

        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.append(DiagonalGate(diag_1), [0])
        circuit.append(DiagonalGate(diag_2), [1, 2])
        circuit.append(DiagonalGate(diag_3), [3, 4, 0])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_diagonal([0], diag_1)
        state.apply_diagonal([1, 2], diag_2)
        state.apply_diagonal([3, 4, 0], diag_3)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_mcx(self):
        """Test applying a mcx gate"""

        class MCX(Gate):
            def validate_parameter(self, param):
                return param

        def mcx(num_control):
            return MCX("mcx", num_control + 1, [])

        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.append(mcx(1), [0, 1])
        circuit.append(mcx(2), [1, 2, 3])
        circuit.append(mcx(3), [4, 0, 1, 2])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_mcx([0], 1)
        state.apply_mcx([1, 2], 3)
        state.apply_mcx([4, 0, 1], 2)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_mcy(self):
        """Test applying a mcy gate"""

        class MCY(Gate):
            def validate_parameter(self, param):
                return param

        def mcy(num_control):
            return MCY("mcy", num_control + 1, [])

        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.append(mcy(1), [0, 1])
        circuit.append(mcy(2), [1, 2, 3])
        circuit.append(mcy(3), [4, 0, 1, 2])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_mcy([0], 1)
        state.apply_mcy([1, 2], 3)
        state.apply_mcy([4, 0, 1], 2)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_mcz(self):
        """Test applying a mcz gate"""

        class MCZ(Gate):
            def validate_parameter(self, param):
                return param

        def mcz(num_control):
            return MCZ("mcz", num_control + 1, [])

        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.append(mcz(1), [0, 1])
        circuit.append(mcz(2), [1, 2, 3])
        circuit.append(mcz(3), [4, 0, 1, 2])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_mcz([0], 1)
        state.apply_mcz([1, 2], 3)
        state.apply_mcz([4, 0, 1], 2)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_reset(self):
        """Test applying a rest gate"""
        seed = 1234
        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.reset(0)
        circuit.reset([2, 4])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector", seed_simulator=seed)
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector", seed_simulator=seed)
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_reset([0])
        state.apply_reset([2, 4])
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_appply_measure(self):
        """Test applying a measure"""
        seed = 1234
        init_state = random_statevector(2**5, seed=111)

        circuit = QuantumCircuit(5, 1)
        circuit.initialize(init_state, [0, 1, 2, 3, 4])
        circuit.measure(0, 0)
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector", seed_simulator=seed)
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector", seed_simulator=seed)
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        state.apply_measure([0])
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)

    def test_probability(self):
        """Test probability() of outcome"""
        init_state = random_statevector(2**5, seed=111)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        expected = init_state.probabilities()

        for idx in range(0, 2**5):
            self.assertAlmostEqual(state.probability(idx), expected[idx])

    def test_probabilities(self):
        """Test probabilities() of outcome"""
        init_state = random_statevector(2**5, seed=111)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize(init_state.data)

        expected = init_state.probabilities()
        actual = state.probabilities()

        for idx in range(0, 2**5):
            self.assertAlmostEqual(actual[idx], expected[idx])

    def test_set_seed(self):
        """Test set_seed"""
        init_state = random_statevector(2**5, seed=111)

        state = AerState(method="statevector", seed_simulator=11111)
        state.allocate_qubits(5)
        state.initialize(init_state.data)
        sample1 = state.sample_counts()
        sample2 = state.sample_counts()

        state.set_seed(11111)
        sample3 = state.sample_counts()

        self.assertNotEqual(sample1, sample2)
        self.assertEqual(sample1, sample3)

    def test_sampling(self):
        """Test sampling"""
        init_state = random_statevector(2**5, seed=111)

        aer_simulator = AerSimulator(method="statevector")
        circuit = QuantumCircuit(5)
        circuit.initialize(init_state.data)
        circuit.measure_all()
        result = aer_simulator.run(circuit, seed_simulator=11111).result()
        expected = result.get_counts(0)

        state = AerState(method="statevector", seed_simulator=11111)
        state.allocate_qubits(5)
        state.initialize(init_state.data)
        actual = state.sample_counts()

        for key, value in actual.items():
            key_str = f"{key:05b}"
            expected_val = expected[key_str] if key_str in expected else 0
            self.assertAlmostEqual(actual[key], expected_val)

    def test_global_phase(self):
        """Test global phase"""
        unitary_1 = random_unitary(2, seed=1111)
        unitary_2 = random_unitary(4, seed=2222)
        unitary_3 = random_unitary(8, seed=3333)

        circuit = QuantumCircuit(5, global_phase=np.pi / 4)
        circuit.unitary(unitary_1, [0])
        circuit.unitary(unitary_2, [1, 2])
        circuit.unitary(unitary_3, [3, 4, 0])
        circuit.save_statevector()

        aer_simulator = AerSimulator(method="statevector")
        result = aer_simulator.run(circuit).result()
        expected = result.get_statevector(0)

        state = AerState(method="statevector")
        state.allocate_qubits(5)
        state.initialize()

        state.apply_global_phase(np.pi / 4)

        state.apply_unitary([0], unitary_1)
        state.apply_unitary([1, 2], unitary_2)
        state.apply_unitary([3, 4, 0], unitary_3)
        actual = state.move_to_ndarray()

        for i, amp in enumerate(actual):
            self.assertAlmostEqual(expected[i], amp)


if __name__ == "__main__":
    unittest.main()
