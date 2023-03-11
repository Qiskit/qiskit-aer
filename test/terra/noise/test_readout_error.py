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
ReadoutError class tests
"""

import unittest
from test.terra.common import QiskitAerTestCase

import numpy as np

from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer.noise.errors.readout_error import ReadoutError


class TestReadoutError(QiskitAerTestCase):
    """Testing ReadoutError class"""

    def test_probabilities_normalized_exception(self):
        """Test exception is raised for probabilities greater than 1."""
        probs = [[0.9, 0.2], [0, 1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[0, 1], [0.9, 0.2]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

    def test_probabilities_negative_exception(self):
        """Test exception is raised for negative probabilities."""
        probs = [[1.1, -0.1], [0, 1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[0, 1], [1.1, -0.1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

    def test_probabilities_dimension_exception(self):
        """Test exception is raised if probabilities are not multi-qubit"""
        probs = [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

    def test_probabilities_length_exception(self):
        """Test exception is raised if probabilities are different lengths"""
        probs = [[1, 0, 0, 0], [0, 1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[0, 1], [1, 0, 0, 0]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

    def test_probabilities_num_outcomes_exception(self):
        """Test exception is raised if not enough probability vectors"""
        probs = [[0, 1]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[1, 0], [0, 1], [0, 0]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

        probs = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        self.assertRaises(NoiseError, lambda: ReadoutError(probs))

    def test_1qubit(self):
        """Test 1-qubit readout error"""

        # Test circuit: ideal outcome "11"
        probs = [[1, 0], [0, 1]]
        roerror_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        roerror = ReadoutError(probs)
        self.assertEqual(roerror.number_of_qubits, 1)
        self.assertEqual(roerror.probabilities.tolist(), probs)
        self.assertEqual(roerror.to_dict(), roerror_dict)

    def test_2qubit(self):
        """Test 2-qubit readout error"""
        # Test circuit: ideal outcome "11"
        probs = [[0.7, 0.2, 0.1, 0], [0, 0.9, 0.1, 0], [0, 0, 1, 0], [0.1, 0.1, 0.2, 0.6]]
        roerror_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        roerror = ReadoutError(probs)
        self.assertEqual(roerror.number_of_qubits, 2)
        self.assertEqual(roerror.probabilities.tolist(), probs)
        self.assertEqual(roerror.to_dict(), roerror_dict)

    def test_tensor(self):
        """Test tensor of two readout errors."""
        probs0 = [[0.9, 0.1], [0.4, 0.6]]
        probs1 = [[0.5, 0.5], [0.2, 0.8]]
        probs = np.kron(probs0, probs1).tolist()
        error_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        error0 = ReadoutError(probs0)
        error1 = ReadoutError(probs1)
        error = error0.tensor(error1)

        self.assertEqual(error.number_of_qubits, 2)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)

    def test_expand(self):
        """Test expand of two readout errors."""
        probs0 = [[0.9, 0.1], [0.4, 0.6]]
        probs1 = [[0.5, 0.5], [0.2, 0.8]]
        probs = np.kron(probs0, probs1).tolist()
        error_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        error0 = ReadoutError(probs0)
        error1 = ReadoutError(probs1)
        error = error1.expand(error0)

        self.assertEqual(error.number_of_qubits, 2)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)

    def test_compose(self):
        """Test compose of two readout errors."""
        probs0 = [[0.9, 0.1], [0.4, 0.6]]
        probs1 = [[0.5, 0.5], [0.2, 0.8]]
        probs = np.dot(probs1, probs0).tolist()
        error_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        error0 = ReadoutError(probs0)
        error1 = ReadoutError(probs1)
        # compose method
        error = error0.compose(error1)
        self.assertEqual(error.number_of_qubits, 1)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)
        # @ method
        error = error0 @ error1
        self.assertEqual(error.number_of_qubits, 1)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)

    def test_dot_front(self):
        """Test dot of two readout errors."""
        probs0 = [[0.9, 0.1], [0.4, 0.6]]
        probs1 = [[0.5, 0.5], [0.2, 0.8]]
        probs = np.dot(probs1, probs0).tolist()
        error_dict = {"type": "roerror", "operations": ["measure"], "probabilities": probs}
        error0 = ReadoutError(probs0)
        error1 = ReadoutError(probs1)
        # dot method
        error = error1.dot(error0)
        self.assertEqual(error.number_of_qubits, 1)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)
        # * method
        error = error1 * error0
        self.assertEqual(error.number_of_qubits, 1)
        self.assertEqual(error.probabilities.tolist(), probs)
        self.assertEqual(error.to_dict(), error_dict)

    def test_equal(self):
        """Test two readout errors are equal"""
        error1 = ReadoutError([[0.9, 0.1], [0.5, 0.5]])
        error2 = ReadoutError(np.array([[0.9, 0.1], [0.5, 0.5]]))
        self.assertEqual(error1, error2)

    def test_to_instruction(self):
        """Test conversion of ReadoutError to Instruction."""
        # 1-qubit case
        probs1 = [[0.8, 0.2], [0.5, 0.5]]
        instr1 = ReadoutError(probs1).to_instruction()
        self.assertEqual(instr1.name, "roerror")
        self.assertEqual(instr1.num_clbits, 1)
        self.assertEqual(instr1.num_qubits, 0)
        self.assertTrue(np.allclose(instr1.params, probs1))

        # 2-qubit case
        probs2 = np.kron(probs1, probs1)
        instr2 = ReadoutError(probs2).to_instruction()
        self.assertEqual(instr2.name, "roerror")
        self.assertEqual(instr2.num_clbits, 2)
        self.assertEqual(instr2.num_qubits, 0)
        self.assertTrue(np.allclose(instr2.params, probs2))


if __name__ == "__main__":
    unittest.main()
