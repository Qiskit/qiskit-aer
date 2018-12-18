# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
ReadoutError class tests
"""

import unittest
from test.terra.utils import common
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.readout_error import ReadoutError


class TestReadoutError(common.QiskitAerTestCase):
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
        """Test reset error noise model"""

        # Test circuit: ideal outcome "11"
        probs = [[1, 0], [0, 1]]
        roerror_dict = {'type': 'roerror',
                        'operations': ['measure'],
                        'probabilities': probs}
        roerror = ReadoutError(probs)
        self.assertEqual(roerror.number_of_qubits, 1)
        self.assertEqual(roerror.probabilities, probs)
        self.assertEqual(roerror.as_dict(), roerror_dict)

    def test_2qubit(self):
        # Test circuit: ideal outcome "11"
        probs = [[0.7, 0.2, 0.1, 0],
                 [0, 0.9, 0.1, 0],
                 [0, 0, 1, 0],
                 [0.1, 0.1, 0.2, 0.6]]
        roerror_dict = {'type': 'roerror',
                        'operations': ['measure'],
                        'probabilities': probs}
        roerror = ReadoutError(probs)
        self.assertEqual(roerror.number_of_qubits, 2)
        self.assertEqual(roerror.probabilities, probs)
        self.assertEqual(roerror.as_dict(), roerror_dict)


if __name__ == '__main__':
    unittest.main()
