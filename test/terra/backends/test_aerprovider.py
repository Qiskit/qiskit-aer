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
Integration Tests for AerProvider.
"""

import unittest
from math import pi
import numpy as np

from test.terra import common
from qiskit.providers.aer import Aer, AerSimulator

class TestAerProvider(common.QiskitAerTestCase):
    """AerProvider tests"""

    def test_get_aer_simulator(self):
        """Test Aer.get_backend."""
        backend = Aer.get_backend('aer_simulator')
        self.assertEqual(backend.__class__, AerSimulator)
        self.assertEqual(backend.options.get('method'), 'automatic')

    def test_get_aer_simulator_with_statevector(self):
        """Test Aer.get_backend with statevector."""
        backend = Aer.get_backend('aer_simulator', method='statevector')
        self.assertEqual(backend.__class__, AerSimulator)
        self.assertEqual(backend.options.get('method'), 'statevector')

    def test_get_aer_simulator_with_mps(self):
        """Test Aer.get_backend with mps."""
        backend = Aer.get_backend('aer_simulator', method='matrix_product_state')
        self.assertEqual(backend.__class__, AerSimulator)
        self.assertEqual(backend.options.get('method'), 'matrix_product_state')

    def test_get_aer_simulator_with_statevector_parallel_threshold(self):
        """Test Aer.get_backend with statevector."""
        backend = Aer.get_backend('aer_simulator', method='statevector', statevector_parallel_threshold=1)
        self.assertEqual(backend.__class__, AerSimulator)
        self.assertEqual(backend.options.get('method'), 'statevector')
        self.assertEqual(backend.options.get('statevector_parallel_threshold'), 1)


if __name__ == '__main__':
    unittest.main()
