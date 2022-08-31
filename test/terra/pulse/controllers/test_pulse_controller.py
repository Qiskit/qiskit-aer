# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""tests for pulse_controller.py"""

import unittest
import numpy as np
from qiskit_aer.pulse.controllers.pulse_controller import setup_rhs_dict_freqs

from ...common import QiskitAerTestCase


class TestSetupRHSDictFreqs(QiskitAerTestCase):
    """Tests for setup_rhs_dict_freqs"""

    def setUp(self):
        super().setUp()
        self.default_dict = {'freqs': [1., 2., 3.]}

        def calculate_channel_frequencies(qubit_lo_freq):
            return {'D0': qubit_lo_freq[0],
                    'U0': qubit_lo_freq[0] - qubit_lo_freq[1],
                    'D1': qubit_lo_freq[1]}

        self.calculate_channel_frequencies = calculate_channel_frequencies

    def test_without_override(self):
        """Test maintenance of default values if no frequencies specified in exp."""

        output_dict = setup_rhs_dict_freqs(self.default_dict,
                                           {},
                                           self.calculate_channel_frequencies)

        self.assertAlmostEqual(np.array(output_dict['freqs']),
                               np.array(self.default_dict['freqs']))

        output_dict = setup_rhs_dict_freqs(self.default_dict,
                                           {'qubit_lo_freq': None},
                                           self.calculate_channel_frequencies)

        self.assertAlmostEqual(np.array(output_dict['freqs']),
                               np.array(self.default_dict['freqs']))

    def test_with_override(self):
        """Test overriding of default values with qubit_lo_freq in exp."""

        output_dict = setup_rhs_dict_freqs(self.default_dict,
                                           {'qubit_lo_freq': [5, 11]},
                                           self.calculate_channel_frequencies)
        self.assertAlmostEqual(np.array(output_dict['freqs']),
                               np.array([5, -6, 11]))

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
