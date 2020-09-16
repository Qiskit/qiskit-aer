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
Tests for signal transfer functions.
"""

import numpy as np

from test.terra.common import QiskitAerTestCase
from qiskit.providers.aer.pulse_new import Convolution, PiecewiseConstant


class TestTransferFunctions(QiskitAerTestCase):
    """Tests for transfer functions."""

    def setUp(self):
        pass

    def test_convolution(self):
        """Test of convolution function."""
        ts = np.linspace(0, 100, 200)

        def gaus(t):
            sigma = 4
            return 2. * ts[1] / np.sqrt(2. * np.pi * sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))

        # Test the simple convolution of a signal without a carrier
        convolution = Convolution(gaus)

        samples = [0. if t < 20. or t > 80. else 1. for t in ts]  # Defines a square pulse.
        piecewise_const = PiecewiseConstant(dt=ts[1] - ts[0], samples=samples, carrier_freq=0.0, start_time=0)

        self.assertEquals(piecewise_const.duration, len(ts))
        self.assertEquals(piecewise_const.value(21.0), 1.0)
        self.assertEquals(piecewise_const.value(81.0), 0.0)

        convolved = convolution.apply(piecewise_const)

        self.assertLess(convolved.value(21.0), 1.0)
        self.assertGreater(convolved.value(81.0), 0.0)
        self.assertEquals(convolved.duration, 2*len(ts)-1)
