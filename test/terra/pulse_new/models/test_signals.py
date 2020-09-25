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
Tests for signals.
"""

import numpy as np

from test.terra.common import QiskitAerTestCase
from qiskit.providers.aer.pulse_new import Constant, PiecewiseConstant, Signal


class TestSignals(QiskitAerTestCase):
    """Tests for signals."""

    def setUp(self):
        pass

    def test_constant(self):
        """Test Constant signal"""

        constant = Constant(0.5)

        self.assertEquals(constant.envelope_value(), 0.5)
        self.assertEquals(constant.envelope_value(10.0), 0.5)
        self.assertEquals(constant.value(), 0.5)
        self.assertEquals(constant.value(10.0), 0.5)

    def test_signal(self):
        """Test Signal."""

        # Signal with constant amplitude
        signal = Signal(0.25, carrier_freq=0.3)
        self.assertEquals(signal.envelope_value(), 0.25)
        self.assertEquals(signal.envelope_value(1.23), 0.25)
        self.assertEquals(signal.value(), 0.25)
        self.assertEquals(signal.value(1.0), 0.25*np.exp(0.3*2.j*np.pi))

        # Signal with parabolic amplitude
        signal = Signal(lambda t: 2.0*t**2, carrier_freq=0.1)
        self.assertEquals(signal.envelope_value(), 0.0)
        self.assertEquals(signal.envelope_value(3.0), 18.0)
        self.assertEquals(signal.value(), 0.0)
        self.assertEquals(signal.value(2.0), 8.0*np.exp(0.1*2.j*np.pi*2.0))

    def test_piecewise_constant(self):
        """Test PWC signal."""

        dt = 1.
        samples = np.array([0., 0., 1., 2., 1., 0., 0.])
        carrier_freq = 0.5
        piecewise_const = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

        self.assertEquals(piecewise_const.envelope_value(), 0.0)
        self.assertEquals(piecewise_const.envelope_value(2.0), 1.0)
        self.assertEquals(piecewise_const.value(), 0.0)
        self.assertEquals(piecewise_const.value(3.0), 2.0*np.exp(0.5*2.j*np.pi*3.0))

    def test_multiplication(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1*const2, Constant))
        self.assertEquals((const1*const2).value(), 0.15)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0*t**2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 * signal1, Signal))
        self.assertTrue(isinstance(signal1 * const1, Signal))
        self.assertTrue(isinstance(signal1 * signal2, Signal))
        self.assertEquals((signal1*signal2).carrier_freq, 0.2)
        self.assertEquals((signal1 * const1).carrier_freq, 0.1)
        self.assertEquals((signal1 * signal2).envelope_value(), 0.0)
        self.assertEquals((signal1 * signal2).envelope_value(3.0), 3.*18.0)
        self.assertEquals((signal1 * signal2).value(), 0.0)
        self.assertEquals((signal1 * signal2).value(2.0), 24.0*np.exp(0.2*2.j*np.pi*2.0))

        # Test piecewise constant
        dt = 1.
        samples = np.array([0., 0., 1., 2., 1., 0., 0.])
        carrier_freq = 0.5
        pwc1 = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 2.
        samples = np.array([0., 0., 1., 2., 1., 0., 0.])
        carrier_freq = 0.1
        pwc2 = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(isinstance(const1 * pwc1, PiecewiseConstant))
        self.assertTrue(isinstance(signal1 * pwc1, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 * pwc2, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 * const1, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 * signal1, PiecewiseConstant))

        # Test values
        self.assertEquals((pwc1 * pwc2).dt, 1.0)
        self.assertEquals((pwc1 * pwc2).duration, 7.0)
        self.assertEquals((pwc1 * pwc2).carrier_freq, 0.6)

        self.assertEquals((pwc1 * pwc2).envelope_value(), 0.0)
        self.assertEquals((pwc1 * pwc2).envelope_value(4.0), 1.)
        self.assertEquals((pwc1 * pwc2).value(), 0.0)
        self.assertEquals((pwc1 * pwc2).value(4.0), 1.0*np.exp(0.6*2.j*np.pi*4.0))

    def test_addition(self):
        """Tests the multiplication of signals."""

        # Test Constant
        const1 = Constant(0.3)
        const2 = Constant(0.5)
        self.assertTrue(isinstance(const1 + const2, Constant))
        self.assertEquals((const1 + const2).value(), 0.8)

        # Test Signal
        signal1 = Signal(3.0, carrier_freq=0.1)
        signal2 = Signal(lambda t: 2.0*t**2, carrier_freq=0.1)
        self.assertTrue(isinstance(const1 + signal1, Signal))
        self.assertTrue(isinstance(signal1 + const1, Signal))
        self.assertTrue(isinstance(signal1 + signal2, Signal))
        self.assertEquals((signal1 + signal2).carrier_freq, 0.1)
        self.assertEquals((signal1 + const1).carrier_freq, 0.)
        self.assertEquals((signal1 + signal2).envelope_value(), 3.)
        self.assertEquals((signal1 + signal2).envelope_value(3.0), 3. + 18.)
        self.assertEquals((signal1 + signal2).value(), 3.0)
        self.assertEquals((signal1 + signal2).value(2.0), 11.0*np.exp(0.1*2.j*np.pi*2.0))

        # Test piecewise constant
        dt = 1.
        samples = np.array([0., 0., 1., 2., 1., 0., 0.])
        carrier_freq = 0.5
        pwc1 = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

        dt = 1.
        samples = np.array([0., 0., 1., 2., 1., 0., 0.])
        carrier_freq = 0.1
        pwc2 = PiecewiseConstant(dt=dt, samples=samples, carrier_freq=carrier_freq)

        # Test types
        self.assertTrue(isinstance(const1 + pwc1, PiecewiseConstant))
        self.assertTrue(isinstance(signal1 + pwc1, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 + pwc2, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 + const1, PiecewiseConstant))
        self.assertTrue(isinstance(pwc1 + signal1, PiecewiseConstant))

        # Test values
        self.assertEquals((pwc1 + pwc2).dt, 1.0)
        self.assertEquals((pwc1 + pwc2).duration, 7.0)
        self.assertEquals((pwc1 + pwc2).carrier_freq, 0.0)

        self.assertEquals((pwc1 + pwc2).envelope_value(), 0.0)
        expected = 1.*np.exp(0.5*2.j*np.pi*4.0) + 1.*np.exp(0.1*2.j*np.pi*4.0)
        self.assertEquals((pwc1 + pwc2).envelope_value(4.0), expected)
        self.assertEquals((pwc1 + pwc2).value(), 0.0)
        self.assertEquals((pwc1 + pwc2).value(4.0), expected)
