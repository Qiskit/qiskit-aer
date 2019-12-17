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
Tests for option handling in digest.py
"""

import unittest
from test.terra.common import QiskitAerTestCase
import qiskit
import qiskit.pulse as pulse
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
from qiskit.providers.aer.openpulse.pulse_system_model import PulseSystemModel
from qiskit.test.mock.fake_openpulse_2q import FakeOpenPulse2Q


class BaseTestPulseSystemModel(QiskitAerTestCase):
    """Testing of functions in providers.aer.openpulse.qobj.digest."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()


class TestPulseSystemModel(BaseTestPulseSystemModel):
    r"""Tests for Hamiltonian options and processing."""

    def test_qubit_lo_default(self):
        """Test backend_options['qubit_lo_freq'] defaults."""
        test_model = PulseSystemModel.from_backend(self.backend)

        default_qubit_lo_freq = getattr(self.backend.defaults(), 'qubit_freq_est')
        default_u_lo_freq = self._compute_u_lo_freqs(default_qubit_lo_freq)

        # test output of default qubit_lo_freq
        freqs = test_model.calculate_channel_frequencies()
        self.assertAlmostEqual(freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(freqs['U1'], default_u_lo_freq[1])

        # test defaults again, but with non-default hamiltonian
        test_model.hamiltonian.set_vars({'v0' : 5.1, 'v1' : 4.9, 'j' : 0.02})
        freqs = test_model.calculate_channel_frequencies()
        self.assertAlmostEqual(freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(freqs['U1'], default_u_lo_freq[1])

    def test_qubit_lo_from_hamiltonian(self):
        """Test backend_options['qubit_lo_freq'] = 'from_hamiltonian'."""
        test_model = PulseSystemModel.from_backend(self.backend)

        qubit_lo_from_hamiltonian = test_model.hamiltonian.get_qubit_lo_from_drift()
        freqs = test_model.calculate_channel_frequencies(qubit_lo_from_hamiltonian)
        self.assertAlmostEqual(freqs['D0'], 4.999009804864)
        self.assertAlmostEqual(freqs['D1'], 5.100990195135)
        self.assertAlmostEqual(freqs['U0'], 4.999009804864)
        self.assertAlmostEqual(freqs['U1'], 0.101980390271)

        # test again with different parameters
        test_model.hamiltonian.set_vars({'v0' : 5.1, 'v1' : 4.9, 'j' : 0.02})
        qubit_lo_from_hamiltonian = test_model.hamiltonian.get_qubit_lo_from_drift()
        freqs = test_model.calculate_channel_frequencies(qubit_lo_from_hamiltonian)
        self.assertAlmostEqual(freqs['D0'], 5.101980390271)
        self.assertAlmostEqual(freqs['D1'], 4.898019609728)
        self.assertAlmostEqual(freqs['U0'], 5.101980390271)
        self.assertAlmostEqual(freqs['U1'], -0.203960780543)

    def _compute_u_lo_freqs(self, qubit_lo_freq):
        """
        Given qubit_lo_freq, return the computed u_channel_lo.
        """
        u_lo_freqs = []
        for scales in self.config.to_dict()['u_channel_lo']:
            u_lo_freq = 0
            for u_lo_idx in scales:
                qfreq = qubit_lo_freq[u_lo_idx['q']]
                qscale = u_lo_idx['scale'][0]
                u_lo_freq += qfreq * qscale
            u_lo_freqs.append(u_lo_freq)
        return u_lo_freqs

if __name__ == '__main__':
    unittest.main()
