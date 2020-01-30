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
Tests for PulseSystemModel and HamiltonianModel functionality
"""

import unittest
import warnings
from numpy import array
from numpy.linalg import norm
from test.terra.common import QiskitAerTestCase
import qiskit
from qiskit.test.mock import FakeOpenPulse2Q
import qiskit.pulse as pulse
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
from qiskit.providers.aer.openpulse.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.openpulse.hamiltonian_model import HamiltonianModel


class BaseTestPulseSystemModel(QiskitAerTestCase):
    """Tests for PulseSystemModel"""

    def setUp(self):
        self._default_qubit_lo_freq = [4.9, 5.0]
        self._u_channel_lo = [[{'q': 0, 'scale': [1.0, 0.0]}],
                             [{'q': 0, 'scale': [-1.0, 0.0]}, {'q': 1, 'scale': [1.0, 0.0]}]]

    def _simple_system_model(self, v0=5.0, v1=5.1, j=0.01, r=0.02, alpha0=-0.33, alpha1=-0.33):
        hamiltonian = {}
        hamiltonian['h_str'] = ['np.pi*(2*v0-alpha0)*O0',
                              'np.pi*alpha0*O0*O0',
                              '2*np.pi*r*X0||D0',
                              '2*np.pi*r*X0||U1',
                              '2*np.pi*r*X1||U0',
                              'np.pi*(2*v1-alpha1)*O1',
                              'np.pi*alpha1*O1*O1',
                              '2*np.pi*r*X1||D1',
                              '2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)']
        hamiltonian['qub'] = {'0' : 3, '1' : 3}
        hamiltonian['vars'] = {'v0': v0,
                               'v1': v1,
                               'j': j,
                               'r': r,
                               'alpha0': alpha0,
                               'alpha1': alpha1}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        subsystem_list =[0, 1]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                qubit_freq_est=self._default_qubit_lo_freq,
                                u_channel_lo=self._u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)


class TestPulseSystemModel(BaseTestPulseSystemModel):
    r"""Tests for Hamiltonian options and processing."""

    def test_control_channel_index(self):
        """Test PulseSystemModel.control_channel_index()."""

        # get the model with no control channel dict yet
        test_model = self._simple_system_model()

        # test that it gives a warning when a key has no corresponding control channel
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            ctrl_idx = test_model.control_channel_index('no_key')

            self.assertEqual(len(w), 1)
            self.assertTrue('ControlChannel' in str(w[-1].message))

        # control channel labels
        test_model.control_channel_labels = [(0,1)]

        self.assertEqual(test_model.control_channel_index((0,1)), 0)

        # test that it still correctly gives a warning for nonexistant indices
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            ctrl_idx = test_model.control_channel_index((1,0))

            self.assertEqual(len(w), 1)
            self.assertTrue('ControlChannel' in str(w[-1].message))

    def test_control_channel_labels_from_backend(self):
        """Test correct importing of backend control channel description."""
        backend = FakeOpenPulse2Q()

        system_model = PulseSystemModel.from_backend(backend)
        expected = [{'driven_q': 1, 'freq': '(1+0j)q0'},
                    {'driven_q': 0, 'freq': '(-1+0j)q0 + (1+0j)q1'}]

        self.assertEqual(system_model.control_channel_labels, expected)

    def test_qubit_lo_default(self):
        """Test drawing of defaults form a backend."""
        test_model = self._simple_system_model()
        default_qubit_lo_freq = self._default_qubit_lo_freq
        default_u_lo_freq = self._compute_u_lo_freqs(default_qubit_lo_freq)

        # test output of default qubit_lo_freq
        freqs = test_model.calculate_channel_frequencies()
        self.assertAlmostEqual(freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(freqs['U1'], default_u_lo_freq[1])

        # test defaults again, but with non-default hamiltonian
        test_model = self._simple_system_model(v0=5.1, v1=4.9, j=0.02)
        freqs = test_model.calculate_channel_frequencies()
        self.assertAlmostEqual(freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(freqs['U1'], default_u_lo_freq[1])

    def test_qubit_lo_from_hamiltonian(self):
        """Test computation of qubit_lo_freq from the hamiltonian itself."""
        test_model = self._simple_system_model()

        qubit_lo_from_hamiltonian = test_model.hamiltonian.get_qubit_lo_from_drift()
        freqs = test_model.calculate_channel_frequencies(qubit_lo_from_hamiltonian)
        self.assertAlmostEqual(freqs['D0'], 4.999009804864)
        self.assertAlmostEqual(freqs['D1'], 5.100990195135)
        self.assertAlmostEqual(freqs['U0'], 4.999009804864)
        self.assertAlmostEqual(freqs['U1'], 0.101980390271)

        # test again with different parameters
        test_model = self._simple_system_model(v0=5.1, v1=4.9, j=0.02)
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
        for scales in self._u_channel_lo:
            u_lo_freq = 0
            for u_lo_idx in scales:
                qfreq = qubit_lo_freq[u_lo_idx['q']]
                qscale = u_lo_idx['scale'][0]
                u_lo_freq += qfreq * qscale
            u_lo_freqs.append(u_lo_freq)
        return u_lo_freqs

class TestHamiltonianModel(QiskitAerTestCase):
    """Tests for HamiltonianModel"""

    def test_eigen_sorting(self):
        """Test estate mappings"""

        X = array([[0,1],[1,0]])
        Y = array([[0,-1j], [1j, 0]])
        Z = array([[1,0], [0, -1]])

        simple_ham = {'h_str': ['a*X0','b*Y0', 'c*Z0'],
                      'vars': {'a': 0.1, 'b': 0.1, 'c' : 1},
                      'qub': {'0': 2}}

        ham_model = HamiltonianModel.from_dict(simple_ham)

        # check norm
        for estate in ham_model._estates:
            self.assertAlmostEqual(norm(estate), 1)

        # check actually an eigenstate
        mat = 0.1 * X + 0.1 * Y + 1 * Z
        for idx, eval in enumerate(ham_model._evals):
            diff = mat @ ham_model._estates[:, idx] - eval * ham_model._estates[:, idx]
            self.assertAlmostEqual(norm(diff), 0)

        # Same test but with strongly off-diagonal hamiltonian, which should raise warning
        simple_ham = {'h_str': ['a*X0','b*Y0', 'c*Z0'],
                      'vars': {'a': 100, 'b': 32.1, 'c' : 0.12},
                      'qub': {'0': 2}}

        ham_model = HamiltonianModel.from_dict(simple_ham)

        # check norm
        for estate in ham_model._estates:
            self.assertAlmostEqual(norm(estate), 1)

        # check actually an eigenstate
        mat = 100 * X + 32.1 * Y + 0.12 * Z
        for idx, eval in enumerate(ham_model._evals):
            diff = mat @ ham_model._estates[:, idx] - eval * ham_model._estates[:, idx]
            self.assertAlmostEqual(norm(diff), 0)

if __name__ == '__main__':
    unittest.main()
