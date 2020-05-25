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
import numpy as np
from numpy.linalg import norm
from test.terra.common import QiskitAerTestCase
import qiskit
from qiskit.test.mock import FakeOpenPulse2Q
import qiskit.pulse as pulse
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.test.mock import FakeArmonk
from qiskit.providers.models.backendconfiguration import UchannelLO


class BaseTestPulseSystemModel(QiskitAerTestCase):
    """Tests for PulseSystemModel"""

    def setUp(self):
        self._default_qubit_lo_freq = [4.9, 5.0]
        self._u_channel_lo = []
        self._u_channel_lo.append([UchannelLO(0, 1.0+0.0j)])
        self._u_channel_lo.append([UchannelLO(0, -1.0+0.0j), UchannelLO(1, 1.0+0.0j)])

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

    def test_qubit_lo_from_configurable_backend(self):
        backend = FakeArmonk()
        test_model = PulseSystemModel.from_backend(backend)
        qubit_lo_from_hamiltonian = test_model.hamiltonian.get_qubit_lo_from_drift()
        freqs = test_model.calculate_channel_frequencies(qubit_lo_from_hamiltonian)
        self.assertAlmostEqual(freqs['D0'], 4.974286046328553)

    def _compute_u_lo_freqs(self, qubit_lo_freq):
        """
        Given qubit_lo_freq, return the computed u_channel_lo.
        """
        u_lo_freqs = []
        for scales in self._u_channel_lo:
            u_lo_freq = 0
            for u_lo_idx in scales:
                qfreq = qubit_lo_freq[u_lo_idx.q]
                qscale = u_lo_idx.scale.real
                u_lo_freq += qfreq * qscale
            u_lo_freqs.append(u_lo_freq)
        return u_lo_freqs

class TestHamiltonianModel(QiskitAerTestCase):
    """Tests for HamiltonianModel"""

    def test_subsystem_list_from_dict(self):
        """Test correct restriction of a Hamiltonian dict to a subset of systems"""

        # construct 2 duffing oscillator hamiltonian
        v0=5.0
        v1=5.1
        j=0.01
        r=0.02
        alpha0=-0.33
        alpha1=-0.33

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

        # restrict to qubit 0 and verify some properties
        ham_model0 = HamiltonianModel.from_dict(hamiltonian, subsystem_list=[0])
        evals_expected0 = np.array([0,
                                    np.pi*(2*v0-alpha0) + np.pi*alpha0,
                                    (2 * np.pi*(2*v0-alpha0)) + (4 * np.pi*alpha0)])
        eval_diff = norm(evals_expected0 - ham_model0._evals)
        self.assertAlmostEqual(eval_diff, 0)

        channel_labels0 = ham_model0._channels.keys()
        for key in ['D0', 'U1']:
            self.assertTrue(key in channel_labels0)
        self.assertEqual(len(channel_labels0), 2)

        qubit_lo_freq0 = ham_model0.get_qubit_lo_from_drift()
        expected_freq0 = np.array([(np.pi*(2*v0-alpha0) + np.pi*alpha0) / (2 * np.pi)])
        self.assertAlmostEqual(norm(qubit_lo_freq0 - expected_freq0), 0)

        # restrict to qubit 1 and verify some properties
        ham_model1 = HamiltonianModel.from_dict(hamiltonian, subsystem_list=[1])
        evals_expected1 = np.array([0,
                                    np.pi*(2*v1-alpha1) + np.pi*alpha1,
                                    (2 * np.pi*(2*v1-alpha1)) + (4 * np.pi*alpha1)])
        eval_diff = norm(evals_expected1 - ham_model1._evals)
        self.assertAlmostEqual(eval_diff, 0)

        channel_labels1 = ham_model1._channels.keys()
        for key in ['D1', 'U0']:
            self.assertTrue(key in channel_labels1)
        self.assertEqual(len(channel_labels1), 2)

        qubit_lo_freq1 = ham_model1.get_qubit_lo_from_drift()
        expected_freq1 = np.array([0, (np.pi*(2*v1-alpha1) + np.pi*alpha1) / (2 * np.pi)])
        self.assertAlmostEqual(norm(qubit_lo_freq1 - expected_freq1), 0)

    def test_eigen_sorting(self):
        """Test estate mappings"""

        X = np.array([[0,1],[1,0]])
        Y = np.array([[0,-1j], [1j, 0]])
        Z = np.array([[1,0], [0, -1]])

        simple_ham = {'h_str': ['a*X0','b*Y0', 'c*Z0', 'd*X0||D0'],
                      'vars': {'a': 0.1, 'b': 0.1, 'c' : 1, 'd': 0.},
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
        simple_ham = {'h_str': ['a*X0','b*Y0', 'c*Z0', 'd*X0||D0'],
                      'vars': {'a': 100, 'b': 32.1, 'c' : 0.12, 'd': 0.},
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

    def test_no_variables(self):
        """Test successful construction of Hamiltonian without variables"""

        # fully specified hamiltonian without variables
        ham_dict = {'h_str': ['2*np.pi*O0', '2*np.pi*X0||D0'], 'qub': {'0': 2}}
        ham_model = HamiltonianModel.from_dict(ham_dict)

        qubit_lo = ham_model.get_qubit_lo_from_drift()
        self.assertAlmostEqual(norm(qubit_lo - np.array([1.])), 0)

    def test_empty_hamiltonian_string_exception(self):
        """Test exception raising for empty hamiltonian string"""

        message = "Hamiltonian dict requires a non-empty 'h_str' entry."

        ham_dict = {}
        self.assert_hamiltonian_parse_exception(ham_dict, message)

        ham_dict = {'h_str': ['']}
        self.assert_hamiltonian_parse_exception(ham_dict, message)


    def test_empty_qub_exception(self):
        """Test exception raising for empty qub"""

        message = "Hamiltonian dict requires non-empty 'qub' entry with subsystem dimensions."

        ham_dict = {'h_str': 'X0'}
        self.assert_hamiltonian_parse_exception(ham_dict, message)

        ham_dict = {'h_str': 'X0', 'qub': {}}
        self.assert_hamiltonian_parse_exception(ham_dict, message)

    def test_no_channels_exception(self):
        """Test exception raising for a hamiltonian with no channels"""

        message = 'HamiltonianModel must contain channels to simulate.'

        # hamiltonian with no channels at all
        ham_dict = {'h_str': ['X0'], 'qub': {'0': 2}}
        self.assert_hamiltonian_parse_exception(ham_dict, message)

        # hamiltonian with channels that get removed by parsing
        ham_dict = {'h_str': ['X1||D1'], 'qub': {'0': 2}}
        self.assert_hamiltonian_parse_exception(ham_dict, message)


    def assert_hamiltonian_parse_exception(self, ham_dict, message):
        """Test that an attempt to parse a given ham_dict results in an exception with
        the given message.
        """
        try:
            ham_model = HamiltonianModel.from_dict(ham_dict)
        except Exception as exception:
            self.assertEqual(exception.message, message)

if __name__ == '__main__':
    unittest.main()
