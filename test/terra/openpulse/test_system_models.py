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
from qiskit.providers.aer.openpulse.system_model import SystemModel
from qiskit.test.mock.fake_openpulse_2q import FakeOpenPulse2Q


class BaseTestSystemModel(QiskitAerTestCase):
    """Testing of functions in providers.aer.openpulse.qobj.digest."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.system = pulse.PulseChannelSpec.from_backend(self.backend)
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')


class TestSystemModel(BaseTestSystemModel):
    r"""Tests for Hamiltonian options and processing."""

    def test_qubit_lo_default(self):
        """Test backend_options['qubit_lo_freq'] defaults."""
        test_model = SystemModel.from_backend(self.backend)

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

        # set up inputs to _get_pulse_digest
        # Note: for this test the only relevant parameter input to assemble() is self.backend,
        # but the others args still need to be valid to assemble correctly
        sched_list = [_valid_2q_schedule()]
        qobj = assemble(sched_list,
                        self.backend,
                        meas_level=1,
                        meas_return='avg',
                        memory_slots=2,
                        shots=1)

        # set backend_options
        backend_options = self.config.to_dict()
        backend_options['hamiltonian'] = _create_2q_ham()
        backend_options['qubit_list'] = [0, 1]
        backend_options['qubit_lo_freq'] = 'from_hamiltonian'

        # test auto determination frequencies from_hamiltonian
        # (these values were computed by hand)
        backend_options['qubit_lo_freq'] = 'from_hamiltonian'
        openpulse_system = digest_pulse_obj(qobj, backend_options=backend_options, noise_model=None)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], 4.999009804864)
        self.assertAlmostEqual(openpulse_system.freqs['D1'], 5.100990195135)
        self.assertAlmostEqual(openpulse_system.freqs['U0'], 4.999009804864)
        self.assertAlmostEqual(openpulse_system.freqs['U1'], 0.101980390271)

        # test again with different parameters
        backend_options['hamiltonian'] = _create_2q_ham(v0=5.1, v1=4.9, j=0.02)
        openpulse_system = digest_pulse_obj(qobj, backend_options=backend_options, noise_model=None)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], 5.101980390271)
        self.assertAlmostEqual(openpulse_system.freqs['D1'], 4.898019609728)
        self.assertAlmostEqual(openpulse_system.freqs['U0'], 5.101980390271)
        self.assertAlmostEqual(openpulse_system.freqs['U1'], -0.203960780543)

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

def _valid_2q_schedule(self):
    """ Helper method to make a valid 2 qubit schedule
    Returns:
        schedule (pulse schedule): schedule for 2q experiment
    """
    rabi_pulse = pulse_lib.gaussian(duration=128,
                                    amp=0.5,
                                    sigma=16,
                                    name='rabi_pulse')
    meas_pulse = pulse_lib.gaussian_square(duration=1200,
                                           amp=0.025,
                                           sigma=4,
                                           risefall=25,
                                           name='meas_pulse')
    acq_cmd = pulse.Acquire(duration=10)

    # create measurement schedule
    measure_and_acquire = \
        meas_pulse(self.system.qubits[0].measure) | acq_cmd(self.system.acquires,
                   self.system.memoryslots)

    # add commands to schedule
    schedule = pulse.Schedule(name='rabi_exp')

    schedule += rabi_pulse(self.system.qubits[0].drive)
    schedule += measure_and_acquire << schedule.duration

    return schedule


def _create_2q_ham(v0=5.0,
                   v1=5.1,
                   j=0.01,
                   r=0.02,
                   alpha0=-0.33,
                   alpha1=-0.33,
                   qub_dim=3):
    """Create standard 2 qubit Hamiltonian, with various parameters.
    The defaults are those present in self.config.hamiltonian.
    Returns:
        hamiltonian (dict): dictionary representation of two qubit hamiltonian
    """

    hamiltonian = {}
    hamiltonian['h_str'] = []
    # Q0 terms
    hamiltonian['h_str'].append('np.pi*(2*v0-alpha0)*O0')
    hamiltonian['h_str'].append('np.pi*alpha0*O0*O0')
    hamiltonian['h_str'].append('2*np.pi*r*X0||D0')
    hamiltonian['h_str'].append('2*np.pi*r*X0||U1')
    hamiltonian['h_str'].append('2*np.pi*r*X1||U0')

    # Q1 terms
    hamiltonian['h_str'].append('np.pi*(2*v1-alpha1)*O1')
    hamiltonian['h_str'].append('np.pi*alpha1*O1*O1')
    hamiltonian['h_str'].append('2*np.pi*r*X1||D1')

    # Exchange coupling betwene Q0 and Q1
    hamiltonian['h_str'].append('2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)')
    hamiltonian['vars'] = {'v0': v0, 'v1': v1, 'j': j,
                           'r': r, 'alpha0': alpha0, 'alpha1': alpha1}

    # set the qubit dimensions to 3
    hamiltonian['qub'] = {'0': qub_dim, '1': qub_dim}

    return hamiltonian


if __name__ == '__main__':
    unittest.main()
