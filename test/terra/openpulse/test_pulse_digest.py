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

import numpy as np

import qiskit
import qiskit.pulse as pulse
from qiskit.pulse import pulse_lib
from qiskit.compiler import assemble

#from qiskit.compiler import assemble

from qiskit.providers.aer.backends.pulse_simulator import digest_pulse_obj
from qiskit.test.mock.fake_openpulse_2q import FakeOpenPulse2Q
#from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue


class BaseTestDigest(QiskitAerTestCase):
    """Testing of functions in providers.aer.openpulse.qobj.digest."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.system = pulse.PulseChannelSpec.from_backend(self.backend)
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

class TestDigestHamiltonian(BaseTestDigest):
    r"""Tests for Hamiltonian options and processing."""

    def test_qubit_lo_from_hamiltonian(self):
        """Test backend_options['qubit_lo_freq'] = 'from_hamiltonian'."""

        # set up inputs to _get_pulse_digest
        # Note: for this test the only relevant parameter input to assemble() is self.backend,
        # but the others args still need to be valid to assemble correctly
        sched_list = [self._valid_2q_schedule()]
        qobj = assemble(sched_list,
                        self.backend,
                        meas_level=1,
                        meas_return='avg',
                        memory_slots=2,
                        shots=1)

        # set backend_options
        backend_options = self.config.to_dict()
        backend_options['hamiltonian'] = self._create_2q_ham()
        backend_options['qubit_list'] = [0,1]
        default_qubit_lo_freq = getattr(self.backend.defaults(),'qubit_freq_est')
        default_u_lo_freq = self._compute_u_lo_freqs(default_qubit_lo_freq)

        # test output of default qubit_lo_freq
        openpulse_system = self._get_pulse_digest(qobj,backend_options=backend_options)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(openpulse_system.freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(openpulse_system.freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(openpulse_system.freqs['U1'], default_u_lo_freq[1])

        # test auto determination frequencies from_hamiltonian
        # (these values were computed by hand)
        backend_options['qubit_lo_freq'] = 'from_hamiltonian'
        openpulse_system = self._get_pulse_digest(qobj,backend_options=backend_options)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], 4.999009804864)
        self.assertAlmostEqual(openpulse_system.freqs['D1'], 5.100990195135)
        self.assertAlmostEqual(openpulse_system.freqs['U0'], 4.999009804864)
        self.assertAlmostEqual(openpulse_system.freqs['U1'], 0.101980390271)

        # test again with different parameters
        backend_options['hamiltonian'] = self._create_2q_ham(v0=5.1, v1 = 4.9, j=0.02)
        openpulse_system = self._get_pulse_digest(qobj,backend_options=backend_options)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], 5.101980390271)
        self.assertAlmostEqual(openpulse_system.freqs['D1'], 4.898019609728)
        self.assertAlmostEqual(openpulse_system.freqs['U0'], 5.101980390271)
        self.assertAlmostEqual(openpulse_system.freqs['U1'], -0.203960780543)

        # test defaults again, but with non-default hamiltonian
        del backend_options['qubit_lo_freq']
        openpulse_system = self._get_pulse_digest(qobj,backend_options=backend_options)
        self.assertAlmostEqual(openpulse_system.freqs['D0'], default_qubit_lo_freq[0])
        self.assertAlmostEqual(openpulse_system.freqs['D1'], default_qubit_lo_freq[1])
        self.assertAlmostEqual(openpulse_system.freqs['U0'], default_u_lo_freq[0])
        self.assertAlmostEqual(openpulse_system.freqs['U1'], default_u_lo_freq[1])

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

    def _get_pulse_digest(self, qobj, backend_options=None):
        qobj_dict = self.backend_sim._format_qobj_dict(qobj,
                                                       backend_options=backend_options,
                                                       noise_model=None)
        return digest_pulse_obj(qobj_dict)


    def _valid_2q_schedule(self):
        """ Helper method to make a valid 2 qubit schedule
        Returns:
            schedule (pulse schedule): schedule for 2q experiment
        """
        #qubit to use for exeperiment
        qubit = 0

        # Rabi pulse
        drive_amp = 0.5
        drive_samples = 128
        drive_sigma = 16

        # Measurement pulse
        meas_amp = 0.025
        meas_samples = 1200
        meas_sigma = 4
        meas_risefall = 25

        # Measurement pulse (common for all experiment)
        meas_pulse = pulse_lib.gaussian_square(duration=meas_samples,
                                               amp=meas_amp,
                                               sigma=meas_sigma,
                                               risefall=meas_risefall,
                                               name='meas_pulse')
        acq_cmd=pulse.Acquire(duration=meas_samples)

        # create measurement schedule
        measure_and_acquire = meas_pulse(
                                self.system.qubits[qubit].measure) | acq_cmd(self.system.acquires,
                                self.system.memoryslots)

        # add commands to schedule
        schedule = pulse.Schedule(name='rabi_exp_amp_%s' % drive_amp)

        rabi_pulse = pulse_lib.gaussian(duration=drive_samples,
                                    amp=drive_amp,
                                    sigma=drive_sigma, name='rabi_pulse')
        schedule += rabi_pulse(self.system.qubits[qubit].drive)
        schedule += measure_and_acquire << schedule.duration

        return schedule

    def _create_2q_ham(self,
                      v0=5.0,
                      v1=5.1,
                      j=0.01,
                      r = 0.02,
                      alpha0 = -0.33,
                      alpha1 = -0.33,
                      qub_dim=3):
        """Create standard 2 qubit Hamiltonian, with various parameters.
        The defaults are those present in self.config.hamiltonian.
        Returns:
            hamiltonian (dict): dictionary representation of two qubit hamiltonian
        """

        hamiltonian = {}
        hamiltonian['h_str'] = []
        #Q0 terms
        hamiltonian['h_str'].append('np.pi*(2*v0-alpha0)*O0')
        hamiltonian['h_str'].append('np.pi*alpha0*O0*O0')
        hamiltonian['h_str'].append('2*np.pi*r*X0||D0')
        hamiltonian['h_str'].append('2*np.pi*r*X0||U1')
        hamiltonian['h_str'].append('2*np.pi*r*X1||U0')

        #Q1 terms
        hamiltonian['h_str'].append('np.pi*(2*v1-alpha1)*O1')
        hamiltonian['h_str'].append('np.pi*alpha1*O1*O1')
        hamiltonian['h_str'].append('2*np.pi*r*X1||D1')

        #Exchange coupling betwene Q0 and Q1
        hamiltonian['h_str'].append('2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)')
        hamiltonian['vars'] =  {'v0': v0, 'v1': v1, 'j': j,
                                'r': r, 'alpha0': alpha0, 'alpha1': alpha1}

        #set the qubit dimensions to 3
        hamiltonian['qub'] = {'0': qub_dim, '1': qub_dim}

        return hamiltonian

if __name__ == '__main__':
    unittest.main()
