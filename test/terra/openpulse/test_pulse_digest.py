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
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel


class TestDigest(QiskitAerTestCase):
    """Testing of functions in providers.aer.pulse.qobj.digest.

    This may need to be totally removed"""
    def setUp(self):
        self.backend_sim = backend_sim = qiskit.Aer.get_backend('pulse_simulator')
        self.skipTest('The functionality in digest is being refactored.')

    def test_qubit_lo_freq_handling(self):
        """Test how digest_pulse_obj retrieves qubit_lo_freq from various locations."""

        # construct valid schedule list for passing to assemble
        schedules = [self._valid_2q_schedule()]

        # test qubit_lo_freq drawn from Hamiltonian when not specified elsewhere
        system_model = self._system_model_2Q()

        pulse_qobj = assemble(schedules, backend=self.backend_sim)
        op_system = full_digest(pulse_qobj, system_model)
        self.assertAlmostEqual(op_system.freqs['D0'], 4.999009804864)
        self.assertAlmostEqual(op_system.freqs['D1'], 5.100990195135)
        self.assertAlmostEqual(op_system.freqs['U0'], 5.100990195135)
        self.assertAlmostEqual(op_system.freqs['U1'], 4.999009804864)

        # test qubit_lo_freq taken from estimates in system_model if present and not in assemble
        system_model._qubit_freq_est = [4.9, 5.1]

        pulse_qobj = assemble(schedules, backend=self.backend_sim)
        op_system = full_digest(pulse_qobj, system_model)
        self.assertAlmostEqual(op_system.freqs['D0'], 4.9)
        self.assertAlmostEqual(op_system.freqs['D1'], 5.1)
        self.assertAlmostEqual(op_system.freqs['U0'], 5.1)
        self.assertAlmostEqual(op_system.freqs['U1'], 4.9)

        # test qubit_lo_freq passed to assemble overrides est
        system_model._qubit_freq_est = [4.9, 5.1]

        pulse_qobj = assemble(schedules, qubit_lo_freq=[4.8, 5.2], backend=self.backend_sim)
        op_system = full_digest(pulse_qobj, system_model)
        self.assertAlmostEqual(op_system.freqs['D0'], 4.8)
        self.assertAlmostEqual(op_system.freqs['D1'], 5.2)
        self.assertAlmostEqual(op_system.freqs['U0'], 5.2)
        self.assertAlmostEqual(op_system.freqs['U1'], 4.8)



    def _valid_2q_schedule(self):
        """Returns a valid 2 qubit schedule."""

        valid_pulse = pulse_lib.gaussian(duration=128,
                                         amp=0.5,
                                         sigma=16,
                                         name='valid_pulse')
        valid_meas_pulse = pulse_lib.gaussian_square(duration=1200,
                                                     amp=0.025,
                                                     sigma=4,
                                                     risefall=25,
                                                     name='valid_meas_pulse')
        acq_cmd = pulse.Acquire(duration=10)

        acquires = [pulse.AcquireChannel(0), pulse.AcquireChannel(1)]
        memoryslots = [pulse.MemorySlot(0), pulse.MemorySlot(1)]

        # create measurement schedule
        measure_and_acquire = \
            valid_meas_pulse(pulse.MeasureChannel(0)) | acq_cmd(acquires, memoryslots)

        # add commands to schedule
        schedule = pulse.Schedule(name='valid_exp')

        schedule += valid_pulse(pulse.DriveChannel(0))
        schedule += measure_and_acquire << schedule.duration

        return schedule

    def _system_model_2Q(self,
                         v0=5.0,
                         v1=5.1,
                         j=0.01,
                         r=0.02,
                         alpha0=-0.33,
                         alpha1=-0.33,
                         qub_dim=3):
        """Constructs a simple 2 transmon PulseSystemModel."""

        hamiltonian = {}
        hamiltonian['h_str'] = []
        # Q0 terms
        hamiltonian['h_str'].append('np.pi*(2*v0-alpha0)*O0')
        hamiltonian['h_str'].append('np.pi*alpha0*O0*O0')
        hamiltonian['h_str'].append('2*np.pi*r*X0||D0')

        # Q1 terms
        hamiltonian['h_str'].append('np.pi*(2*v1-alpha1)*O1')
        hamiltonian['h_str'].append('np.pi*alpha1*O1*O1')
        hamiltonian['h_str'].append('2*np.pi*r*X1||D1')

        # Exchange coupling and ControlChannel terms
        hamiltonian['h_str'].append('2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)')
        hamiltonian['h_str'].append('2*np.pi*r*X0||U0')
        hamiltonian['h_str'].append('2*np.pi*r*X1||U1')

        # set vars and qubit dimensions
        hamiltonian['vars'] = {'v0': v0, 'v1': v1, 'j': j,
                               'r': r, 'alpha0': alpha0, 'alpha1': alpha1}
        hamiltonian['qub'] = {'0' : qub_dim, '1' : qub_dim}

        ham_model = HamiltonianModel.from_dict(hamiltonian)


        # set up u channel freqs,
        u_channel_lo = [[{'q': 1, 'scale': [1.0, 0.0]}],
                        [{'q': 0, 'scale': [1.0, 0.0]}]]
        subsystem_list = [0, 1]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)



if __name__ == '__main__':
    unittest.main()
