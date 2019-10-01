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
PulseSimulator Integration Tests
"""

import unittest
from test.terra import common

import numpy as np

import qiskit
import qiskit.pulse as pulse

from qiskit.compiler import assemble

from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse.commands import SamplePulse

class TestPulseSimulator(common.QiskitAerTestCase):
    """ PulseSimulator tests.

    Uses Hamiltonian H = -0.5*omega0*sigmaz + 0.5*omega1*os(omega*t+phi)*sigmax,
    as it is analytically solvable. Action of the gates is computed by performing
    the rotating frame transformation R = e^(-i*omega*t*sigmaz/2). The Hamiltonian in the
    rotating frame, applying the Rotating Wave Approximation (RWA), is
    Hrot = 0.5*(omega-omega0)*sigmaz + 0.5*omega1*cos(phi)*sigmax (note time independence).
    The unitary evolution is then given by Urot = e^(-i*Hrot*t).
    """

    def setUp(self):
        """ Set configuration settings for pulse simulator """
        qiskit.IBMQ.load_account()

        #Get a pulse configuration from the fake backend
        self.backend_fake = FakeOpenPulse2Q()
        self.back_config = self.backend_fake.configuration()
        self.back_config_dict = self.back_config.to_dict()
        self.system = pulse.PulseChannelSpec.from_backend(self.backend_fake)
        self.defaults = self.backend_fake.defaults()

        self.qubit = 0 # use 0 qubit for tests
        self.freq_qubit = self.defaults.qubit_freq_est[self.qubit]

        #Get pulse simulator backend
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

        self.drive_samples = 100

    def create_qobj(self, shots, omega0, omega1, omega, phi):
        """ Creates qobj for the specified pulse experiment

        Args:
            See Hamiltonian at top of file
            shots (int): number of times to perform experiment
            omega0 (float): qubit frequency
            omega1 (float): drive power/amplitude
            omega (float): drive frequency
            phi (float): drive phase

        Returns:
            Qobj: qobj representing this pulse experiment
        """

        # Acquire pulse
        acq_cmd = pulse.Acquire(duration=self.drive_samples)
        acquire = acq_cmd(self.system.acquires, self.system.memoryslots)

        # Create schedule

        # drive pulse (just phase; omega1 included in Hamiltonian)
        const_pulse = np.ones(self.drive_samples)
        phase = np.exp(1j*phi)
        drive_pulse = SamplePulse(phase*const_pulse, name='drive_pulse')

        # add commands to schedule
        schedule = pulse.Schedule(name='drive_pulse')
        schedule += drive_pulse(self.system.qubits[self.qubit].drive)
        schedule += acquire << schedule.duration

        # Create the hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = []

        #Q0 terms
        hamiltonian['h_str'].append('-0.5*omega0*Z0')
        hamiltonian['h_str'].append('0.5*omega1*X0||D0')

        hamiltonian['vars'] = {'omega0': omega0, 'omega1': omega1}

        # set the qubit dimension to 2
        hamiltonian['qub'] = {'0': 2}

        # update the back_end
        self.back_config_dict['hamiltonian'] = hamiltonian
        self.back_config_dict['noise'] = {}
        self.back_config_dict['dt'] = 1.0

        self.back_config_dict['ode_options'] = {} # optionally set ode settings
        self.back_config_dict['qubit_list'] = [0] # restrict qubit set

        qobj = assemble([schedule], self.backend_fake,
                        meas_level=2, meas_return='single',
                        memory_slots=1, qubit_lo_freq=[omega/(2*np.pi)],
                        shots=shots, sim_config=self.back_config_dict)

        return qobj

    # ---------------------------------------------------------------------
    # Test x gate
    # ---------------------------------------------------------------------
    def test_x_gate(self):
        """ Test x gate

        Set omega=omega0 (drive on resonance), phi=0, omega1 = pi/time
        """

        # set variables

        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0

        # Require omega1*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega1 = np.pi/self.drive_samples

        phi = 0

        x_qobj = self.create_qobj(shots=256, omega0=omega0, omega1=omega1, omega=omega, phi=phi)
        result = self.backend_sim.run(x_qobj).result()
        counts = result.get_counts()

        exp_result = {'10':256}

        self.assertDictAlmostEqual(counts, exp_result)

if __name__ == '__main__':
    unittest.main()
