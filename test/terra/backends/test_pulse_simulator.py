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

    Uses Hamiltonian H = -0.5*omega0*sigmaz + omega1*os(omega*t+phi)*sigmax,
    as it is analytically solvable. Action of the gates is computed by performing
    the rotating frame transformation R = e^(-i*omega*t*sigmaz/2). The Hamiltonian in the
    rotating frame, applying the Rotating Wave Approximation (RWA), is
    Hrot = 0.5*(omega-omega0)*sigmaz + 0.5*omega1*cos(phi)*sigmax - 0.5*omega1*sin(phi)*sigmay
    (note time independence). The unitary evolution is then given by Urot = e^(-i*Hrot*t).
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

    def create_qobj(self, shots, omega0, omega1, omega, phi, meas_level):
        """ Creates qobj for the specified pulse experiment

        Args:
            See Hamiltonian at top of file
            shots (int): number of times to perform experiment
            omega0 (float): qubit frequency
            omega1 (float): drive power/amplitude
            omega (float): drive frequency
            phi (float): drive phase
            meas_level (int): how to return the data

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
                        meas_level=meas_level, meas_return='single',
                        memory_slots=1, qubit_lo_freq=[omega/(2*np.pi)],
                        shots=shots, sim_config=self.back_config_dict)

        return qobj

    # ---------------------------------------------------------------------
    # Test gates (using meas level 2)
    # ---------------------------------------------------------------------
    def test_x_gate(self):
        """ Test x gate. Set omega=omega0 (drive on resonance), phi=0, omega1 = pi/time
        """

        # set variables

        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0

        # Require omega1*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega1 = np.pi/self.drive_samples

        phi = 0

        x_qobj = self.create_qobj(shots=256, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                  meas_level=2)
        result = self.backend_sim.run(x_qobj).result()
        counts = result.get_counts()

        exp_result = {'10':256} # Bitstring format of the 1 state

        self.assertDictAlmostEqual(counts, exp_result)

    def test_hadamard_gate(self):
        """ Test Hadamard. Is a rotation of pi/2 about the y-axis. Set omega=omega0
        (drive on resonance), phi=-pi/2, omega1 = pi/2/time
        """

        # set variables

        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0

        # Require omega1*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega1 = np.pi/2/self.drive_samples

        phi = -np.pi/2
        shots = 100000 # large number of shots so get good probabilities

        had_qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                    phi=phi, meas_level=2)
        result = self.backend_sim.run(had_qobj).result()
        counts = result.get_counts()

        # compare proportions
        prop = {}
        for key in counts.keys():
            prop[key] = counts[key]/shots

        exp_prop = {'0':0.5, '10':0.5}

        self.assertDictAlmostEqual(prop, exp_prop, delta=0.01)

    def test_arbitrary_gate(self):
        """ Test a few examples w/ arbitary drive, phase and amplitude. Compare to result obtained
        via the rotating Hamiltonian at top of file (Hrot).
        """

        # Gate 1

        # set variables
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0+0.01

        omega1 = 2*np.pi/3/self.drive_samples

        phi = 5*np.pi/7
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare proportions to expected result
        qobj1 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result1 = self.backend_sim.run(qobj1).result()
        counts1 = result1.get_counts()

        prop1 = {}
        for key in counts1.keys():
            prop1[key] = counts1[key]/shots

        exp_prop1 = {'0':0.315253, '10':1-0.315253}

        self.assertDictAlmostEqual(prop1, exp_prop1, delta=0.001)

        # Gate 2

        # set variables
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0+0.02

        omega1 = 7*np.pi/5/self.drive_samples

        phi = 19*np.pi/14
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare proportions to expected result
        qobj2 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result2 = self.backend_sim.run(qobj2).result()
        counts2 = result2.get_counts()

        prop2 = {}
        for key in counts2.keys():
            prop2[key] = counts2[key]/shots

        exp_prop2 = {'0':0.634952, '10':1-0.634952}

        self.assertDictAlmostEqual(prop2, exp_prop2, delta=0.001)

        # Gate 3

        # set variables
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0+0.005

        omega1 = 0.1

        phi = np.pi/4
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare proportions to expected result
        qobj3 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result3 = self.backend_sim.run(qobj3).result()
        counts3 = result3.get_counts()

        prop3 = {}
        for key in counts3.keys():
            prop3[key] = counts3[key]/shots

        exp_prop3 = {'0':0.0861794, '10':1-0.0861794}

        self.assertDictAlmostEqual(prop3, exp_prop3, delta=0.001)

    # ---------------------------------------------------------------------
    # Test meas level 1
    # ---------------------------------------------------------------------

    def test_meas_level_1(self):
        """ Test measurement level 1. """

        # perform hadamard setup (so get some 0's and some 1's), but use meas_level = 1

         # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit
        omega = omega0

        # Require omega1*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega1 = np.pi/2/self.drive_samples

        phi = -np.pi/2

        shots = 100000 # run large number of shots for good proportions

        qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                meas_level=1)
        result = self.backend_sim.run(qobj).result()

        # Verify that (about) half the IQ vals have abs val 1 and half have abs val 0
        # (use proportions for easier comparison)
        mem = np.abs(result.get_memory()[:, self.qubit])

        iq_prop = {'0': 0, '10': 0}
        for i in mem:
            if i == 0:
                iq_prop['0'] += 1/shots
            else:
                iq_prop['10'] += 1/shots

        exp_prop = {'0': 0.5, '10': 0.5}

        self.assertDictAlmostEqual(iq_prop, exp_prop, delta=0.01)


if __name__ == '__main__':
    unittest.main()
