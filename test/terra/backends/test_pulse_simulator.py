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
from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue

class TestPulseSimulator(common.QiskitAerTestCase):
    """ PulseSimulator tests.

    # pylint: disable=anomalous backslash in string
    Uses Hamiltonian `H = -0.5*\omega_0*\sigma_z + \omega_1*\cos(\omega*t+\phi)*\sigma_x`,
    as it has a closed form solution under the rotating frame transformation. Described in readme.
    """

    def setUp(self):
        """ Set configuration settings for pulse simulator """
        qiskit.IBMQ.load_account()

        #Get a pulse configuration from the mock real device
        self.backend_real = FakeOpenPulse2Q()
        self.back_config = self.backend_real.configuration().to_dict()
        self.system = pulse.PulseChannelSpec.from_backend(self.backend_real)
        self.defaults = self.backend_real.defaults()

        # define the qubits
        self.qubit_0 = 0
        self.freq_qubit_0 = self.defaults.qubit_freq_est[self.qubit_0]

        self.qubit_1 = 1
        self.freq_qubit_1 = self.defaults.qubit_freq_est[self.qubit_1]

        # define the measurement map (so can measure the qubits seperately)
        self.meas_map = [[self.qubit_0], [self.qubit_1]]

        # define the pulse time (# of samples)
        self.drive_samples = 100

         # Define acquisition
        acq_cmd = pulse.Acquire(duration=self.drive_samples)
        self.acq_0 = acq_cmd(self.system.acquires[self.qubit_0],
                             self.system.memoryslots[self.qubit_0])

        #Get pulse simulator backend
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

    def single_pulse_schedule(self, phi):
        """ Creates schedule for single pulse test
        Args:
            phi (float): drive phase (phi in Hamiltonian)
        Returns:
            schedule (pulse schedule): schedule for this test
        """
         # drive pulse (just phase; omega1 included in Hamiltonian)
        const_pulse = np.ones(self.drive_samples)
        phase = np.exp(1j*phi)
        drive_pulse = SamplePulse(phase*const_pulse, name='drive_pulse')

        # add commands to schedule
        schedule = pulse.Schedule(name='drive_pulse')
        schedule |= drive_pulse(self.system.qubits[self.qubit_0].drive)
        schedule |= self.acq_0 << schedule.duration

        return schedule

    def frame_change_schedule(self, phi, fc_phi, fc_dur1, fc_dur2):
        """ Creates schedule for frame change test. Does a pulse w/ phase phi of duration dur1,
        then frame change of phase fc_phi, then another pulse of phase phi of duration dur2.
        The different durations for the pulses allow manipulation of rotation angles on Bloch sphere
        Args:
            phi (float): drive phase (phi in Hamiltonian)
            fc_phi (float): phase for frame change
            fc_dur1 (int): duration of first pulse
            fc_dur2 (int): duration of second pulse

        Returns:
            schedule (pulse schedule): schedule for frame change test
        """
         # drive pulse (just phase; omega1 included in Hamiltonian)
        phase = np.exp(1j*phi)
        drive_pulse_1 = SamplePulse(phase*np.ones(fc_dur1), name='drive_pulse_1')
        drive_pulse_2 = SamplePulse(phase*np.ones(fc_dur2), name='drive_pulse_2')

        # frame change
        fc_pulse = FrameChange(phase=fc_phi, name='fc')

        # add commands to schedule
        schedule = pulse.Schedule(name='fc_schedule')
        schedule |= drive_pulse_1(self.system.qubits[self.qubit_0].drive)
        schedule += fc_pulse(self.system.qubits[self.qubit_0].drive)
        schedule += drive_pulse_2(self.system.qubits[self.qubit_0].drive)
        schedule |= self.acq_0 << schedule.duration

        return schedule

    def persistent_value_schedule(self, omega1_pv):
        """ Creates schedule for persistent value experiment. Creates pv pulse w/ drive amplitude
        omega1_pv. It does this by setting the omega1 term in the Hamiltonian = 1. Sets length of
        the pv pulse = self.drive_samples. The product omega1_pv*self.drive_samples, then, controls
        the resulting state.
        Args:
            omega1_pv (float): drive amplitude from the pv pulse

        Returns:
            schedule (pulse schedule): schedule for pv experiment
        """

        # pv pulse
        pv_pulse = PersistentValue(value=omega1_pv, name='pv')

        # 0 amp drive pulse
        #drive_pulse = SamplePulse(np.ones(self.drive_samples), name='drive_pulse_0amp')

        # add commands to schedule
        schedule = pulse.Schedule(name='pv_schedule')
        schedule |= pv_pulse(self.system.qubits[self.qubit_0].drive)
        # make pv_pulse last exactly self.drive_samples
        #schedule += drive_pulse(self.system.qubits[self.qubit_0].drive) << self.drive_samples
        schedule |= self.acq_0 << self.drive_samples

        return schedule

    def create_qobj(self, shots, omega0, omega1, omega, phi, meas_level, schedule_type=None,
                    fc_phi=0, fc_dur1=0, fc_dur2=0, omega1_pv=0, qub_dim=2):
        """ Creates qobj for the specified pulse experiment. Uses Hamiltonian above.
        Args:
            shots (int): number of times to perform experiment
            omega0 (float): qubit frequency
            omega1 (float): drive power/amplitude
            omega (float): drive frequency
            phi (float): drive phase
            meas_level (int): how to return the data
            schedule_type (str): type of schedule. Default is single pulse, can also set to
            'fc' or 'pv'
            fc_phi (float): frame change phase (set to 0 unless fc experiment)
            fc_dur1 (int): duration of first pulse in fc experiment (set to 0 unless fc experiment)
            fc_dur2 (int): duration of second pulse in fc experiment (set to 0 unless fc experiment)
            omega1_pv (float): drive amplitude in pv_experiment (set to 0 unless pv
            experiment)
            qub_dim (int): dimension of the qubit subspace (defaults to 2)
        Returns:
            Qobj: qobj representing this pulse experiment
        """
        # Create schedule (default is single pulse schedule)
        schedule = self.single_pulse_schedule(phi=phi)

        if schedule_type == 'fc':
            schedule = self.frame_change_schedule(phi=phi, fc_phi=fc_phi,
                                                  fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        elif schedule_type == 'pv':
            omega1 = 1 # make sure omega1 is equal to 1 (so that drive amp is omega1_pv)
            schedule = self.persistent_value_schedule(omega1_pv=omega1_pv)

        # Create the hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = []

        # Q0 terms
        hamiltonian['h_str'].append('-0.5*omega0*Z0')
        hamiltonian['h_str'].append('0.5*omega1*X0||D0')

        # Q1 terms
        # none for now

        # Set variables in ham
        hamiltonian['vars'] = {'omega0': omega0, 'omega1': omega1}

        # set the qubit dimension to qub_dim
        hamiltonian['qub'] = {'0': qub_dim}

        # update the back_end
        self.back_config['hamiltonian'] = hamiltonian
        self.back_config['noise'] = {}
        self.back_config['dt'] = 1.0 # makes time = self.drive_samples

        self.back_config['ode_options'] = {} # optionally set ode settings
        self.back_config['qubit_list'] = [self.qubit_0] # restrict qubit set to 0

        qobj = assemble([schedule], self.backend_real,
                        meas_level=meas_level, meas_return='single',
                        meas_map=self.meas_map, qubit_lo_freq=[omega/(2*np.pi)],
                        memory_slots=1, shots=shots, sim_config=self.back_config)

        return qobj

    # ---------------------------------------------------------------------
    # Test gates (using meas level 2)
    # ---------------------------------------------------------------------
    def test_x_gate(self):
        """ Test x gate. Set omega=omega0 (drive on resonance), phi=0, omega1 = pi/time
        """

        # set variables

        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0

        # Require omega1*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega1 = np.pi/self.drive_samples

        phi = 0

        x_qobj = self.create_qobj(shots=256, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                  meas_level=2)
        result = self.backend_sim.run(x_qobj).result()
        counts = result.get_counts()

        exp_result = {'1':256}

        self.assertDictAlmostEqual(counts, exp_result)

    def test_hadamard_gate(self):
        """ Test Hadamard. Is a rotation of pi/2 about the y-axis. Set omega=omega0
        (drive on resonance), phi=-pi/2, omega1 = pi/2/time
        """

        # set variables

        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit_0
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

        # compare probs
        prob = {}
        for key in counts.keys():
            prob[key] = counts[key]/shots

        exp_prob = {'0':0.5, '1':0.5}

        self.assertDictAlmostEqual(prob, exp_prob, delta=0.01)

    def test_arbitrary_gate(self):
        """ Test a few examples w/ arbitary drive, phase and amplitude. """

        # Gate 1

        # set variables
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0+0.01

        omega1 = 2*np.pi/3/self.drive_samples

        phi = 5*np.pi/7
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare probs to expected result
        qobj1 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result1 = self.backend_sim.run(qobj1).result()
        counts1 = result1.get_counts()

        prob1 = {}
        for key in counts1.keys():
            prob1[key] = counts1[key]/shots

        exp_prob1 = {'0':0.315253, '1':1-0.315253}

        self.assertDictAlmostEqual(prob1, exp_prob1, delta=0.001)

        # Gate 2

        # set variables
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0+0.02

        omega1 = 7*np.pi/5/self.drive_samples

        phi = 19*np.pi/14
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare probs to expected result
        qobj2 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result2 = self.backend_sim.run(qobj2).result()
        counts2 = result2.get_counts()

        prob2 = {}
        for key in counts2.keys():
            prob2[key] = counts2[key]/shots

        exp_prob2 = {'0':0.634952, '1':1-0.634952}

        self.assertDictAlmostEqual(prob2, exp_prob2, delta=0.001)

        # Gate 3

        # set variables
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0+0.005

        omega1 = 0.1

        phi = np.pi/4
        shots = 10000000 # large number of shots so get good probabilities

        # Run qobj and compare probs to expected result
        qobj3 = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega,
                                 phi=phi, meas_level=2)
        result3 = self.backend_sim.run(qobj3).result()
        counts3 = result3.get_counts()

        prob3 = {}
        for key in counts3.keys():
            prob3[key] = counts3[key]/shots

        exp_prob3 = {'0':0.0861794, '1':1-0.0861794}

        self.assertDictAlmostEqual(prob3, exp_prob3, delta=0.001)

    # ---------------------------------------------------------------------
    # Test meas level 1
    # ---------------------------------------------------------------------

    def test_meas_level_1(self):
        """ Test measurement level 1. """

        # perform hadamard setup (so get some 0's and some 1's), but use meas_level = 1

         # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0

        # Require omega1*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega1 = np.pi/2/self.drive_samples

        phi = -np.pi/2

        shots = 100000 # run large number of shots for good probs

        qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                meas_level=1)
        result = self.backend_sim.run(qobj).result()

        # Verify that (about) half the IQ vals have abs val 1 and half have abs val 0
        # (use probs for easier comparison)
        mem = np.abs(result.get_memory()[:, self.qubit_0])

        iq_prob = {'0': 0, '1': 0}
        for i in mem:
            if i == 0:
                iq_prob['0'] += 1/shots
            else:
                iq_prob['1'] += 1/shots

        exp_prob = {'0': 0.5, '1': 0.5}

        self.assertDictAlmostEqual(iq_prob, exp_prob, delta=0.01)

    # ---------------------------------------------------------------------
    # Test FrameChange and PersistentValue commands
    # ---------------------------------------------------------------------

    def test_frame_change(self):
        """ Test frame change command. """
        shots = 1000000
        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0

        # set phi = 0
        phi = 0

        # Test 1
        # do pi/2 pulse, then pi phase change, then another pi/2 pulse. Verify left in |0> state
        fc_dur1 = self.drive_samples
        fc_dur2 = fc_dur1
        omega1 = np.pi/2/fc_dur1

        fc_phi = np.pi

        fc_qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                   meas_level=2, schedule_type='fc', fc_phi=fc_phi,
                                   fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        result = self.backend_sim.run(fc_qobj).result()
        counts = result.get_counts()
        exp_result = {'0':shots}

        self.assertDictAlmostEqual(counts, exp_result)

        # Test 2
        # do pi/4 pulse, then pi phase change, then do pi/8 pulse. Should get |0> w/ prob = 0.96194.
        fc_dur1 = self.drive_samples
        fc_dur2 = fc_dur1//2 # half time to do the pi/8 pulse (halves angle)
        omega1 = np.pi/4/fc_dur1

        fc_phi = np.pi

        fc_qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                   meas_level=2, schedule_type='fc', fc_phi=fc_phi,
                                   fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        result = self.backend_sim.run(fc_qobj).result()
        counts = result.get_counts()

        # verify probs
        prob = {}
        for key in counts.keys():
            prob[key] = counts[key]/shots

        exp_prob = {'0':0.96194, '1':(1-0.96194)}
        self.assertDictAlmostEqual(prob, exp_prob, delta=0.001)

    def test_persistent_value(self):
        """ Test persistent value command. """

        shots = 256
        # set omega0, omega equal (use qubit frequency) -> drive on resonance
        omega0 = 2*np.pi*self.freq_qubit_0
        omega = omega0

        # set phi = 0 (so don't need to account for this in pv amplitude)
        phi = 0

        # Set omega1 = 1 and do pi pulse w/ omega1_pv. Verify result is the |1> state
        omega1 = 1
        omega1_pv = np.pi/self.drive_samples # pi pulse

        pv_qobj = self.create_qobj(shots=shots, omega0=omega0, omega1=omega1, omega=omega, phi=phi,
                                   meas_level=2, schedule_type='pv',
                                   omega1_pv=omega1_pv)
        result = self.backend_sim.run(pv_qobj).result()
        counts = result.get_counts()
        exp_result = {'1':shots}

        self.assertDictAlmostEqual(counts, exp_result)

if __name__ == '__main__':
    unittest.main()
