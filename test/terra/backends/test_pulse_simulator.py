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
    r"""PulseSimulator tests.

    Mathematical expressions are formulated in latex in docstrings for this class.

    # pylint: disable=anomalous backslash in string
    Uses single qubit Hamiltonian "H = -\frac{1}{2} \omega_0 \sigma_z + \frac{1}{2} \omega_a
    e^{i(\omega_{d0} t+\phi)} \sigma_x", as it has a closed form solution under the rotating frame
    transformation. We make sure H is Hermitian by taking the complex conjugate of the lower
    triangular piece (as done by the simulator). To find the closed form, we apply the unitary
    "Urot = e^{-i \omega_0 t \sigma_z/2}". In this frame, the Hamiltonian becomes
    "Hrot = \frac{1}{2} \omega_a (\cos(\phi) \sigma_x + \sin(\phi) \sigma_y)",
    which is easily solvable.
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

        # 1q measurement map (so can measure the qubits seperately)
        self.meas_map_1q =  [[self.qubit_0], [self.qubit_1]]
        # 2q measurement map
        self.meas_map_2q = [[self.qubit_0, self.qubit_1]]

        # define the pulse time (# of samples)
        self.drive_samples = 100

         # Define acquisition
        acq_cmd = pulse.Acquire(duration=self.drive_samples)
        self.acq_0 = acq_cmd(self.system.acquires[self.qubit_0],
                             self.system.memoryslots[self.qubit_0])
        self.acq_01 = acq_cmd(self.system.acquires, self.system.memoryslots)

        #Get pulse simulator backend
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

    def single_pulse_schedule(self, phi):
        """Creates schedule for single pulse test
        Args:
            phi (float): drive phase (phi in Hamiltonian)
        Returns:
            schedule (pulse schedule): schedule for this test
        """
         # drive pulse (just phase; omega_a included in Hamiltonian)
        const_pulse = np.ones(self.drive_samples)
        phase = np.exp(1j*phi)
        drive_pulse = SamplePulse(phase*const_pulse, name='drive_pulse')

        # add commands to schedule
        schedule = pulse.Schedule(name='drive_pulse')
        schedule |= drive_pulse(self.system.qubits[self.qubit_0].drive)

        schedule |= self.acq_0 << schedule.duration

        return schedule

    def frame_change_schedule(self, phi, fc_phi, fc_dur1, fc_dur2):
        """Creates schedule for frame change test. Does a pulse w/ phase phi of duration dur1,
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
         # drive pulse (just phase; omega_a included in Hamiltonian)
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

    def persistent_value_schedule(self, omega_a_pv):
        """Creates schedule for persistent value experiment. Creates pv pulse w/ drive amplitude
        omega_a_pv. It does this by setting the omega_a term in the Hamiltonian = 1. Sets length of
        the pv pulse = self.drive_samples. The product omega_a_pv*self.drive_samples, then, controls
        the resulting state.
        Args:
            omega_a_pv (float): drive amplitude from the pv pulse

        Returns:
            schedule (pulse schedule): schedule for pv experiment
        """
        # pv pulse
        pv_pulse = PersistentValue(value=omega_a_pv, name='pv')

        # add commands to schedule
        schedule = pulse.Schedule(name='pv_schedule')
        schedule |= pv_pulse(self.system.qubits[self.qubit_0].drive)
        schedule |= self.acq_0 << self.drive_samples

        return schedule

    def schedule_2q(self):
        """Creates schedule for testing two qubit interaction. Specifically, do a pi pulse on qub 0
        so it starts in the `1` state (drive channel) and then apply constant pulses to each
        qubit (on control channel 1). This will allow us to test a swap gate.
        Returns:
            schedule (pulse schedule): schedule for 2q experiment
        """
        # set up const pulse
        const_pulse = SamplePulse(np.ones(self.drive_samples), name='const_pulse')

        # set u channel
        uchannel = 1 # gives omega1-omega0 (we will set equal, so don't need negation)

        # add commands to schedule
        schedule = pulse.Schedule(name='2q_schedule')
        schedule |= const_pulse(self.system.qubits[self.qubit_0].drive) # pi pulse drive
        schedule += const_pulse(self.system.controls[uchannel]) # u channel pulse
        schedule |= self.acq_01 << schedule.duration

        return schedule


    def create_ham_1q(self, omega_0, omega_a, qub_dim):
        """Create single qubit Hamiltonian as given in class docstring
        Args:
            omega_0 (float): qubit 0 frequency
            omega_a (float): drive amplitude
            qub_dim (int): dimension of qubit subspace
        Returns:
            hamiltonian (dict): dictionary representation of single qubit hamiltonian
        """

        # Create the hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = []

        # Q0 terms
        hamiltonian['h_str'].append('-0.5*omega0*Z0')
        hamiltonian['h_str'].append('0.5*omegaa*X0||D0')

        # Q1 terms
        # none

        # Set variables in ham
        hamiltonian['vars'] = {'omega0': omega_0, 'omegaa': omega_a}

        # set the qubit dimension to qub_dim
        hamiltonian['qub'] = {'0': qub_dim}

        return hamiltonian

    def create_ham_2q(self, omega_0, omega_a, omega_i, qub_dim):
        """Create two qubit Hamiltonian as given in comment of interaction test
        Args:
            omega_a (float): Q0 drive amplitude
            omega_i (float): interaction amplitude
            qub_dim (int): dimension of qubit subspace (same for both qubits)
        Returns:
            hamiltonian (dict): dictionary representation of two qubit hamiltonian
        """

         # Create the hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = []

        # Q0 single qubit term (used to pi pulse Q0)
        hamiltonian['h_str'].append('-0.5*omega0*Z0')
        hamiltonian['h_str'].append('0.5*omegaa*X0||D0')

        # interaction term (uses U channels to get exponential piece)
        hamiltonian['h_str'].append('omegai*Sp0*Sm1||U1')
        hamiltonian['h_str'].append('omegai*Sm0*Sp1||U1')


        # Set variables in ham
        hamiltonian['vars'] = {'omega0': omega_0, 'omegaa': omega_a, 'omegai': omega_i}

        # set the qubit dimensions to qub_dim
        hamiltonian['qub'] = {'0': qub_dim, '1': qub_dim}

        return hamiltonian



    def create_qobj(self, shots, omega_0, omega_a, omega_d0, phi, meas_level, schedule_type=None,
                    fc_phi=0, fc_dur1=0, fc_dur2=0, omega_a_pv=0, qub_dim=2, omega_d1=0, omega_i=0,
                    is2q=False):
        """Creates qobj for the specified pulse experiment. Uses Hamiltonian from class docstring.
        Args:
            shots (int): number of times to perform experiment
            omega_0 (float): qubit 0 frequency
            omega_a (float): drive power/amplitude
            omega_d0 (float): qubit 0 drive frequency
            phi (float): drive phase
            meas_level (int): how to return the data
            schedule_type (str): type of schedule. Default is single pulse, can also set to
            'fc' or 'pv'
            fc_phi (float): frame change phase (set to 0 unless fc experiment)
            fc_dur1 (int): duration of first pulse in fc experiment (set to 0 unless fc experiment)
            fc_dur2 (int): duration of second pulse in fc experiment (set to 0 unless fc experiment)
            omega_a_pv (float): drive amplitude in pv_experiment (set to 0 unless pv
            experiment)
            qub_dim (int): dimension of the qubit subspace (defaults to 2)
            omega_d1 (float): qubit 1 drive frequency (defaults to 0)
            omega_i (float): amplitude for interaction term
            is2q (boolean): if true, use 2 qubit Hamiltonian given in interaction test comment
        Returns:
            Qobj: qobj representing this pulse experiment
        """
        # Create schedule (default is single pulse schedule)
        schedule = self.single_pulse_schedule(phi=phi)

        if schedule_type == 'fc':
            schedule = self.frame_change_schedule(phi=phi, fc_phi=fc_phi,
                                                  fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        elif schedule_type == 'pv':
            omega_a = 1 # make sure omega_a is equal to 1 (so that drive amp is omega_a_pv)
            schedule = self.persistent_value_schedule(omega_a_pv=omega_a_pv)

        # create the proper hamiltonian
        hamiltonian = self.create_ham_1q(omega_0=omega_0, omega_a=omega_a, qub_dim=qub_dim)
        # if 2q, update hamiltonian and the schedule
        if is2q:
            hamiltonian = self.create_ham_2q(omega_0=omega_0, omega_a=omega_a, omega_i=omega_i,
                                             qub_dim=qub_dim)
            schedule = self.schedule_2q()

        # update the back_end
        self.back_config['hamiltonian'] = hamiltonian
        self.back_config['noise'] = {}
        self.back_config['dt'] = 1.0 # makes time = self.drive_samples

        self.back_config['ode_options'] = {} # optionally set ode settings

        # 1 qubit settings
        qubit_list = [self.qubit_0]
        memory_slots = 1
        qubit_lo_freq = [omega_d0/(2*np.pi)]
        meas_map = self.meas_map_1q
        # update for 2 qubits
        if is2q:
            qubit_list.append(self.qubit_1)
            memory_slots = 2
            qubit_lo_freq.append(omega_d1/(2*np.pi))
            meas_map = self.meas_map_2q


        self.back_config['qubit_list'] = qubit_list
        qobj = assemble([schedule], self.backend_real,
                        meas_level=meas_level, meas_return='single',
                        meas_map=meas_map, qubit_lo_freq=qubit_lo_freq,
                        memory_slots=memory_slots, shots=shots, sim_config=self.back_config)

        return qobj

    #### SINGLE QUBIT TESTS ########################################################################

    # ---------------------------------------------------------------------
    # Test gates (using meas level 2)
    # ---------------------------------------------------------------------
    def test_x_gate(self):
        """Test x gate. Set omega_d0=omega_0 (drive on resonance), phi=0, omega_a = pi/time
        """

        # set variables

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega_a = np.pi/self.drive_samples

        phi = 0

        x_qobj = self.create_qobj(shots=256, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                  phi=phi, meas_level=2)
        result = self.backend_sim.run(x_qobj).result()
        counts = result.get_counts()

        exp_result = {'1':256}

        self.assertDictAlmostEqual(counts, exp_result)

    def test_hadamard_gate(self):
        """Test Hadamard. Is a rotation of pi/2 about the y-axis. Set omega_d0=omega_0
        (drive on resonance), phi=-pi/2, omega_a = pi/2/time
        """

        # set variables

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega_a = np.pi/2/self.drive_samples

        phi = -np.pi/2
        shots = 100000 # large number of shots so get good proportions

        had_qobj = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a,
                                    omega_d0=omega_d0, phi=phi, meas_level=2)
        result = self.backend_sim.run(had_qobj).result()
        counts = result.get_counts()

        # compare prop
        prop = {}
        for key in counts.keys():
            prop[key] = counts[key]/shots

        exp_prop = {'0':0.5, '1':0.5}

        self.assertDictAlmostEqual(prop, exp_prop, delta=0.01)

    def test_arbitrary_gate(self):
        """Test a few examples w/ arbitary drive, phase and amplitude. """

        # Gate 1

        # set variables
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0+0.01

        omega_a = 2*np.pi/3/self.drive_samples

        phi = 5*np.pi/7
        shots = 10000000 # large number of shots so get good proportions

        # Run qobj and compare prop to expected result
        qobj1 = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                 phi=phi, meas_level=2)
        result1 = self.backend_sim.run(qobj1).result()
        counts1 = result1.get_counts()

        prop1 = {}
        for key in counts1.keys():
            prop1[key] = counts1[key]/shots

        exp_prop1 = {'0':0.315253, '1':1-0.315253}

        self.assertDictAlmostEqual(prop1, exp_prop1, delta=0.001)

        # Gate 2

        # set variables
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0+0.02

        omega_a = 7*np.pi/5/self.drive_samples

        phi = 19*np.pi/14
        shots = 10000000 # large number of shots so get good proportions

        # Run qobj and compare prop to expected result
        qobj2 = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                 phi=phi, meas_level=2)
        result2 = self.backend_sim.run(qobj2).result()
        counts2 = result2.get_counts()

        prop2 = {}
        for key in counts2.keys():
            prop2[key] = counts2[key]/shots

        exp_prop2 = {'0':0.634952, '1':1-0.634952}

        self.assertDictAlmostEqual(prop2, exp_prop2, delta=0.001)

        # Gate 3

        # set variables
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0+0.005

        omega_a = 0.1

        phi = np.pi/4
        shots = 10000000 # large number of shots so get good proportions

        # Run qobj and compare prop to expected result
        qobj3 = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                 phi=phi, meas_level=2)
        result3 = self.backend_sim.run(qobj3).result()
        counts3 = result3.get_counts()

        prop3 = {}
        for key in counts3.keys():
            prop3[key] = counts3[key]/shots

        exp_prop3 = {'0':0.0861794, '1':1-0.0861794}

        self.assertDictAlmostEqual(prop3, exp_prop3, delta=0.001)

    # ---------------------------------------------------------------------
    # Test meas level 1
    # ---------------------------------------------------------------------

    def test_meas_level_1(self):
        """Test measurement level 1. """

        # perform hadamard setup (so get some 0's and some 1's), but use meas_level = 1

         # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega_a = np.pi/2/self.drive_samples

        phi = -np.pi/2

        shots = 100000 # run large number of shots for good proportions

        qobj = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                phi=phi, meas_level=1)
        result = self.backend_sim.run(qobj).result()

        # Verify that (about) half the IQ vals have abs val 1 and half have abs val 0
        # (use prop for easier comparison)
        mem = np.abs(result.get_memory()[:, self.qubit_0])

        iq_prop = {'0': 0, '1': 0}
        for i in mem:
            if i == 0:
                iq_prop['0'] += 1/shots
            else:
                iq_prop['1'] += 1/shots

        exp_prop = {'0': 0.5, '1': 0.5}

        self.assertDictAlmostEqual(iq_prop, exp_prop, delta=0.01)

    # ---------------------------------------------------------------------
    # Test FrameChange and PersistentValue commands
    # ---------------------------------------------------------------------

    def test_frame_change(self):
        """Test frame change command. """
        shots = 1000000
        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # set phi = 0
        phi = 0

        # Test 1
        # do pi/2 pulse, then pi phase change, then another pi/2 pulse. Verify left in |0> state
        fc_dur1 = self.drive_samples
        fc_dur2 = fc_dur1
        omega_a = np.pi/2/fc_dur1

        fc_phi = np.pi

        fc_qobj = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, schedule_type='fc', fc_phi=fc_phi,
                                   fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        result = self.backend_sim.run(fc_qobj).result()
        counts = result.get_counts()
        exp_result = {'0':shots}

        self.assertDictAlmostEqual(counts, exp_result)

        # Test 2
        # do pi/4 pulse, then pi phase change, then do pi/8 pulse. Should get |0> w/ prop = 0.96194.
        fc_dur1 = self.drive_samples
        fc_dur2 = fc_dur1//2 # half time to do the pi/8 pulse (halves angle)
        omega_a = np.pi/4/fc_dur1

        fc_phi = np.pi

        fc_qobj = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, schedule_type='fc', fc_phi=fc_phi,
                                   fc_dur1=fc_dur1, fc_dur2=fc_dur2)
        result = self.backend_sim.run(fc_qobj).result()
        counts = result.get_counts()

        # verify props
        prop = {}
        for key in counts.keys():
            prop[key] = counts[key]/shots

        exp_prop = {'0':0.96194, '1':(1-0.96194)}
        self.assertDictAlmostEqual(prop, exp_prop, delta=0.001)

    def test_persistent_value(self):
        """Test persistent value command. """

        shots = 256
        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # set phi = 0 (so don't need to account for this in pv amplitude)
        phi = 0

        # Set omega_a = 1 and do pi pulse w/ omega_a_pv. Verify result is the |1> state
        omega_a = 1
        omega_a_pv = np.pi/self.drive_samples # pi pulse

        pv_qobj = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, schedule_type='pv', omega_a_pv=omega_a_pv)
        result = self.backend_sim.run(pv_qobj).result()
        counts = result.get_counts()
        exp_result = {'1':shots}

        self.assertDictAlmostEqual(counts, exp_result)

    # ---------------------------------------------------------------------
    # Test higher energy levels (take 3 level system for simplicity)
    # `\sigma_x \rightarrow a+\dagger{a}`,
    # `\sigma_y \rightarrow -\imag (a-\dagger{a})`, etc
    # ---------------------------------------------------------------------

    def test_three_level(self):
        r"""Test 3 level system. Compare statevectors as counts only use bitstrings. Analytically, 
        the expected statevector is "(\frac{1}{3} (2+\cos(\frac{\sqrt{3}}{2} \omega_a t)),
        -\frac{i}{\sqrt{3}} \sin(\frac{\sqrt{3}}{2} \omega_a t),
        -\frac{2\sqrt{2}}{3} \sin(\frac{\sqrt{3}}{4} \omega_a t)^2)".
        """

        # set qubit dimension to 3
        qub_dim = 3

        shots = 1000
        # set omega_0,omega_d0 (use qubit frequency) -> drive on resonance
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # set phi = 0 for simplicity
        phi = 0

        # Test omega_a*t = pi
        omega_a = np.pi/self.drive_samples

        qobj_1 = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                  phi=phi, meas_level=2, qub_dim=qub_dim)
        result_1 = self.backend_sim.run(qobj_1).result()
        state_vector_1 = result_1.get_statevector()

        exp_state_vector_1 = [0.362425, 0. - 0.235892j, -0.901667]

        # compare vectors element-wise
        for i, _ in enumerate(state_vector_1):
            self.assertAlmostEqual(state_vector_1[i], exp_state_vector_1[i], places=4)

        # Test omega_a*t = 2 pi
        omega_a = 2*np.pi/self.drive_samples

        qobj_2 = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                  phi=phi, meas_level=2, qub_dim=qub_dim)
        result_2 = self.backend_sim.run(qobj_2).result()
        state_vector_2 = result_2.get_statevector()

        exp_state_vector_2 = [0.88871, 0.430608j, -0.157387]

        # compare vectors element-wise
        for i, _ in enumerate(state_vector_2):
            self.assertAlmostEqual(state_vector_2[i], exp_state_vector_2[i], places=4)

    #### TWO QUBIT TESTS ###########################################################################

    # ----------------------------------------------------------------------------------------------
    # Test qubit interaction (use 2 qubits for simplicity)
    # For these tests, we use a different 2-qubit Hamiltonian that tests both
    # interaction and control (U) channels. In the lab frame, it is given
    # by `H = -\frac{1}{2} \omega_0 \sigma_z^0 + \frac{1}{2} \omega_a e^{i \omega_{d0} t} \sigma_x^0
    # `+ \omega_i (e^{i (\omega_{d0}-\omega_{d1}) t} \sigma_p \otimes \sigma_{m1} + `
    # `+ e^{-i (\omega_{d0}-\omega_{d1}) t} \sigma_m \otimes \sigma_p)`. First 2 terms allow us to
    # excite the 0 qubit. Latter 2 terms define the interaction.
    # ----------------------------------------------------------------------------------------------

    def test_interaction(self):
        r"""Test 2 qubit interaction. Set `\omega_d0 = \omega_d1` (for first two test) as this
        removes the time dependence and makes the Hamiltonian easily solvable.
        Using `U = e^{-i H t}` one can see that H defines a swap gate. """

        shots = 1000000
        # set omega_d0=omega_0 (resonance)
        omega_0 = 2*np.pi*self.freq_qubit_0
        omega_d0 = omega_0

        # Set omega_d0=omega_d1
        omega_d1 = omega_d0

         # set phi = 0
        phi = 0

        # do pi pulse on Q0 and verify state swaps from '01' to '10'

        # Q0 drive amp -> pi pulse
        omega_a = np.pi/self.drive_samples
        # Interaction amp -> pi/2 pulse (creates the swap gate)
        omega_i = np.pi/2/self.drive_samples

        qobj_2q = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, omega_d1=omega_d1, omega_i=omega_i,
                                   is2q=True)
        result = self.backend_sim.run(qobj_2q).result()
        counts = result.get_counts()

        exp_counts = {'10': shots} # reverse ordering; after flip Q0, state is '01'
        self.assertDictAlmostEqual(counts, exp_counts, delta=2)

        # do pi/2 pulse on Q0 and verify half the counts are '00' and half are swapped state '10'

        # Q0 drive amp -> pi/2 pulse
        omega_a = np.pi/2/self.drive_samples
        # Interaction amp -> pi/2 pulse (creates the swap gate)
        omega_i = np.pi/2/self.drive_samples

        qobj_2q = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, omega_d1=omega_d1, omega_i=omega_i,
                                   is2q=True)
        result = self.backend_sim.run(qobj_2q).result()
        counts = result.get_counts()

        # compare proportions for improved accuracy
        prop = {}
        for key in counts.keys():
            prop[key] = counts[key]/shots

        exp_prop = {'00':0.5, '10':0.5}

        self.assertDictAlmostEqual(prop, exp_prop, delta=0.01)

        # again do pi pulse but now set omega_d1=0 (ie omega_d0 != omega_d1); verify swap does not occur
        omega_d1 = 0
        # Q0 drive amp -> pi pulse
        omega_a = np.pi/self.drive_samples
        # Interaction amp -> pi/2 pulse (creates the swap gate)
        omega_i = np.pi/2/self.drive_samples

        qobj_2q = self.create_qobj(shots=shots, omega_0=omega_0, omega_a=omega_a, omega_d0=omega_d0,
                                   phi=phi, meas_level=2, omega_d1=omega_d1, omega_i=omega_i,
                                   is2q=True)
        result = self.backend_sim.run(qobj_2q).result()
        counts = result.get_counts()

        swap_counts = {'10': shots}
        self.assertNotEqual(counts, swap_counts)

if __name__ == '__main__':
    unittest.main()
