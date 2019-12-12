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
from scipy.linalg import expm
from scipy.special import erf

import qiskit
import qiskit.pulse as pulse

from qiskit.compiler import assemble

from qiskit.test.mock.fake_openpulse_2q import FakeOpenPulse2Q
from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue


class TestPulseSimulator(common.QiskitAerTestCase):
    r"""PulseSimulator tests.

    Mathematical expressions are formulated in latex in docstrings for this class.

    # pylint: disable=anomalous backslash in string
    Uses single qubit Hamiltonian `H = -\frac{1}{2} \omega_0 \sigma_z + \frac{1}{2} \omega_a
    e^{i(\omega_{d0} t+\phi)} \sigma_x`. We make sure H is Hermitian by taking the complex conjugate
    of the lower triangular piece (as done by the simulator). To find the closed form, we move
    to a rotating frame via the unitary `Urot = e^{-i \omega t \sigma_z/2}
    (\ket{psi_{rot}}=Urot \ket{psi_{rot}})`. In this frame, the Hamiltonian becomes
    `Hrot = \frac{1}{2} \omega_a (\cos(\phi) \sigma_x - \sin(\phi) \sigma_y)
    + \frac{\omega_{d0}-\omega_0}{2} \sigma_z`.
    """
    def setUp(self):
        """ Set configuration settings for pulse simulator """
        # Get a pulse configuration from the mock real device
        self.backend_mock = FakeOpenPulse2Q()
        self.system = pulse.PulseChannelSpec.from_backend(self.backend_mock)
        self.defaults = self.backend_mock.defaults()

        # define the qubits
        self.qubit_0 = 0
        self.freq_qubit_0 = self.defaults.qubit_freq_est[self.qubit_0]

        self.qubit_1 = 1
        self.freq_qubit_1 = self.defaults.qubit_freq_est[self.qubit_1]

        # 1q measurement map (so can measure the qubits seperately)
        self.meas_map_1q = [[self.qubit_0], [self.qubit_1]]
        # 2q measurement map
        self.meas_map_2q = [[self.qubit_0, self.qubit_1]]

        # define the pulse time (# of samples)
        self.drive_samples = 100

        # Define acquisition
        acq_cmd = pulse.Acquire(duration=self.drive_samples)
        self.acq_0 = acq_cmd(self.system.acquires[self.qubit_0],
                             self.system.memoryslots[self.qubit_0])
        self.acq_01 = acq_cmd(self.system.acquires, self.system.memoryslots)

        # Get pulse simulator backend
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

    def single_pulse_schedule(self, phi, shape="square", gauss_sigma=0):
        """Creates schedule for single pulse test
        Args:
            phi (float): drive phase (phi in Hamiltonian)
            shape (str): shape of the pulse; defaults to square pulse
            gauss_sigma (float): std dev for gaussian pulse if shape=="gaussian"
        Returns:
            schedule (pulse schedule): schedule for this test
        """
        # define default square drive pulse (add phase only; omega_a included in Hamiltonian)
        const_pulse = np.ones(self.drive_samples)
        phase = np.exp(1j * phi)
        drive_pulse = SamplePulse(phase * const_pulse, name='drive_pulse')

        # create simple Gaussian drive if set this shape
        if shape == "gaussian":
            times = 1.0 * np.arange(self.drive_samples)
            gaussian = np.exp(-times**2 / 2 / gauss_sigma**2)
            drive_pulse = SamplePulse(gaussian, name='drive_pulse')

        # add commands to schedule
        schedule = pulse.Schedule(name='drive_pulse')
        schedule |= drive_pulse(self.system.qubits[self.qubit_0].drive)

        schedule |= self.acq_0 << schedule.duration

        return schedule

    def frame_change_schedule(self, phi, fc_phi, dur_drive1, dur_drive2):
        """Creates schedule for frame change test. Does a pulse w/ phase phi of duration dur_drive1,
        then frame change of phase fc_phi, then another pulse of phase phi of duration dur_drive2.
        The different durations for the pulses allow manipulation of rotation angles on Bloch sphere
        Args:
            phi (float): drive phase (phi in Hamiltonian)
            fc_phi (float): phase for frame change
            dur_drive1 (int): duration of first pulse
            dur_drive2 (int): duration of second pulse

        Returns:
            schedule (pulse schedule): schedule for frame change test
        """
        # drive pulse (just phase; omega_a included in Hamiltonian)
        phase = np.exp(1j * phi)
        drive_pulse_1 = SamplePulse(phase * np.ones(dur_drive1),
                                    name='drive_pulse_1')
        drive_pulse_2 = SamplePulse(phase * np.ones(dur_drive2),
                                    name='drive_pulse_2')

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
        const_pulse = SamplePulse(np.ones(self.drive_samples),
                                  name='const_pulse')

        # set u channel
        uchannel = 1  # gives omega1-omega0 (we will set equal, so don't need negation)

        # add commands to schedule
        schedule = pulse.Schedule(name='2q_schedule')
        schedule |= const_pulse(
            self.system.qubits[self.qubit_0].drive)  # pi pulse drive
        schedule += const_pulse(self.system.controls[uchannel]
                                ) << schedule.duration  # u chan pulse
        schedule |= self.acq_01 << schedule.duration

        return schedule

    def create_ham_1q(self, omega_0, omega_a, qub_dim=2):
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

    def create_ham_2q(self, omega_0, omega_a, omega_i, qub_dim=2):
        """Create two qubit Hamiltonian as given in comment of interaction test
        Args:
            omega_0 (float): Q0 frequency
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
        hamiltonian['h_str'].append(
            'omegai*Sm0*Sp1||U1')  # U1 gives diff omega_d1-omega_d0

        # Set variables in ham
        hamiltonian['vars'] = {
            'omega0': omega_0,
            'omegaa': omega_a,
            'omegai': omega_i
        }

        # set the qubit dimensions to qub_dim
        hamiltonian['qub'] = {'0': qub_dim, '1': qub_dim}

        return hamiltonian

    def backend_options_1q(self, omega_0, omega_a, qub_dim=2):
        """Creates backend_options dictionary for 1 qubit pulse simulation.

        Args:
            omega_0 (float): qubit 0 frequency
            omega_a (float): drive amplitude
            qub_dim (int): dimension of qubit subspace

        Returns:
            dict: backend_options dictionary.
        """
        hamiltonian = self.create_ham_1q(omega_0, omega_a, qub_dim)
        backend_options = self.backend_mock.configuration().to_dict()
        backend_options['hamiltonian'] = hamiltonian
        backend_options['qubit_list'] = [self.qubit_0]
        backend_options['dt'] = 1.0  # makes time = self.drive_samples
        backend_options['ode_options'] = {}  # optionally set ode settings
        backend_options['seed'] = 90841
        return backend_options

    def backend_options_2q(self, omega_0, omega_a, omega_i, qub_dim=2):
        """Creates backend_options dictionary for 1 qubit pulse simulation.

        Args:
            omega_0 (float): Q0 frequency
            omega_a (float): Q0 drive amplitude
            omega_i (float): interaction amplitude
            qub_dim (int): dimension of qubit subspace (same for both qubits)

        Returns:
            dict: backend_options dictionary.
        """
        hamiltonian = self.create_ham_2q(omega_0, omega_a, omega_i, qub_dim)
        backend_options = self.backend_mock.configuration().to_dict()
        backend_options['hamiltonian'] = hamiltonian
        backend_options['qubit_list'] = [self.qubit_0, self.qubit_1]
        backend_options['dt'] = 1.0  # makes time = self.drive_samples
        backend_options['ode_options'] = {}  # optionally set ode settings
        backend_options['seed'] = 12387
        return backend_options

    def qobj_params_1q(self, omega_d0):
        """Set params needed to create qobj for 1q tests
        Args:
            omega_d0 (float): qubit 0 drive (lo) frequency
        Returns:
            dict: contains memory_slots, qubit_lo_freq, meas_map for 1q qobj
        """
        memory_slots = 1
        qubit_lo_freq = [omega_d0 / (2 * np.pi)]
        meas_map = self.meas_map_1q

        return (memory_slots, qubit_lo_freq, meas_map)

    def qobj_params_2q(self, omega_d0, omega_d1):
        """Set params needed to create qobj for 2q tests
        Args:
            omega_d0 (float): qubit 0 drive (lo) frequency
            omega_d1 (float): qubit 1 drive (lo) frequency
         Returns:
            dict: contains memory_slots, qubit_lo_freq, meas_map for 2q qobj
        """
        memory_slots = 2
        qubit_lo_freq = [omega_d0 / (2 * np.pi), omega_d1 / (2 * np.pi)]
        meas_map = self.meas_map_2q

        return (memory_slots, qubit_lo_freq, meas_map)

    def create_qobj(self, shots, meas_level, schedule, qobj_params):
        """Creates qobj for the specified pulse experiment. Uses Hamiltonian from class docstring
        (except for 2q tests, which use Hamiltonian specified in comment of that test section).
        Args:
            shots (int): number of times to perform experiment
            meas_level (int): level of data to return
            schedule (Schedule): pulse schedule for the qobj
            qobj_params (tuple): tuple of memory_slots, qubit_lo_freq, meas_map
        Returns:
            Qobj: qobj representing this pulse experiment
        """
        # set qobj params
        memory_slots = qobj_params[0]
        qubit_lo_freq = qobj_params[1]
        meas_map = qobj_params[2]

        # construct the qobj
        qobj = assemble([schedule],
                        self.backend_mock,
                        meas_level=meas_level,
                        meas_return='single',
                        meas_map=meas_map,
                        qubit_lo_freq=qubit_lo_freq,
                        memory_slots=memory_slots,
                        shots=shots)

        return qobj

    # ---------------------------------------------------------------------
    # Test single qubit gates (using meas level 2 and square drive)
    # ---------------------------------------------------------------------

    def test_dt_scaling_x_gate(self):
        """
        Test that dt is being used correctly by the simulator
        """

        # do the same thing as test_x_gate, but scale dt and all frequency parameters
        # define test case for a single scaling
        def scale_test(scale):
            # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
            omega_0 = 2 * np.pi * self.freq_qubit_0/scale
            omega_d0 = omega_0

            # Require omega_a*time = pi to implement pi pulse (x gate)
            # num of samples gives time
            omega_a = np.pi / self.drive_samples/scale

            phi = 0

            x_schedule = self.single_pulse_schedule(phi)
            x_qobj_params = self.qobj_params_1q(omega_d0)
            x_qobj = self.create_qobj(shots=256,
                                      meas_level=2,
                                      schedule=x_schedule,
                                      qobj_params=x_qobj_params)
            x_backend_opts = self.backend_options_1q(omega_0, omega_a)
            x_backend_opts['dt'] = x_backend_opts['dt']*scale
            result = self.backend_sim.run(x_qobj,
                                          backend_options=x_backend_opts).result()
            counts = result.get_counts()
            exp_counts = {'1': 256}

            self.assertDictAlmostEqual(counts, exp_counts)
        # set scales and run tests
        scales = [2., 1.3453, 0.1234, 10.**5, 10**-5]
        for scale in scales:
            scale_test(scale)

    def test_x_gate(self):
        """
        Test x gate. Set omega_d0=omega_0 (drive on resonance), phi=0, omega_a = pi/time
        """

        # set variables

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega_a = np.pi / self.drive_samples

        phi = 0

        x_schedule = self.single_pulse_schedule(phi)
        x_qobj_params = self.qobj_params_1q(omega_d0)
        x_qobj = self.create_qobj(shots=256,
                                  meas_level=2,
                                  schedule=x_schedule,
                                  qobj_params=x_qobj_params)
        x_backend_opts = self.backend_options_1q(omega_0, omega_a)
        result = self.backend_sim.run(x_qobj,
                                      backend_options=x_backend_opts).result()
        counts = result.get_counts()
        exp_counts = {'1': 256}

        self.assertDictAlmostEqual(counts, exp_counts)

    def test_hadamard_gate(self):
        """Test Hadamard. Is a rotation of pi/2 about the y-axis. Set omega_d0=omega_0
        (drive on resonance), phi=-pi/2, omega_a = pi/2/time
        """

        # set variables
        shots = 100000  # large number of shots so get good proportions

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega_a = np.pi / 2 / self.drive_samples

        phi = -np.pi / 2

        had_schedule = self.single_pulse_schedule(phi)
        had_qobj_params = self.qobj_params_1q(omega_d0=omega_d0)

        had_qobj = self.create_qobj(shots=shots,
                                    meas_level=2,
                                    schedule=had_schedule,
                                    qobj_params=had_qobj_params)
        had_backend_opts = self.backend_options_1q(omega_0, omega_a)
        result = self.backend_sim.run(
            had_qobj, backend_options=had_backend_opts).result()
        counts = result.get_counts()

        # compare prop
        prop = {}
        for key in counts.keys():
            prop[key] = counts[key] / shots

        exp_prop = {'0': 0.5, '1': 0.5}

        self.assertDictAlmostEqual(prop, exp_prop, delta=0.01)

    def _analytic_prop_1q_gates(self, omega_0, omega_a, omega_d0, phi):
        """Compute proportion for 0 and 1 states analytically for single qubit gates.
        Args:
            omega_0 (float): Q0 freq
            omega_a (float): Q0 drive amplitude
            omega_d0 (flaot): Q0 drive frequency
            phi (float): drive phase
        Returns:
            exp_prop (dict): expected value of 0 and 1 proportions from analytic computation
            """
        time = self.drive_samples
        # write Hrot analytically
        h_rot = np.array([[
            (omega_d0 - omega_0) / 2,
            np.exp(1j * phi) * omega_a / 2
        ], [np.exp(-1j * phi) * omega_a / 2, -(omega_d0 - omega_0) / 2]])
        # exponentiate
        u_rot = expm(-1j * h_rot * time)
        state0 = np.array([1, 0])

        # compute analytic prob (proportion) of 0 state
        mat_elem0 = np.vdot(state0, np.dot(u_rot, state0))
        prop0 = np.abs(mat_elem0)**2

        # return expected proportion
        exp_prop = {'0': prop0, '1': 1 - prop0}
        return exp_prop

    def test_arbitrary_gate(self):
        """Test a few examples w/ arbitary drive, phase and amplitude. """
        shots = 10000  # large number of shots so get good proportions
        num_tests = 3
        # set variables for each test
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0_vals = [omega_0 + 1, omega_0 + 0.02, omega_0 + 0.005]
        omega_a_vals = [
            2 * np.pi / 3 / self.drive_samples,
            7 * np.pi / 5 / self.drive_samples, 0.1
        ]
        phi_vals = [5 * np.pi / 7, 19 * np.pi / 14, np.pi / 4]

        for i in range(num_tests):
            with self.subTest(i=i):
                schedule = self.single_pulse_schedule(phi_vals[i])
                qobj_params = self.qobj_params_1q(omega_d0=omega_d0_vals[i])

                qobj = self.create_qobj(shots=shots,
                                        meas_level=2,
                                        schedule=schedule,
                                        qobj_params=qobj_params)

                # Run qobj and compare prop to expected result
                backend_options = self.backend_options_1q(
                    omega_0, omega_a_vals[i])
                result = self.backend_sim.run(
                    qobj, backend_options=backend_options).result()
                counts = result.get_counts()

                prop = {}
                for key in counts.keys():
                    prop[key] = counts[key] / shots

                exp_prop = self._analytic_prop_1q_gates(
                    omega_0=omega_0,
                    omega_a=omega_a_vals[i],
                    omega_d0=omega_d0_vals[i],
                    phi=phi_vals[i])

                self.assertDictAlmostEqual(prop, exp_prop, delta=0.01)

    # ---------------------------------------------------------------------
    # Test meas level 1 (using square drive)
    # Note: the simulator generates approximate IQ data with the proper
    # data structure; it should not, however, be compared with an actual
    # device.
    # ---------------------------------------------------------------------

    def test_meas_level_1(self):
        """Test measurement level 1. """

        shots = 10000  # run large number of shots for good proportions
        # perform hadamard setup (so get some 0's and some 1's), but use meas_level = 1

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega_a = np.pi / 2 / self.drive_samples

        phi = -np.pi / 2

        schedule = self.single_pulse_schedule(phi)
        qobj_params = self.qobj_params_1q(omega_d0=omega_d0)
        qobj = self.create_qobj(shots=shots,
                                meas_level=1,
                                schedule=schedule,
                                qobj_params=qobj_params)
        backend_options = self.backend_options_1q(omega_0=omega_0,
                                                  omega_a=omega_a)
        result = self.backend_sim.run(
            qobj, backend_options=backend_options).result()

        # Verify that (about) half the IQ vals have abs val 1 and half have abs val 0
        # (use prop for easier comparison)
        mem = np.abs(result.get_memory()[:, self.qubit_0])

        iq_prop = {'0': 0, '1': 0}
        for i in mem:
            if i == 0:
                iq_prop['0'] += 1 / shots
            else:
                iq_prop['1'] += 1 / shots

        exp_prop = {'0': 0.5, '1': 0.5}

        self.assertDictAlmostEqual(iq_prop, exp_prop, delta=0.01)

    # ---------------------------------------------------------------------
    # Test Gaussian drive (using meas_level=2)
    # ---------------------------------------------------------------------

    def _analytic_gaussian_statevector(self, gauss_sigma, omega_a):
        r"""Computes analytic statevector for gaussian drive. Solving the Schrodinger equation in
        the rotating frame leads to the analytic solution `(\cos(x), -i\sin(x)) with
        `x = \frac{1}{2}\sqrt{\frac{\pi}{2}}\sigma\omega_a erf(\frac{t}{\sqrt{2}\sigma}).
        Args:
            gauss_sigma (float): std dev for the gaussian drive
            omega_a (float): Q0 drive amplitude
        Returns:
            exp_statevector (list): analytic form of the statevector computed for gaussian drive
                (Returned in the rotating frame)
        """
        time = self.drive_samples
        arg = 1 / 2 * np.sqrt(np.pi / 2) * gauss_sigma * omega_a * erf(
            time / np.sqrt(2) / gauss_sigma)
        exp_statevector = [np.cos(arg), -1j * np.sin(arg)]
        return exp_statevector

    def test_gaussian_drive(self):
        """Test gaussian drive pulse using meas_level_2. Set omega_d0=omega_0 (drive on resonance),
        phi=0, omega_a = pi/time
        """

        # set variables

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Require omega_a*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        omega_a = np.pi / self.drive_samples

        phi = 0

        # Test gaussian drive results for a few different sigma
        gauss_sigmas = {
            self.drive_samples / 6, self.drive_samples / 3, self.drive_samples
        }
        for gauss_sigma in gauss_sigmas:
            with self.subTest(gauss_sigma=gauss_sigma):
                schedule = self.single_pulse_schedule(phi=phi,
                                                      shape="gaussian",
                                                      gauss_sigma=gauss_sigma)
                qobj_params = self.qobj_params_1q(omega_d0=omega_d0)

                qobj = self.create_qobj(shots=1000,
                                        meas_level=2,
                                        schedule=schedule,
                                        qobj_params=qobj_params)
                backend_options = self.backend_options_1q(omega_0=omega_0,
                                                          omega_a=omega_a)
                result = self.backend_sim.run(
                    qobj, backend_options=backend_options).result()
                statevector = result.get_statevector()
                exp_statevector = self._analytic_gaussian_statevector(
                    gauss_sigma=gauss_sigma, omega_a=omega_a)
                # compare statevectors element-wise (comparision only accurate to 1 dec place)
                for i, _ in enumerate(statevector):
                    self.assertAlmostEqual(statevector[i],
                                           exp_statevector[i],
                                           places=1)

    # ---------------------------------------------------------------------
    # Test FrameChange and PersistentValue commands
    # ---------------------------------------------------------------------

    def _analytic_prop_fc(self, phi_net):
        """Compute analytic proportion of 0 and 1 from a given frame change. Analytically can show
        that the 0 prop is given by `cos(phi_net/2)^2`
        Args:
            phi_net (float): net rotation on Bloch sphere due to the frame change schedule
        Returns:
            exp_prop (dict): expected proportion of 0, 1 counts as computed analytically
        """
        prop0 = np.cos(phi_net / 2)**2
        exp_prop = {'0': prop0, '1': 1 - prop0}
        return exp_prop

    def test_frame_change(self):
        """Test frame change command. """
        shots = 10000
        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # set phi = 0
        phi = 0

        dur_drive1 = self.drive_samples  # first pulse duration
        fc_phi = np.pi

        # Test frame change where no shift in state results
        # specfically: do pi/2 pulse, then pi frame change, then another pi/2 pulse.
        # Verify left in |0> state
        dur_drive2_no_shift = dur_drive1  # same duration for both pulses
        omega_a_no_shift = np.pi / 2 / dur_drive1  # pi/2 pulse amplitude
        schedule_no_shift = self.frame_change_schedule(
            phi=phi,
            fc_phi=fc_phi,
            dur_drive1=dur_drive1,
            dur_drive2=dur_drive2_no_shift)
        qobj_params_no_shift = self.qobj_params_1q(omega_d0=omega_d0)

        qobj_no_shift = self.create_qobj(shots=shots,
                                         meas_level=2,
                                         schedule=schedule_no_shift,
                                         qobj_params=qobj_params_no_shift)
        backend_options_no_shift = self.backend_options_1q(
            omega_0, omega_a_no_shift)
        result_no_shift = self.backend_sim.run(
            qobj_no_shift, backend_options=backend_options_no_shift).result()
        counts_no_shift = result_no_shift.get_counts()
        exp_result_no_shift = {'0': shots}

        self.assertDictAlmostEqual(counts_no_shift, exp_result_no_shift)

        # Test frame change where a shift does result
        # specifically: do pi/4 pulse, then pi phase change, then do pi/8 pulse.
        # check that a net rotation of pi/4-pi/8 has occured on the Bloch sphere
        dur_drive2_shift = dur_drive1 // 2  # half time for second pulse (halves angle)
        omega_a_shift = np.pi / 4 / dur_drive1  # pi/4 pulse amplitude

        schedule_shift = self.frame_change_schedule(
            phi=phi,
            fc_phi=fc_phi,
            dur_drive1=dur_drive1,
            dur_drive2=dur_drive2_shift)

        qobj_params_shift = self.qobj_params_1q(omega_d0=omega_d0)
        qobj_shift = self.create_qobj(shots=shots,
                                      meas_level=2,
                                      schedule=schedule_shift,
                                      qobj_params=qobj_params_shift)
        backend_options_shift = self.backend_options_1q(omega_0, omega_a_shift)
        result_shift = self.backend_sim.run(
            qobj_shift, backend_options=backend_options_shift).result()
        counts_shift = result_shift.get_counts()

        # verify props
        prop_shift = {}
        for key in counts_shift.keys():
            prop_shift[key] = counts_shift[key] / shots

        # net angle is given by pi/4-pi/8
        exp_prop_shift = self._analytic_prop_fc(np.pi / 4 - np.pi / 8)
        self.assertDictAlmostEqual(prop_shift, exp_prop_shift, delta=0.01)

    @unittest.skip("PerisitentValue pulses are currently not supported.")
    def test_persistent_value(self):
        """Test persistent value command. """

        shots = 256
        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Set omega_a = 1 and do pi pulse w/ omega_a_pv. Verify result is the |1> state
        omega_a = 1
        omega_a_pv = np.pi / self.drive_samples  # pi pulse

        schedule = self.persistent_value_schedule(omega_a_pv)
        qobj_params = self.qobj_params_1q(omega_d0=omega_d0)

        pv_qobj = self.create_qobj(shots=shots,
                                   meas_level=2,
                                   schedule=schedule,
                                   qobj_params=qobj_params)
        backend_options = self.backend_options_1q(omega_0, omega_a)
        result = self.backend_sim.run(
            pv_qobj, backend_options=backend_options).result()
        counts = result.get_counts()
        exp_result = {'1': shots}

        self.assertDictAlmostEqual(counts, exp_result)

    # ---------------------------------------------------------------------
    # Test higher energy levels (take 3 level system for simplicity,
    # use square drive)
    # `\sigma_x \rightarrow a+\dagger{a}`,
    # `\sigma_y \rightarrow -\imag (a-\dagger{a})`, etc
    # ---------------------------------------------------------------------

    def _analytic_statevector_three_level(self, omega_a):
        r"""Returns analytically computed statevector for 3 level system with our Hamiltonian. Is
        given by `(\frac{1}{3} (2+\cos(\frac{\sqrt{3}}{2} \omega_a t)),
        -\frac{i}{\sqrt{3}} \sin(\frac{\sqrt{3}}{2} \omega_a t),
        -\frac{2\sqrt{2}}{3} \sin(\frac{\sqrt{3}}{4} \omega_a t)^2)`.
        Args:
            omega_a (float): Q0 drive amplitude
        Returns:
            exp_statevector (list): analytically computed statevector with Hamiltonian from above
                (Returned in the rotating frame)
        """
        time = self.drive_samples
        arg1 = np.sqrt(3) * omega_a * time / 2  # cos arg for first component
        arg2 = arg1  # sin arg for first component
        arg3 = arg1 / 2  # sin arg for 3rd component
        exp_statevector = np.array([(2 + np.cos(arg1)) / 3,
                                    -1j * np.sin(arg2) / np.sqrt(3),
                                    -2 * np.sqrt(2) * np.sin(arg3)**2 / 3],
                                   dtype=complex)
        return exp_statevector

    def test_three_level(self):
        r"""Test 3 level system. Compare statevectors as counts only use bitstrings. Analytic form
        given in _analytic_statevector_3level function docstring.
        """

        shots = 1000
        # Set omega_0,omega_d0 (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # Set phi = 0 for simplicity
        phi = 0

        # Test pi pulse
        omega_a_pi = np.pi / self.drive_samples

        schedule_pi = self.single_pulse_schedule(phi)
        qobj_params_pi = self.qobj_params_1q(omega_d0=omega_d0)

        qobj_pi = self.create_qobj(shots=shots,
                                   meas_level=2,
                                   schedule=schedule_pi,
                                   qobj_params=qobj_params_pi)

        # Set qub_dim=3 in hamiltonian
        backend_options_pi = self.backend_options_1q(omega_0,
                                                     omega_a_pi,
                                                     qub_dim=3)
        result_pi = self.backend_sim.run(
            qobj_pi, backend_options=backend_options_pi).result()
        statevector_pi = result_pi.get_statevector()

        exp_statevector_pi = self._analytic_statevector_three_level(omega_a_pi)

        # compare vectors element-wise
        for i, _ in enumerate(statevector_pi):
            self.assertAlmostEqual(statevector_pi[i],
                                   exp_statevector_pi[i],
                                   places=4)

        # Test 2*pi pulse
        omega_a_2pi = 2 * np.pi / self.drive_samples

        schedule_2pi = self.single_pulse_schedule(phi)
        qobj_params_2pi = self.qobj_params_1q(omega_d0=omega_d0)
        qobj_2pi = self.create_qobj(shots=shots,
                                    meas_level=2,
                                    schedule=schedule_2pi,
                                    qobj_params=qobj_params_2pi)
        # set qub_dim=3 in hamiltonian
        backend_options_2pi = self.backend_options_1q(omega_0,
                                                      omega_a_2pi,
                                                      qub_dim=3)
        result_2pi = self.backend_sim.run(
            qobj_2pi, backend_options=backend_options_2pi).result()
        statevector_2pi = result_2pi.get_statevector()

        exp_statevector_2pi = self._analytic_statevector_three_level(
            omega_a_2pi)

        # compare vectors element-wise
        for i, _ in enumerate(statevector_2pi):
            self.assertAlmostEqual(statevector_2pi[i],
                                   exp_statevector_2pi[i],
                                   places=4)

    # ----------------------------------------------------------------------------------------------
    # Test qubit interaction (use 2 qubits for simplicity)
    # For these tests, we use a different 2-qubit Hamiltonian that tests both
    # interaction and control (U) channels. In the lab frame, it is given
    # by `H = -\frac{1}{2} \omega_0 \sigma_z^0 + \frac{1}{2} \omega_a e^{i \omega_{d0} t} \sigma_x^0
    # `+ \omega_i (e^{i (\omega_{d0}-\omega_{d1}) t} \sigma_{p0} \otimes \sigma_{m1} + `
    # `+ e^{-i (\omega_{d0}-\omega_{d1}) t} \sigma_{m0} \otimes \sigma_{p1})`. First 2 terms allow
    # us to excite the 0 qubit. Latter 2 terms define the interaction.
    # ----------------------------------------------------------------------------------------------

    def test_interaction(self):
        r"""Test 2 qubit interaction via swap gates."""

        shots = 100000

        # Do a standard SWAP gate

        # Interaction amp (any non-zero creates the swap gate)
        omega_i_swap = np.pi / 2 / self.drive_samples
        # set omega_d0=omega_0 (resonance)
        omega_0 = 2 * np.pi * self.freq_qubit_0
        omega_d0 = omega_0

        # For swapping, set omega_d1 = 0 (drive on Q0 resonance)
        omega_d1_swap = 0

        # do pi pulse on Q0 and verify state swaps from '01' to '10' (reverse bit order)

        # Q0 drive amp -> pi pulse
        omega_a_pi_swap = np.pi / self.drive_samples

        schedule_pi_swap = self.schedule_2q()
        qobj_params_pi_swap = self.qobj_params_2q(omega_d0=omega_d0,
                                                  omega_d1=omega_d1_swap)

        qobj_pi_swap = self.create_qobj(shots=shots,
                                        meas_level=2,
                                        schedule=schedule_pi_swap,
                                        qobj_params=qobj_params_pi_swap)

        backend_options_pi_swap = self.backend_options_2q(
            omega_0, omega_a_pi_swap, omega_i_swap)
        result_pi_swap = self.backend_sim.run(
            qobj_pi_swap, backend_options=backend_options_pi_swap).result()
        counts_pi_swap = result_pi_swap.get_counts()

        exp_counts_pi_swap = {
            '10': shots
        }  # reverse bit order (qiskit convention)
        self.assertDictAlmostEqual(counts_pi_swap, exp_counts_pi_swap, delta=2)

        # do pi/2 pulse on Q0 and verify half the counts are '00' and half are swapped state '10'

        # Q0 drive amp -> pi/2 pulse
        omega_a_pi2_swap = np.pi / 2 / self.drive_samples

        schedule_pi2_swap = self.schedule_2q()
        qobj_params_pi2_swap = self.qobj_params_2q(omega_d0=omega_d0,
                                                   omega_d1=omega_d1_swap)

        qobj_pi2_swap = self.create_qobj(shots=shots,
                                         meas_level=2,
                                         schedule=schedule_pi2_swap,
                                         qobj_params=qobj_params_pi2_swap)

        backend_options_pi2_swap = self.backend_options_2q(
            omega_0, omega_a_pi2_swap, omega_i_swap)
        result_pi2_swap = self.backend_sim.run(
            qobj_pi2_swap, backend_options=backend_options_pi2_swap).result()
        counts_pi2_swap = result_pi2_swap.get_counts()

        # compare proportions for improved accuracy
        prop_pi2_swap = {}
        for key in counts_pi2_swap.keys():
            prop_pi2_swap[key] = counts_pi2_swap[key] / shots

        exp_prop_pi2_swap = {'00': 0.5, '10': 0.5}  # reverse bit order

        self.assertDictAlmostEqual(prop_pi2_swap,
                                   exp_prop_pi2_swap,
                                   delta=0.01)

        # Test that no SWAP occurs when omega_i=0 (no interaction)
        omega_i_no_swap = 0

        # Set arbitrary params for omega_d0, omega_d1
        omega_d1_no_swap = omega_d0

        # Q0 drive amp -> pi pulse
        omega_a_no_swap = np.pi / self.drive_samples

        schedule_no_swap = self.schedule_2q()
        qobj_params_no_swap = self.qobj_params_2q(omega_d0=omega_d0,
                                                  omega_d1=omega_d1_no_swap)

        qobj_no_swap = self.create_qobj(shots=shots,
                                        meas_level=2,
                                        schedule=schedule_no_swap,
                                        qobj_params=qobj_params_no_swap)
        backend_options_swap = self.backend_options_2q(omega_0,
                                                       omega_a_no_swap,
                                                       omega_i_no_swap)
        result_no_swap = self.backend_sim.run(
            qobj_no_swap, backend_options=backend_options_swap).result()
        counts_no_swap = result_no_swap.get_counts()

        exp_counts_no_swap = {
            '01': shots
        }  # non-swapped state (reverse bit order)
        self.assertDictAlmostEqual(counts_no_swap, exp_counts_no_swap)


if __name__ == '__main__':
    unittest.main()
