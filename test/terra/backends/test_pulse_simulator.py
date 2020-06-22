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

import sys
import unittest
import functools
from test.terra import common

import numpy as np
from scipy.linalg import expm
from scipy.special import erf

from qiskit.providers.aer.backends import PulseSimulator

from qiskit.compiler import assemble
from qiskit.quantum_info import state_fidelity
from qiskit.pulse import (Schedule, Play, ShiftPhase, Acquire, SamplePulse, DriveChannel,
                          ControlChannel, AcquireChannel, MemorySlot)
from qiskit.providers.aer.pulse.de.DE_Methods import ScipyODE
from qiskit.providers.aer.pulse.de.DE_Options import DE_Options
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.providers.models.backendconfiguration import UchannelLO


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
        """ Set configuration settings for pulse simulator
        WARNING: We do not support Python 3.5 because the digest algorithm relies on dictionary insertion order.
        This "feature" was introduced later on Python 3.6 and there's no official support for OrderedDict in the C API so
        Python 3.5 support has been disabled while looking for a propper fix.
        """
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
           self.skipTest("We don't support Python 3.5 for Pulse simulator")

        # Get pulse simulator backend
        self.backend_sim = PulseSimulator()

        self.X = np.array([[0., 1.], [1., 0.]])

    # ---------------------------------------------------------------------
    # Test single qubit gates
    # ---------------------------------------------------------------------

    def test_x_gate(self):
        """Test a schedule for a pi pulse on a 2 level system."""

        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 100

        system_model = self._system_model_1Q_new(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        # set up schedule and qobj
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=1,
                        shots=256)

        # set backend backend_options including initial state
        y0 = np.array([1.0, 0.0])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        # run simulation
        result = self.backend_sim.run(qobj,
                                      system_model=system_model,
                                      backend_options=backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = self._independent_1Q_constant_sched_sim(y0,
                                                     amp=1.,
                                                     r=r,
                                                     omega_d=omega_d,
                                                     T=total_samples)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 256}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_x_half_gate(self):
        """Test a schedule for a pi/2 pulse on a 2 level system. Same setup as test_x_gate but
        with half the time."""

        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 50

        system_model = self._system_model_1Q_new(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        # set up schedule and qobj
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=1,
                        shots=256)

        # set backend backend_options
        y0 = np.array([1.0, 0.0])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        # run simulation
        result = self.backend_sim.run(qobj,
                                      system_model=system_model,
                                      backend_options=backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = self._independent_1Q_constant_sched_sim(y0,
                                                     amp=1.,
                                                     r=r,
                                                     omega_d=omega_d,
                                                     T=total_samples)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 132, '0': 124}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_y_half_gate(self):
        """Test a schedule for a pi/2 pulse about the y axis on a 2 level system.
        Same setup as test_x_half_gate but with half the time."""

        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 50

        system_model = self._system_model_1Q_new(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples, amp=1j)

        # set up schedule and qobj
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=1,
                        shots=256)

        # set backend backend_options
        y0 = np.array([1.0, 0.0])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        # run simulation
        result = self.backend_sim.run(qobj,
                                      system_model=system_model,
                                      backend_options=backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = self._independent_1Q_constant_sched_sim(y0,
                                                     amp=1j,
                                                     r=r,
                                                     omega_d=omega_d,
                                                     T=total_samples)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 132, '0': 124}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_1Q_noise(self):
        """
        Tests simulation of noise operators. Uses the same schedule as test_x_gate, but
        with a high level of amplitude damping noise.
        """

        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 100

        system_model = self._system_model_1Q_new(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=2,
                        shots=100)

        # set seed for simulation, and set noise
        backend_options = {'seed' : 9000}
        backend_options['noise_model'] = {"qubit": {"0": {"Sm": 1.}}}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results
        # This level of noise is high enough that all counts should yield 0,
        # whereas in the noiseless simulation (in test_x_gate) all counts yield 1
        counts = result.get_counts()
        exp_counts = {'0': 100}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_unitary_parallel(self):
        """
        Test for parallel solving in unitary simulation. Uses same schedule as test_x_gate but
        runs it twice to trigger parallel execution.
        """
        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 100

        system_model = self._system_model_1Q_new(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        qobj = assemble([schedule, schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=2,
                        shots=256)

        # set backend backend_options
        backend_options = {'seed' : 9000}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results, checking both runs in parallel
        counts = result.get_counts()
        exp_counts = {'1': 256}
        self.assertDictAlmostEqual(counts[0], exp_counts)
        self.assertDictAlmostEqual(counts[1], exp_counts)


    def test_dt_scaling_x_gate(self):
        """
        Test that dt is being used correctly by the solver.
        """

        total_samples = 100
        # do the same thing as test_x_gate, but scale dt and all frequency parameters
        # define test case for a single scaling
        def scale_test(scale):

            # qubit frequency and drive frequency
            omega_0 = 1. / scale
            omega_d = omega_0

            # drive strength and length of pulse
            r = 0.01 / scale
            total_samples = 100

            # set up system model and scale time
            system_model = self._system_model_1Q_new(omega_0, r)
            system_model.dt = system_model.dt * scale

            # set up constant pulse for doing a pi pulse
            schedule = self._1Q_constant_sched(total_samples)

            qobj = assemble([schedule],
                            backend=self.backend_sim,
                            meas_level=2,
                            meas_return='single',
                            meas_map=[[0]],
                            qubit_lo_freq=[omega_d],
                            memory_slots=2,
                            shots=256)

            # set backend backend_options
            y0 = np.array([1., 0.])
            backend_options = {'seed' : 9000, 'initial_state': y0}

            # run simulation
            result = self.backend_sim.run(qobj, system_model=system_model,
                                          backend_options=backend_options).result()

            pulse_sim_yf = result.get_statevector()

            yf = self._independent_1Q_constant_sched_sim(y0,
                                                         amp=1.,
                                                         r=r,
                                                         omega_d=omega_d,
                                                         T=total_samples * scale)

            # test final state
            self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)


            counts = result.get_counts()
            exp_counts = {'1': 256}

            self.assertDictAlmostEqual(counts, exp_counts)

        # set scales and run tests
        scales = [2., 0.1234, 10.**5, 10**-5]
        for scale in scales:
            scale_test(scale)

    def test_arbitrary_constant_drive(self):
        """Test a few examples w/ arbitary drive, phase and amplitude. """
        total_samples = 100
        num_tests = 3

        omega_0 = 1.
        omega_d_vals = [omega_0 + 1., omega_0 + 0.02, omega_0 + 0.005]
        r_vals = [3 / total_samples, 5 / total_samples, 0.1]
        phase_vals = [5 * np.pi / 7, 19 * np.pi / 14, np.pi / 4]

        for i in range(num_tests):
            with self.subTest(i=i):

                system_model = self._system_model_1Q_new(omega_0, r_vals[i])
                schedule = self._1Q_constant_sched(total_samples, amp=np.exp(-1j * phase_vals[i]))

                qobj = assemble([schedule],
                                backend=self.backend_sim,
                                meas_level=2,
                                meas_return='single',
                                meas_map=[[0]],
                                qubit_lo_freq=[omega_d_vals[i]],
                                memory_slots=2,
                                shots=1)

                # Run qobj and compare prop to expected result
                y0 = np.array([1., 0.])
                backend_options = {'seed' : 9000, 'initial_state' : y0}
                result = self.backend_sim.run(qobj, system_model, backend_options).result()

                pulse_sim_yf = result.get_statevector()

                yf = self._independent_1Q_constant_sched_sim(y0,
                                                             amp=np.exp(-1j * phase_vals[i]),
                                                             r=r_vals[i],
                                                             omega_d=omega_d_vals[i],
                                                             T=total_samples,
                                                             detuning=omega_0 - omega_d_vals[i])

                # test final state
                self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)

    def test_gaussian_drive(self):
        """Test gaussian drive pulse using meas_level_2. Set omega_d0=omega_0 (drive on resonance),
        phi=0, omega_a = pi/time
        """

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        total_samples = 100
        omega_0 = 1.
        omega_d = omega_0

        # Require omega_a*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        r = np.pi / total_samples

        phi = 0

        # Test gaussian drive results for a few different sigma
        gauss_sigmas = [total_samples / 6, total_samples / 3, total_samples]

        system_model = self._system_model_1Q_new(omega_0, r)

        for gauss_sigma in gauss_sigmas:
            with self.subTest(gauss_sigma=gauss_sigma):
                schedule = self._simple_1Q_schedule(phi,
                                                    total_samples,
                                                    "gaussian",
                                                    gauss_sigma)

                times = 1.0 * np.arange(total_samples)
                gaussian_samples = np.exp(-times**2 / 2 / gauss_sigma**2)
                drive_pulse = SamplePulse(gaussian_samples, name='drive_pulse')

                # construct schedule
                schedule = Schedule()
                schedule |= Play(drive_pulse, DriveChannel(0))
                schedule |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << schedule.duration

                qobj = assemble([schedule],
                                backend=self.backend_sim,
                                meas_level=2,
                                meas_return='single',
                                meas_map=[[0]],
                                qubit_lo_freq=[omega_d],
                                memory_slots=2,
                                shots=1000)
                y0 = np.array([1., 0.])
                backend_options = {'seed' : 9000, 'initial_state' : y0}

                result = self.backend_sim.run(qobj, system_model, backend_options).result()
                pulse_sim_yf = result.get_statevector()

                # run independent simulation
                T = total_samples
                def rhs(t, y):
                    # handle time edge case
                    if t >= 100:
                        t = 99.9
                    amp = gaussian_samples[int(t)]
                    chan_val = np.real(amp * np.exp(1j * 2 * np.pi * omega_d * t))
                    phi = 2 * np.pi * omega_d * t
                    Xrot = np.array([[0, np.exp(1j * phi)], [np.exp(-1j * phi), 0.]])
                    return (-1j * 2 * np.pi * r * chan_val * Xrot / 2) @ y

                de_options = DE_Options(method='RK45', atol=10**-8, rtol=10**-8)
                ode_method = ScipyODE(t0=0., y0=y0, rhs=rhs, options=de_options)
                ode_method.integrate(T)
                yf = np.exp(np.array([-1j * np.pi * omega_d * T, 1j * np.pi * omega_d * T])) * ode_method.y

                # Check fidelity of statevectors
                self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 0.99)


    def test_meas_level_1(self):
        """Test measurement level 1. """

        shots = 10000  # run large number of shots for good proportions
        total_samples = 100
        # perform hadamard setup (so get some 0's and some 1's), but use meas_level = 1

        # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi
        omega_d0 = omega_0

        # Require omega_a*time = pi/2 to implement pi/2 rotation pulse
        # num of samples gives time
        omega_a = np.pi / 2 / total_samples

        phi = -np.pi / 2

        system_model = self._system_model_1Q(omega_0, omega_a)

        phi = -np.pi / 2
        schedule = self._simple_1Q_schedule(phi, total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=1,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d0/(2*np.pi)],
                        memory_slots=2,
                        shots=shots)

        # set backend backend_options
        backend_options = {'seed' : 9000}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()

        # Verify that (about) half the IQ vals have abs val 1 and half have abs val 0
        # (use prop for easier comparison)
        mem = np.abs(result.get_memory()[:, 0])

        iq_prop = {'0': 0, '1': 0}
        for i in mem:
            if i == 0:
                iq_prop['0'] += 1 / shots
            else:
                iq_prop['1'] += 1 / shots

        exp_prop = {'0': 0.5, '1': 0.5}

        self.assertDictAlmostEqual(iq_prop, exp_prop, delta=0.01)

    def test_three_level(self):
        r"""Test 3 level system. Compare statevectors as counts only use bitstrings. Analytic form
        given in _analytic_statevector_3level function docstring.
        """

        def analytic_state_vector(omega_a, total_samples):
            r"""Returns analytically computed statevector for 3 level system with our Hamiltonian.
            Is given by `(\frac{1}{3} (2+\cos(\frac{\sqrt{3}}{2} \omega_a t)),
            -\frac{i}{\sqrt{3}} \sin(\frac{\sqrt{3}}{2} \omega_a t),
            -\frac{2\sqrt{2}}{3} \sin(\frac{\sqrt{3}}{4} \omega_a t)^2)`.
            Args:
                omega_a (float): Q0 drive amplitude
                total_samples (int): number of samples to use in pulses_idx
            Returns:
                exp_statevector (list): analytically computed statevector with Hamiltonian from
                    above (Returned in the rotating frame)
            """
            time = total_samples
            arg1 = np.sqrt(3) * omega_a * time / 2  # cos arg for first component
            arg2 = arg1  # sin arg for first component
            arg3 = arg1 / 2  # sin arg for 3rd component
            exp_statevector = np.array([(2 + np.cos(arg1)) / 3,
                                        -1j * np.sin(arg2) / np.sqrt(3),
                                        -2 * np.sqrt(2) * np.sin(arg3)**2 / 3],
                                       dtype=complex)
            return exp_statevector


        shots = 1000
        total_samples = 100
        # Set omega_0,omega_d0 (use qubit frequency) -> drive on resonance
        omega_0 = 2 * np.pi
        omega_d0 = omega_0

        # Set phi = 0 for simplicity
        phi = 0

        # Test pi pulse
        omega_a = np.pi / total_samples

        system_model = self._system_model_1Q(omega_0, omega_a, qubit_dim=3)
        schedule = self._simple_1Q_schedule(phi, total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d0/(2*np.pi)],
                        memory_slots=2,
                        shots=shots)
        backend_options = {'seed' : 9000}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        statevector = result.get_statevector()

        exp_statevector = analytic_state_vector(omega_a, total_samples)

        # Check fidelity of statevectors
        self.assertGreaterEqual(
            state_fidelity(statevector, exp_statevector), 0.99)

        # Test 2*pi pulse
        omega_a = 2 * np.pi / total_samples

        system_model = self._system_model_1Q(omega_0, omega_a, qubit_dim=3)
        schedule = self._simple_1Q_schedule(phi, total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d0/(2*np.pi)],
                        memory_slots=2,
                        shots=shots)
        backend_options = {'seed' : 9000}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        statevector = result.get_statevector()

        exp_statevector = analytic_state_vector(omega_a, total_samples)

        # Check fidelity of vectors
        self.assertGreaterEqual(
            state_fidelity(statevector, exp_statevector), 0.99)

    def test_interaction(self):
        r"""Test 2 qubit interaction via swap gates."""

        shots = 100000
        total_samples = 100
        # Do a standard SWAP gate

        # Interaction amp (any non-zero creates the swap gate)
        omega_i_swap = np.pi / 2 / total_samples
        # set omega_d0=omega_0 (resonance)
        omega_0 = 2 * np.pi
        omega_d0 = omega_0

        # For swapping, set omega_d1 = 0 (drive on Q0 resonance)
        # Note: confused by this as there is no d1 term
        omega_d1 = 0

        # do pi pulse on Q0 and verify state swaps from '01' to '10' (reverse bit order)

        # Q0 drive amp -> pi pulse
        omega_a_pi_swap = np.pi / total_samples


        system_model = self._system_model_2Q(omega_0, omega_a_pi_swap, omega_i_swap)

        schedule = self._schedule_2Q_interaction(total_samples)
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0, 1]],
                        qubit_lo_freq=[omega_d0 / (2 * np.pi), omega_d1 / (2 * np.pi)],
                        memory_slots=2,
                        shots=shots)
        backend_options = {'seed': 12387}

        result_pi_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_pi_swap = result_pi_swap.get_counts()

        exp_counts_pi_swap = {
            '10': shots
        }  # reverse bit order (qiskit convention)
        self.assertDictAlmostEqual(counts_pi_swap, exp_counts_pi_swap, delta=2)

        # do pi/2 pulse on Q0 and verify half the counts are '00' and half are swapped state '10'

        # Q0 drive amp -> pi/2 pulse
        omega_a_pi2_swap = np.pi / 2 / total_samples

        system_model = self._system_model_2Q(omega_0, omega_a_pi2_swap, omega_i_swap)

        result_pi2_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
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

        # Q0 drive amp -> pi pulse
        omega_a_no_swap = np.pi / total_samples
        system_model = self._system_model_2Q(omega_0, omega_a_no_swap, omega_i_no_swap)

        result_no_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_no_swap = result_no_swap.get_counts()

        exp_counts_no_swap = {
            '01': shots
        }  # non-swapped state (reverse bit order)
        self.assertDictAlmostEqual(counts_no_swap, exp_counts_no_swap)

    def test_subsystem_restriction(self):
        r"""Test behavior of subsystem_list subsystem restriction"""

        shots = 100000
        total_samples = 100
        # Do a standard SWAP gate

        # Interaction amp (any non-zero creates the swap gate)
        omega_i_swap = np.pi / 2 / total_samples
        # set omega_d0=omega_0 (resonance)
        omega_0 = 2 * np.pi
        omega_d0 = omega_0

        # For swapping, set omega_d1 = 0 (drive on Q0 resonance)
        # Note: confused by this as there is no d1 term
        omega_d1 = 0

        # do pi pulse on Q0 and verify state swaps from '01' to '10' (reverse bit order)

        # Q0 drive amp -> pi pulse
        omega_a_pi_swap = np.pi / total_samples


        subsystem_list = [0, 2]
        system_model = self._system_model_3Q(omega_0,
                                             omega_a_pi_swap,
                                             omega_i_swap,
                                             subsystem_list=subsystem_list)

        qubit_lo_freq = system_model.hamiltonian.get_qubit_lo_from_drift()
        schedule = self._schedule_2Q_interaction(total_samples, drive_idx=0, target_idx=2, U_idx=1)
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[subsystem_list],
                        qubit_lo_freq=qubit_lo_freq,
                        memory_slots=2,
                        shots=shots)
        backend_options = {'seed': 12387}

        result_pi_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_pi_swap = result_pi_swap.get_counts()

        exp_counts_pi_swap = {
            '100': shots
        }  # reverse bit order (qiskit convention)
        self.assertDictAlmostEqual(counts_pi_swap, exp_counts_pi_swap, delta=2)

        # do pi/2 pulse on Q0 and verify half the counts are '00' and half are swapped state '10'

        # Q0 drive amp -> pi/2 pulse
        omega_a_pi2_swap = np.pi / 2 / total_samples

        system_model = self._system_model_3Q(omega_0,
                                             omega_a_pi2_swap,
                                             omega_i_swap,
                                             subsystem_list=subsystem_list)

        result_pi2_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_pi2_swap = result_pi2_swap.get_counts()

        # compare proportions for improved accuracy
        prop_pi2_swap = {}
        for key in counts_pi2_swap.keys():
            prop_pi2_swap[key] = counts_pi2_swap[key] / shots

        exp_prop_pi2_swap = {'000': 0.5, '100': 0.5}  # reverse bit order

        self.assertDictAlmostEqual(prop_pi2_swap,
                                   exp_prop_pi2_swap,
                                   delta=0.01)

        # Test that no SWAP occurs when omega_i=0 (no interaction)
        omega_i_no_swap = 0

        # Q0 drive amp -> pi pulse
        omega_a_no_swap = np.pi / total_samples
        system_model = self._system_model_3Q(omega_0,
                                             omega_a_no_swap,
                                             omega_i_no_swap,
                                             subsystem_list=subsystem_list)

        result_no_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_no_swap = result_no_swap.get_counts()

        exp_counts_no_swap = {
            '001': shots
        }  # non-swapped state (reverse bit order)
        self.assertDictAlmostEqual(counts_no_swap, exp_counts_no_swap)

        shots = 100000
        total_samples = 100
        omega_i_swap = np.pi / 2 / total_samples
        omega_0 = 2 * np.pi
        omega_d0 = omega_0
        omega_d1 = 0
        omega_a_pi_swap = np.pi / total_samples

        subsystem_list = [1, 2]
        system_model = self._system_model_3Q(omega_0,
                                             omega_a_pi_swap,
                                             omega_i_swap,
                                             subsystem_list=subsystem_list)
        qubit_lo_freq = system_model.hamiltonian.get_qubit_lo_from_drift()
        schedule = self._schedule_2Q_interaction(total_samples, drive_idx=1, target_idx=2, U_idx=2)
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[subsystem_list],
                        qubit_lo_freq=qubit_lo_freq,
                        memory_slots=2,
                        shots=shots)
        backend_options = {'seed': 12387}
        result_pi_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_pi_swap = result_pi_swap.get_counts()

        exp_counts_pi_swap = {
            '100': shots
        }  # reverse bit order (qiskit convention)
        self.assertDictAlmostEqual(counts_pi_swap, exp_counts_pi_swap, delta=2)

        # do pi/2 pulse on Q0 and verify half the counts are '00' and half are swapped state '10'

        # Q0 drive amp -> pi/2 pulse
        omega_a_pi2_swap = np.pi / 2 / total_samples

        system_model = self._system_model_3Q(omega_0,
                                             omega_a_pi2_swap,
                                             omega_i_swap,
                                             subsystem_list=subsystem_list)

        result_pi2_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_pi2_swap = result_pi2_swap.get_counts()

        # compare proportions for improved accuracy
        prop_pi2_swap = {}
        for key in counts_pi2_swap.keys():
            prop_pi2_swap[key] = counts_pi2_swap[key] / shots

        exp_prop_pi2_swap = {'000': 0.5, '100': 0.5}  # reverse bit order

        self.assertDictAlmostEqual(prop_pi2_swap,
                                   exp_prop_pi2_swap,
                                   delta=0.01)

        # Test that no SWAP occurs when omega_i=0 (no interaction)
        omega_i_no_swap = 0

        # Q0 drive amp -> pi pulse
        omega_a_no_swap = np.pi / total_samples
        system_model = self._system_model_3Q(omega_0,
                                             omega_a_no_swap,
                                             omega_i_no_swap,
                                             subsystem_list=subsystem_list)

        result_no_swap = self.backend_sim.run(qobj, system_model, backend_options).result()
        counts_no_swap = result_no_swap.get_counts()

        exp_counts_no_swap = {
            '010': shots
        }  # non-swapped state (reverse bit order)
        self.assertDictAlmostEqual(counts_no_swap, exp_counts_no_swap)

    def test_simulation_without_variables(self):
        r"""Test behavior of subsystem_list subsystem restriction.
        Same setup as test_x_gate, but with explicit Hamiltonian construction without
        variables
        """

        ham_dict = {'h_str': ['-np.pi*Z0', '0.01*np.pi*X0||D0'], 'qub': {'0': 2}}
        ham_model = HamiltonianModel.from_dict(ham_dict)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        system_model = PulseSystemModel(hamiltonian=ham_model,
                                        u_channel_lo=u_channel_lo,
                                        subsystem_list=subsystem_list,
                                        dt=dt)

        # set up schedule and qobj
        total_samples = 50
        schedule = self._simple_1Q_schedule(0, total_samples)
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[1.],
                        memory_slots=2,
                        shots=256)

        # set backend backend_options
        backend_options = {'seed' : 9000}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results
        counts = result.get_counts()
        exp_counts = {'1': 256}
        self.assertDictAlmostEqual(counts, exp_counts)

    def _system_model_1Q_new(self, omega_0, r):
        """Constructs a standard model for a 1 qubit system.

        Args:
            omega_0 (float): qubit frequency
            r (float): drive strength

        Returns:
            PulseSystemModel: model for qubit system
        """

        hamiltonian = {}
        hamiltonian['h_str'] = ['2*np.pi*omega0*0.5*Z0', '2*np.pi*r*0.5*X0||D0']
        hamiltonian['vars'] = {'omega0': omega_0, 'r': r}
        hamiltonian['qub'] = {'0': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _1Q_constant_sched(self, total_samples, amp=1.):
        """Creates a runnable schedule for 1Q with a constant drive pulse of a given length.

        Args:
            total_samples (int): length of pulse
            amp (float): amplitude of constant pulse (can be complex)

        Returns:
            schedule (pulse schedule): schedule with a drive pulse followed by an acquire
        """

        # set up constant pulse for doing a pi pulse
        drive_pulse = SamplePulse(amp * np.ones(total_samples))
        schedule = Schedule()
        schedule |= Play(drive_pulse, DriveChannel(0))
        schedule |= Acquire(total_samples, AcquireChannel(0), MemorySlot(0)) << schedule.duration

        return schedule

    def _independent_1Q_constant_sched_sim(self, y0, amp, r, omega_d, T, detuning=0.):

        def rhs(t, y):
            chan_val = np.real(amp * np.exp(1j * 2 * np.pi * omega_d * t))
            phi = 2 * np.pi * omega_d * t
            Xrot = np.array([[0, np.exp(1j * phi)], [np.exp(-1j * phi), 0.]])
            return -1j * (2 * np.pi * detuning * np.diag([1, -1]) / 2 + 2 * np.pi * r * chan_val * Xrot / 2) @ y

        de_options = DE_Options(method='RK45', atol=10**-8, rtol=10**-8)
        ode_method = ScipyODE(t0=0., y0=y0, rhs=rhs, options=de_options)
        ode_method.integrate(T)
        yf = np.exp(np.array([-1j * np.pi * omega_d * T, 1j * np.pi * omega_d * T])) * ode_method.y

        return yf

    ###########
    # Old
    ###########

    def _system_model_1Q(self, omega_0, omega_a, qubit_dim=2):
        """Constructs a simple 1 qubit system model.

        Args:
            omega_0 (float): frequency of qubit
            omega_a (float): strength of drive term
            qubit_dim (int): dimension of qubit
        Returns:
            PulseSystemModel: model for qubit system
        """
        # make Hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = ['-0.5*omega0*Z0', '0.5*omegaa*X0||D0']
        hamiltonian['vars'] = {'omega0': omega_0, 'omegaa': omega_a}
        hamiltonian['qub'] = {'0': qubit_dim}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _system_model_2Q(self, omega_0, omega_a, omega_i, qubit_dim=2):
        """Constructs a simple 2 qubit system model.

        Args:
            omega_0 (float): frequency of qubit
            omega_a (float): strength of drive term
            omega_i (float): strength of interaction
            qubit_dim (int): dimension of qubit
        Returns:
            PulseSystemModel: model for qubit system
        """

        # make Hamiltonian
        hamiltonian = {}
        # qubit 0 terms
        hamiltonian['h_str'] = ['-0.5*omega0*Z0', '0.5*omegaa*X0||D0']
        # interaction term
        hamiltonian['h_str'].append('omegai*(Sp0*Sm1+Sm0*Sp1)||U1')
        hamiltonian['vars'] = {
            'omega0': omega_0,
            'omegaa': omega_a,
            'omegai': omega_i
        }
        hamiltonian['qub'] = {'0' : qubit_dim, '1' : qubit_dim}
        ham_model = HamiltonianModel.from_dict(hamiltonian)


        u_channel_lo = [
            [UchannelLO(0, 1.0+0.0j)],
            [UchannelLO(0, -1.0+0.0j),
             UchannelLO(1, 1.0+0.0j)]]
        subsystem_list = [0, 1]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _system_model_3Q(self, omega_0, omega_a, omega_i, qubit_dim=2, subsystem_list=None):
        """Constructs a 3 qubit model. Purpose of this is for testing subsystem restrictions -
        It is set up so that the system restricted to [0, 2] and [1, 2] is the same (up to
        channel labelling).

        Args:
            omega_0 (float): frequency of qubit
            omega_a (float): strength of drive term
            omega_i (float): strength of interaction
            qubit_dim (int): dimension of qubit
        Returns:
            PulseSystemModel: model for qubit system
        """

        # make Hamiltonian
        hamiltonian = {}
        # qubit 0 terms
        hamiltonian['h_str'] = ['-0.5*omega0*Z0',
                                '0.5*omegaa*X0||D0',
                                '-0.5*omega0*Z1',
                                '0.5*omegaa*X1||D1']
        # interaction terms
        hamiltonian['h_str'].append('omegai*(Sp0*Sm2+Sm0*Sp2)||U1')
        hamiltonian['h_str'].append('omegai*(Sp1*Sm2+Sm1*Sp2)||U2')
        hamiltonian['vars'] = {
            'omega0': omega_0,
            'omegaa': omega_a,
            'omegai': omega_i
        }
        hamiltonian['qub'] = {'0' : qubit_dim, '1' : qubit_dim, '2': qubit_dim}
        ham_model = HamiltonianModel.from_dict(hamiltonian, subsystem_list)


        u_channel_lo = [[UchannelLO(0, 1.0+0.0j)]]
        u_channel_lo.append([UchannelLO(0, -1.0+0.0j), UchannelLO(2, 1.0+0.0j)])
        u_channel_lo.append([UchannelLO(1, -1.0+0.0j), UchannelLO(2, 1.0+0.0j)])
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _simple_1Q_schedule(self, phi, total_samples, shape="square", gauss_sigma=0):
        """Creates schedule for single pulse test
        Args:
            phi (float): drive phase (phi in Hamiltonian)
            total_samples (int): length of pulses
            shape (str): shape of the pulse; defaults to square pulse
            gauss_sigma (float): std dev for gaussian pulse if shape=="gaussian"
        Returns:
            schedule (pulse schedule): schedule for this test
        """

        # set up pulse command
        phase = np.exp(1j * phi)
        drive_pulse = None
        if shape == "square":
            const_pulse = np.ones(total_samples)
            drive_pulse = SamplePulse(phase * const_pulse, name='drive_pulse')
        if shape == "gaussian":
            times = 1.0 * np.arange(total_samples)
            gaussian = np.exp(-times**2 / 2 / gauss_sigma**2)
            drive_pulse = SamplePulse(phase * gaussian, name='drive_pulse')

        # add commands into a schedule for first qubit
        schedule = Schedule(name='drive_pulse')
        schedule |= Play(drive_pulse, DriveChannel(0))
        schedule |= Acquire(total_samples, AcquireChannel(0), MemorySlot(0)) << schedule.duration

        return schedule

    def _1Q_frame_change_schedule(self, phi, fc_phi, total_samples, dur_drive1, dur_drive2):
        """Creates schedule for frame change test. Does a pulse w/ phase phi of duration dur_drive1,
        then frame change of phase fc_phi, then another pulse of phase phi of duration dur_drive2.
        The different durations for the pulses allow manipulation of rotation angles on Bloch sphere

        Args:
            phi (float): drive phase (phi in Hamiltonian)
            fc_phi (float): phase for frame change
            total_samples (int): length of pulses
            dur_drive1 (int): duration of first pulse
            dur_drive2 (int): duration of second pulse

        Returns:
            schedule (pulse schedule): schedule for frame change test
        """
        phase = np.exp(1j * phi)
        drive_pulse_1 = SamplePulse(phase * np.ones(dur_drive1),
                                    name='drive_pulse_1')
        drive_pulse_2 = SamplePulse(phase * np.ones(dur_drive2),
                                    name='drive_pulse_2')

        # add commands to schedule
        schedule = Schedule(name='fc_schedule')
        schedule |= Play(drive_pulse_1, DriveChannel(0))
        schedule += ShiftPhase(fc_phi, DriveChannel(0))
        schedule += Play(drive_pulse_2, DriveChannel(0))
        schedule |= Acquire(total_samples, AcquireChannel(0), MemorySlot(0)) << schedule.duration

        return schedule

    def _schedule_2Q_interaction(self, total_samples, drive_idx=0, target_idx=1, U_idx=1):
        """Creates schedule for testing two qubit interaction. Specifically, do a pi pulse on qub 0
        so it starts in the `1` state (drive channel) and then apply constant pulses to each
        qubit (on control channel 1). This will allow us to test a swap gate.

        Args:
            total_samples (int): length of pulses
        Returns:
            schedule (pulse schedule): schedule for 2q experiment
        """

        # create acquire schedule
        acq_sched = Schedule(name='acq_sched')
        acq_sched |= Acquire(total_samples, AcquireChannel(drive_idx), MemorySlot(drive_idx))
        acq_sched += Acquire(total_samples, AcquireChannel(target_idx), MemorySlot(target_idx))

        # set up const pulse
        const_pulse = SamplePulse(np.ones(total_samples), name='const_pulse')

        # add commands to schedule
        schedule = Schedule(name='2q_schedule')
        schedule |= Play(const_pulse, DriveChannel(drive_idx))
        schedule += Play(const_pulse, ControlChannel(U_idx)) << schedule.duration
        schedule |= acq_sched << schedule.duration

        return schedule


if __name__ == '__main__':
    unittest.main()
