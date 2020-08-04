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
from qiskit.pulse import (Schedule, Play, ShiftPhase, SetPhase, Delay, Acquire, SamplePulse,
                          DriveChannel, ControlChannel, AcquireChannel, MemorySlot)
from qiskit.providers.aer.pulse.de.DE_Methods import ScipyODE
from qiskit.providers.aer.pulse.de.DE_Options import DE_Options
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.providers.models.backendconfiguration import UchannelLO

from .pulse_sim_independent import (simulate_1q_model, simulate_2q_exchange_model,
                                    simulate_3d_oscillator_model)


class TestPulseSimulator(common.QiskitAerTestCase):
    r"""PulseSimulator tests."""

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
        self.Y = np.array([[0., -1j], [1j, 0.]])
        self.Z = np.array([[1., 0.], [0., -1.]])

    # ---------------------------------------------------------------------
    # Test single qubit gates
    # ---------------------------------------------------------------------

    def test_x_gate(self):
        """Test a schedule for a pi pulse on a 2 level system."""

        # qubit frequency and drive frequency
        omega_0 = 1.1329824
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 100

        system_model = self._system_model_1Q(omega_0, r)

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

        # set up and run independent simulation
        samples = np.ones((total_samples, 1))

        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_0]), samples, 1.)

        # approximate analytic solution
        phases = np.exp(-1j * 2 * np.pi * omega_0 * total_samples * np.array([1., -1.]) / 2)
        approx_yf = phases * np.array([0., -1j])

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 256}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_x_gate_rwa(self):
        """Test a schedule for a pi pulse on a 2 level system in the rotating frame with a
        the rotating wave approximation."""

        # qubit frequency and drive frequency
        omega_0 = 0.
        omega_d = omega_0

        # drive strength and length of pulse
        # in rotating wave with RWA the drive strength is halved
        r = 0.01 / 2
        total_samples = 100

        system_model = self._system_model_1Q(omega_0, r)

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
                        shots=1)

        # set backend backend_options including initial state
        y0 = np.array([1.0, 0.0])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        # run simulation
        result = self.backend_sim.run(qobj,
                                      system_model=system_model,
                                      backend_options=backend_options).result()
        pulse_sim_yf = result.get_statevector()

        # expected final state
        yf = np.array([0., -1j])

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-10**-5)


    def test_x_half_gate(self):
        """Test a schedule for a pi/2 pulse on a 2 level system. Same setup as test_x_gate but
        with half the time."""

        # qubit frequency and drive frequency
        omega_0 = 1.1329824
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 50

        system_model = self._system_model_1Q(omega_0, r)

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

        # set up and run independent simulation
        samples = np.ones((total_samples, 1))

        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_d]), samples, 1.)

        # approximate analytic solution
        phases = np.exp(-1j * 2 * np.pi * omega_0 * total_samples * np.array([1., -1.]) / 2)
        approx_yf = phases * (expm(-1j * (np.pi / 4) * self.X) @ y0)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 132, '0': 124}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_y_half_gate(self):
        """Test a schedule for a pi/2 pulse about the y axis on a 2 level system.
        Same setup as test_x_half_gate but with amplitude of pulse 1j."""

        # qubit frequency and drive frequency
        omega_0 = 1.1329824
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 50

        system_model = self._system_model_1Q(omega_0, r)

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

        # set up and run independent simulation
        samples = 1j * np.ones((total_samples, 1))

        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_d]), samples, 1.)

        # approximate analytic solution
        phases = np.exp(-1j * 2 * np.pi * omega_0 * total_samples * np.array([1., -1.]) / 2)
        approx_yf = phases * (expm(-1j * (np.pi / 4) * self.Y) @ y0)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

        # test counts
        counts = result.get_counts()
        exp_counts = {'1': 131, '0': 125}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_1Q_noise(self):
        """Tests simulation of noise operators. Uses the same schedule as test_x_gate, but
        with a high level of amplitude damping noise.
        """

        # qubit frequency and drive frequency
        omega_0 = 1.1329824
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 100

        system_model = self._system_model_1Q(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=2,
                        shots=10)

        # set seed for simulation, and set noise
        y0 = np.array([1., 0.])
        backend_options = {'seed' : 9000, 'initial_state' : y0}
        backend_options['noise_model'] = {"qubit": {"0": {"Sm": 1.}}}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results
        # This level of noise is high enough that all counts should yield 0,
        # whereas in the noiseless simulation (in test_x_gate) all counts yield 1
        counts = result.get_counts()
        exp_counts = {'0': 10}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_unitary_parallel(self):
        """Test for parallel solving in unitary simulation. Uses same schedule as test_x_gate but
        runs it twice to trigger parallel execution.
        """
        # qubit frequency and drive frequency
        omega_0 = 1.
        omega_d = omega_0

        # drive strength and length of pulse
        r = 0.01
        total_samples = 50

        system_model = self._system_model_1Q(omega_0, r)

        # set up constant pulse for doing a pi pulse
        schedule = self._1Q_constant_sched(total_samples)

        # set up schedule and qobj
        qobj = assemble([schedule, schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d],
                        memory_slots=1,
                        shots=256)

        # set backend backend_options
        y0 = np.array([1., 0.])
        backend_options = backend_options = {'seed' : 9000, 'initial_state' : y0}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results, checking both runs in parallel
        counts = result.get_counts()
        exp_counts0 = {'1': 132, '0': 124}
        exp_counts1 = {'0': 147, '1': 109}
        self.assertDictAlmostEqual(counts[0], exp_counts0)
        self.assertDictAlmostEqual(counts[1], exp_counts1)


    def test_dt_scaling_x_gate(self):
        """Test that dt is being used correctly by the solver."""

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
            system_model = self._system_model_1Q(omega_0, r)
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

            # set up and run independent simulation
            samples = np.ones((total_samples, 1))

            indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_0]), samples, scale)

            # approximate analytic solution
            phases = np.exp(-1j * 2 * np.pi * omega_0 * total_samples * np.array([1., -1.]) / 2)
            approx_yf = phases * np.array([0., -1j])

            # test final state
            self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)
            self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

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

                system_model = self._system_model_1Q(omega_0, r_vals[i])
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

                # set up and run independent simulation
                samples = np.exp(-1j * phase_vals[i]) * np.ones((total_samples, 1))

                indep_yf = simulate_1q_model(y0, omega_0, r_vals[i], np.array([omega_d_vals[i]]), samples, 1.)

                # approximate analytic solution
                phases = np.exp(-1j * 2 * np.pi * omega_d_vals[i] * total_samples * np.array([1., -1.]) / 2)
                detuning = omega_0 - omega_d_vals[i]
                amp = np.exp(-1j * phase_vals[i])
                rwa_ham = 2 * np.pi * (detuning * self.Z / 2 + r_vals[i] * np.array([[0, amp.conj()], [amp, 0.]]) / 4)
                approx_yf = phases * (expm(-1j * rwa_ham * total_samples) @ y0)

                # test final state
                self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)
                self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

    def test_3d_oscillator(self):
        """Test simulation of a duffing oscillator truncated to 3 dimensions."""

        total_samples = 100

        freq = 5.
        anharm = -0.33

        # Test pi pulse
        r = 0.5 / total_samples

        system_model = self._system_model_3d_oscillator(freq, anharm, r)
        schedule = self._1Q_constant_sched(total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[freq],
                        shots=1)
        backend_options = {'seed' : 9000}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()


        # set up and run independent simulation
        y0 = np.array([1., 0., 0.])
        samples = np.ones((total_samples, 1))
        indep_yf = simulate_3d_oscillator_model(y0, freq, anharm, r, np.array([freq]), samples, 1.)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)


        # Test some irregular value
        r = 1.49815 / total_samples

        system_model = self._system_model_3d_oscillator(freq, anharm, r)
        schedule = self._1Q_constant_sched(total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[freq],
                        shots=1)

        y0 = np.array([0., 0., 1.])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        samples = np.ones((total_samples, 1))
        indep_yf = simulate_3d_oscillator_model(y0, freq, anharm, r, np.array([freq]), samples, 1.)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)

    def test_2Q_interaction(self):
        r"""Test 2 qubit interaction via controlled operations using u channels."""

        total_samples = 100

        # set coupling term and drive channels to 0 frequency
        j = 0.5 / total_samples
        omega_d0 = 0.
        omega_d1 = 0.

        system_model = self._system_model_2Q(j)

        schedule = self._2Q_constant_sched(total_samples)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d0, omega_d1],
                        memory_slots=2,
                        shots=1)

        y0 = np.kron(np.array([1., 0.]), np.array([0., 1.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        # exact analytic solution
        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))

        # run with different initial state
        y0 = np.kron(np.array([1., 0.]), np.array([1., 0.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        # exact analytic solution
        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))


    def test_subsystem_restriction(self):
        r"""Test behavior of subsystem_list subsystem restriction"""

        total_samples = 100

        # set coupling term and drive channels to 0 frequency
        j = 0.5 / total_samples
        omega_d = 0.

        subsystem_list = [0, 2]
        system_model = self._system_model_3Q(j, subsystem_list=subsystem_list)

        schedule = self._3Q_constant_sched(total_samples, u_idx=0, subsystem_list=subsystem_list)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d, omega_d, omega_d],
                        memory_slots=2,
                        shots=1)

        y0 = np.kron(np.array([1., 0.]), np.array([0., 1.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))

        y0 = np.kron(np.array([1., 0.]), np.array([1., 0.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))

        subsystem_list = [1, 2]
        system_model = self._system_model_3Q(j, subsystem_list=subsystem_list)

        schedule = self._3Q_constant_sched(total_samples, u_idx=1, subsystem_list=subsystem_list)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d, omega_d, omega_d],
                        memory_slots=2,
                        shots=1)

        y0 = np.kron(np.array([1., 0.]), np.array([0., 1.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))

        y0 = np.kron(np.array([1., 0.]), np.array([1., 0.]))
        backend_options = {'seed' : 9000, 'initial_state': y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        yf = expm(-1j * 0.5 * 2 * np.pi * np.kron(self.X, self.Z) / 4) @ y0

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1 - (10**-5))

    def test_simulation_without_variables(self):
        r"""Test behavior of subsystem_list subsystem restriction.
        Same setup as test_x_gate, but with explicit Hamiltonian construction without
        variables
        """

        ham_dict = {'h_str': ['np.pi*Z0', '0.02*np.pi*X0||D0'], 'qub': {'0': 2}}
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
        schedule = self._1Q_constant_sched(total_samples)
        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[1.],
                        memory_slots=2,
                        shots=256)

        # set backend backend_options
        backend_options = {'seed' : 9000, 'initial_state' : np.array([1., 0.])}

        # run simulation
        result = self.backend_sim.run(qobj, system_model=system_model,
                                      backend_options=backend_options).result()

        # test results
        counts = result.get_counts()
        exp_counts = {'1': 256}
        self.assertDictAlmostEqual(counts, exp_counts)

    def test_meas_level_1(self):
        """Test measurement level 1. """

        shots = 10000  # run large number of shots for good proportions

        total_samples = 100
        omega_0 = 1.
        omega_d = omega_0

        # Require omega_a*time = pi to implement pi pulse (x gate)
        # num of samples gives time
        r = 1. /  (2 * total_samples)

        system_model = self._system_model_1Q(omega_0, r)

        amp = np.exp(-1j * np.pi / 2)
        schedule = self._1Q_constant_sched(total_samples, amp=amp)

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=1,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[1.],
                        memory_slots=2,
                        shots=shots)

        # set backend backend_options
        y0 = np.array([1.0, 0.0])
        backend_options = {'seed' : 9000, 'initial_state' : y0}
        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        samples = amp * np.ones((total_samples, 1))
        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_d]), samples, 1.)

        # test final state
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1-10**-5)

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

        # Test gaussian drive results for a few different sigma
        gauss_sigmas = [total_samples / 6, total_samples / 3, total_samples]

        system_model = self._system_model_1Q(omega_0, r)

        for gauss_sigma in gauss_sigmas:
            with self.subTest(gauss_sigma=gauss_sigma):
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
                                shots=1)
                y0 = np.array([1., 0.])
                backend_options = {'seed' : 9000, 'initial_state' : y0}

                result = self.backend_sim.run(qobj, system_model, backend_options).result()
                pulse_sim_yf = result.get_statevector()

                # run independent simulation
                yf = simulate_1q_model(y0, omega_0, r, np.array([omega_d]), gaussian_samples, 1.)

                # Check fidelity of statevectors
                self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-(10**-5))

    def test_2Q_exchange(self):
        r"""Test a more complicated 2q simulation"""

        q_freqs = [5., 5.1]
        r = 0.02
        j = 0.02
        total_samples = 25

        hamiltonian = {}
        hamiltonian['h_str'] = ['2*np.pi*v0*0.5*Z0',
                                '2*np.pi*v1*0.5*Z1',
                                '2*np.pi*r*0.5*X0||D0',
                                '2*np.pi*r*0.5*X1||D1',
                                '2*np.pi*j*0.5*I0*I1',
                                '2*np.pi*j*0.5*X0*X1',
                                '2*np.pi*j*0.5*Y0*Y1',
                                '2*np.pi*j*0.5*Z0*Z1']
        hamiltonian['vars'] = {'v0': q_freqs[0],
                               'v1': q_freqs[1],
                               'r': r,
                               'j': j}
        hamiltonian['qub'] = {'0': 2, '1': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        # set the U0 to have frequency of drive channel 0
        u_channel_lo = []
        subsystem_list = [0, 1]
        dt = 1.

        system_model = PulseSystemModel(hamiltonian=ham_model,
                                        u_channel_lo=u_channel_lo,
                                        subsystem_list=subsystem_list,
                                        dt=dt)

        # try some random schedule
        schedule = Schedule()
        drive_pulse = SamplePulse(np.ones(total_samples))
        schedule += Play(drive_pulse, DriveChannel(0))
        schedule |= Play(drive_pulse, DriveChannel(1)) << 2 * total_samples

        schedule |= Acquire(total_samples,
                            AcquireChannel(0),
                            MemorySlot(0)) << 3 * total_samples
        schedule |= Acquire(total_samples,
                            AcquireChannel(1),
                            MemorySlot(1)) << 3 * total_samples

        qobj = assemble([schedule],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=q_freqs,
                        memory_slots=2,
                        shots=1000)
        y0 = np.array([1., 0., 0., 0.])
        backend_options = {'seed' : 9000, 'initial_state' : y0}

        result = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = result.get_statevector()

        # set up and run independent simulation
        d0_samps = np.concatenate((np.ones(total_samples), np.zeros(2 * total_samples)))
        d1_samps = np.concatenate((np.zeros(2 * total_samples), np.ones(total_samples)))
        samples = np.array([d0_samps, d1_samps]).transpose()
        q_freqs = np.array(q_freqs)
        yf = simulate_2q_exchange_model(y0, q_freqs, r, j, q_freqs, samples, 1.)

        # Check fidelity of statevectors
        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, yf), 1-(10**-5))

    def test_delay_instruction(self):
        """Test for delay instruction."""

        # construct system model specifically for this
        hamiltonian = {}
        hamiltonian['h_str'] = ['0.5*r*X0||D0', '0.5*r*Y0||D1']
        hamiltonian['vars'] = {'r': np.pi}
        hamiltonian['qub'] = {'0': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        system_model = PulseSystemModel(hamiltonian=ham_model,
                                        u_channel_lo=u_channel_lo,
                                        subsystem_list=subsystem_list,
                                        dt=dt)

        # construct a schedule that should result in a unitary -Z if delays are correctly handled
        # i.e. do a pi rotation about x, sandwiched by pi/2 rotations about y in opposite directions
        # so that the x rotation is transformed into a z rotation.
        # if delays are not handled correctly this process should fail
        sched = Schedule()
        sched += Play(SamplePulse([0.5]), DriveChannel(1))
        sched += Delay(1, DriveChannel(1))
        sched += Play(SamplePulse([-0.5]), DriveChannel(1))

        sched += Delay(1, DriveChannel(0))
        sched += Play(SamplePulse([1.]), DriveChannel(0))

        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[0., 0.],
                        memory_slots=2,
                        shots=1)


        # Result of schedule should be the unitary -1j*Z, so check rotation of an X eigenstate
        backend_options = {'initial_state': np.array([1., 1.]) / np.sqrt(2)}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()

        statevector = results.get_statevector()
        expected_vector = np.array([-1j, 1j]) / np.sqrt(2)

        self.assertGreaterEqual(state_fidelity(statevector, expected_vector), 1 - (10**-5))

        # verify validity of simulation when no delays included
        sched = Schedule()
        sched += Play(SamplePulse([0.5]), DriveChannel(1))
        sched += Play(SamplePulse([-0.5]), DriveChannel(1))

        sched += Play(SamplePulse([1.]), DriveChannel(0))

        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[0., 0.],
                        memory_slots=2,
                        shots=1)

        backend_options = {'initial_state': np.array([1., 1.]) / np.sqrt(2)}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()

        statevector = results.get_statevector()
        U = expm(1j * np.pi * self.Y /4) @ expm(-1j * np.pi * (self.Y / 4 + self.X / 2))
        expected_vector = U @ np.array([1., 1.]) / np.sqrt(2)

        self.assertGreaterEqual(state_fidelity(statevector, expected_vector), 1 - (10**-5))

    def test_shift_phase(self):
        """Test ShiftPhase command."""

        omega_0 = 1.123
        r = 1.

        system_model = self._system_model_1Q(omega_0, r)

        # run a schedule in which a shifted phase causes a pulse to cancel itself.
        # Also do it in multiple phase shifts to test accumulation
        sched = Schedule()
        amp1 = 0.12
        sched += Play(SamplePulse([amp1]), DriveChannel(0))
        phi1 = 0.12374 * np.pi
        sched += ShiftPhase(phi1, DriveChannel(0))
        amp2 = 0.492
        sched += Play(SamplePulse([amp2]), DriveChannel(0))
        phi2 = 0.5839 * np.pi
        sched += ShiftPhase(phi2, DriveChannel(0))
        amp3 = 0.12 + 0.21 * 1j
        sched += Play(SamplePulse([amp3]), DriveChannel(0))

        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_0],
                        memory_slots=2,
                        shots=1)

        y0 = np.array([1., 0])
        backend_options = {'initial_state': y0}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = results.get_statevector()

        #run independent simulation
        samples = np.array([[amp1],
                            [amp2 * np.exp(1j * phi1)],
                            [amp3 * np.exp(1j * (phi1 + phi2))]])
        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_0]), samples, 1.)

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1 - (10**-5))

        # run another schedule with only a single shift phase to verify
        sched = Schedule()
        amp1 = 0.12
        sched += Play(SamplePulse([amp1]), DriveChannel(0))
        phi1 = 0.12374 * np.pi
        sched += ShiftPhase(phi1, DriveChannel(0))
        amp2 = 0.492
        sched += Play(SamplePulse([amp2]), DriveChannel(0))
        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_0],
                        memory_slots=2,
                        shots=1)

        y0 = np.array([1., 0])
        backend_options = {'initial_state': y0}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = results.get_statevector()

        #run independent simulation
        samples = np.array([[amp1], [amp2 * np.exp(1j * phi1)]])
        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_0]), samples, 1.)

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1 - (10**-5))

    def test_set_phase(self):
        """Test SetPhase command. Similar to the ShiftPhase test but includes a mixing of
        ShiftPhase and SetPhase instructions to test relative vs absolute changes"""

        omega_0 = 1.3981
        r = 1.

        system_model = self._system_model_1Q(omega_0, r)

        # intermix shift and set phase instructions to verify absolute v.s. relative changes
        sched = Schedule()
        amp1 = 0.12
        sched += Play(SamplePulse([amp1]), DriveChannel(0))
        phi1 = 0.12374 * np.pi
        sched += ShiftPhase(phi1, DriveChannel(0))
        amp2 = 0.492
        sched += Play(SamplePulse([amp2]), DriveChannel(0))
        phi2 = 0.5839 * np.pi
        sched += SetPhase(phi2, DriveChannel(0))
        amp3 = 0.12 + 0.21 * 1j
        sched += Play(SamplePulse([amp3]), DriveChannel(0))
        phi3 = 0.1 * np.pi
        sched += ShiftPhase(phi3, DriveChannel(0))
        amp4 = 0.2 + 0.3 * 1j
        sched += Play(SamplePulse([amp4]), DriveChannel(0))

        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_0],
                        memory_slots=2,
                        shots=1)

        y0 = np.array([1., 0.])
        backend_options = {'initial_state': y0}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = results.get_statevector()

        #run independent simulation
        samples = np.array([[amp1],
                            [amp2 * np.exp(1j * phi1)],
                            [amp3 * np.exp(1j * phi2)],
                            [amp4 * np.exp(1j * (phi2 + phi3))]])
        indep_yf = simulate_1q_model(y0, omega_0, r, np.array([omega_0]), samples, 1.)

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, indep_yf), 1 - (10**-5))

    def test_set_phase_rwa(self):
        """Test SetPhase command using an RWA approximate solution."""
        omega_0 = 5.123
        r = 0.01

        system_model = self._system_model_1Q(omega_0, r)

        sched = Schedule()
        sched += SetPhase(np.pi / 2, DriveChannel(0))
        sched += Play(SamplePulse(np.ones(100)), DriveChannel(0))

        sched |= Acquire(1, AcquireChannel(0), MemorySlot(0)) << sched.duration

        qobj = assemble([sched],
                        backend=self.backend_sim,
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_0],
                        memory_slots=2,
                        shots=1)

        y0 = np.array([1., 1.]) / np.sqrt(2)
        backend_options = {'initial_state': y0}

        results = self.backend_sim.run(qobj, system_model, backend_options).result()
        pulse_sim_yf = results.get_statevector()

        #run independent simulation
        phases = np.exp((-1j * 2 * np.pi * omega_0 * np.array([1, -1]) / 2) * 100)
        approx_yf = phases * (expm(-1j * (np.pi / 2) * self.Y) @ y0)

        self.assertGreaterEqual(state_fidelity(pulse_sim_yf, approx_yf), 0.99)

    def _system_model_1Q(self, omega_0, r):
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

    def _system_model_2Q(self, j):
        """Constructs a model for a 2 qubit system with a U channel controlling coupling and
        no other Hamiltonian terms.

        Args:
            j (float): coupling strength

        Returns:
            PulseSystemModel: model for qubit system
        """

        hamiltonian = {}
        hamiltonian['h_str'] = ['a*X0||D0', 'a*X0||D1', '2*np.pi*j*0.25*(Z0*X1)||U0']
        hamiltonian['vars'] = {'a': 0, 'j': j}
        hamiltonian['qub'] = {'0': 2, '1': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        # set the U0 to have frequency of drive channel 0
        u_channel_lo = [[UchannelLO(0, 1.0+0.0j)]]
        subsystem_list = [0, 1]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)


    def _2Q_constant_sched(self, total_samples, amp=1., u_idx=0):
        """Creates a runnable schedule with a single pulse on a U channel for two qubits.

        Args:
            total_samples (int): length of pulse
            amp (float): amplitude of constant pulse (can be complex)
            u_idx (int): index of U channel

        Returns:
            schedule (pulse schedule): schedule with a drive pulse followed by an acquire
        """

        # set up constant pulse for doing a pi pulse
        drive_pulse = SamplePulse(amp * np.ones(total_samples))
        schedule = Schedule()
        schedule |= Play(drive_pulse, ControlChannel(u_idx))
        schedule |= Acquire(total_samples, AcquireChannel(0), MemorySlot(0)) << total_samples
        schedule |= Acquire(total_samples, AcquireChannel(1), MemorySlot(1)) << total_samples

        return schedule


    def _system_model_3Q(self, j, subsystem_list=[0, 2]):
        """Constructs a model for a 3 qubit system, with the goal that the restriction to
        [0, 2] and to qubits [1, 2] is the same as in _system_model_2Q

        Args:
            j (float): coupling strength
            subsystem_list (list): list of subsystems to include

        Returns:
            PulseSystemModel: model for qubit system
        """

        hamiltonian = {}
        hamiltonian['h_str'] = ['2*np.pi*j*0.25*(Z0*X2)||U0', '2*np.pi*j*0.25*(Z1*X2)||U1']
        hamiltonian['vars'] = {'j': j}
        hamiltonian['qub'] = {'0': 2, '1': 2, '2': 2}
        ham_model = HamiltonianModel.from_dict(hamiltonian, subsystem_list=subsystem_list)

        # set the U0 to have frequency of drive channel 0
        u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)], [UchannelLO(0, 1.0 + 0.0j)]]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

    def _3Q_constant_sched(self, total_samples, amp=1., u_idx=0, subsystem_list=[0, 2]):
        """Creates a runnable schedule for the 3Q system after the system is restricted to
        2 qubits.

        Args:
            total_samples (int): length of pulse
            amp (float): amplitude of constant pulse (can be complex)
            u_idx (int): index of U channel
            subsystem_list (list): list of qubits to restrict to

        Returns:
            schedule (pulse schedule): schedule with a drive pulse followed by an acquire
        """

        # set up constant pulse for doing a pi pulse
        drive_pulse = SamplePulse(amp * np.ones(total_samples))
        schedule = Schedule()
        schedule |= Play(drive_pulse, ControlChannel(u_idx))
        for idx in subsystem_list:
            schedule |= Acquire(total_samples,
                                AcquireChannel(idx),
                                MemorySlot(idx)) << total_samples

        return schedule

    def _system_model_3d_oscillator(self, freq, anharm, r):
        """Model for a duffing oscillator truncated to 3 dimensions.

        Args:
            freq (float): frequency of the oscillator
            anharm (float): anharmonicity of the oscillator
            r (float): drive strength

        Returns:
            PulseSystemModel: model for oscillator system
        """
        hamiltonian = {}
        hamiltonian['h_str'] = ['np.pi*(2*v-alpha)*O0',
                                'np.pi*alpha*O0*O0',
                                '2*np.pi*r*X0||D0']
        hamiltonian['vars'] = {'v' : freq, 'alpha': anharm, 'r': r}
        hamiltonian['qub'] = {'0': 3}
        ham_model = HamiltonianModel.from_dict(hamiltonian)

        u_channel_lo = []
        subsystem_list = [0]
        dt = 1.

        return PulseSystemModel(hamiltonian=ham_model,
                                u_channel_lo=u_channel_lo,
                                subsystem_list=subsystem_list,
                                dt=dt)

if __name__ == '__main__':
    unittest.main()
