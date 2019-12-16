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
from qiskit.quantum_info import state_fidelity

from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue
from qiskit.providers.aer.openpulse.system_model import SystemModel, HamiltonianModel


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
        # Get pulse simulator backend
        self.backend_sim = qiskit.Aer.get_backend('pulse_simulator')

    # ---------------------------------------------------------------------
    # Test single qubit gates (using meas level 2 and square drive)
    # ---------------------------------------------------------------------

    def test_x_gate(self):
        """
        Test x gate. Set omega_d0=omega_0 (drive on resonance), phi=0, omega_a = pi/time
        """

        # setup system model
        total_samples = 100
        omega_0 = 2 * np.pi
        omega_d0 = omega_0
        omega_a = np.pi / total_samples
        system_model = self._system_model_1Q(omega_0, omega_a)

        # set up schedule and qobj
        schedule = self._simple_1Q_schedule(system_model, 0, total_samples)
        qobj = assemble([schedule],
                        meas_level=2,
                        meas_return='single',
                        meas_map=[[0]],
                        qubit_lo_freq=[omega_d0/(2*np.pi)],
                        meas_lo_freq=[0.],
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

    def test_dt_scaling_x_gate(self):
        """
        Test that dt is being used correctly by the solver.
        """

        total_samples = 100

        # do the same thing as test_x_gate, but scale dt and all frequency parameters
        # define test case for a single scaling
        def scale_test(scale):
            # set omega_0, omega_d0 equal (use qubit frequency) -> drive on resonance
            omega_0 = 2 * np.pi/scale
            omega_d0 = omega_0

            # Require omega_a*time = pi to implement pi pulse (x gate)
            # num of samples gives time
            omega_a = np.pi / total_samples /scale

            phi = 0

            x_schedule = self.single_pulse_schedule(phi)
            x_qobj_params = self.qobj_params_1q(omega_d0)
            x_qobj = self.create_qobj(shots=256,
                                      meas_level=2,
                                      schedule=x_schedule,
                                      qobj_params=x_qobj_params)
            x_system_model = self.system_model_1q(omega_0, omega_a)
            x_backend_opts = self.backend_options_1q_new()
            result = self.backend_sim.run(x_qobj, system_model=x_system_model,
                                          backend_options=x_backend_opts).result()
            counts = result.get_counts()
            exp_counts = {'1': 256}

            self.assertDictAlmostEqual(counts, exp_counts)
        # set scales and run tests
        scales = [2., 1.3453, 0.1234, 10.**5, 10**-5]
        for scale in scales:
            scale_test(scale)

    def _system_model_1Q(self, omega_0, omega_a):

        # make Hamiltonian
        hamiltonian = {}
        hamiltonian['h_str'] = ['-0.5*omega0*Z0', '0.5*omegaa*X0||D0']
        hamiltonian['vars'] = {'omega0': omega_0, 'omegaa': omega_a}
        hamiltonian['qub'] = {'0': 2}
        ham_model = HamiltonianModel.from_string_spec(hamiltonian)

        u_channel_lo = []
        qubit_list = [0]
        dt = 1.

        return SystemModel(hamiltonian=ham_model,
                           u_channel_lo=u_channel_lo,
                           qubit_list=qubit_list,
                           dt=dt)

    def _simple_1Q_schedule(self, system_model, phi, total_samples, shape="square", gauss_sigma=0):
        """Creates schedule for single pulse test
        Args:
            phi (float): drive phase (phi in Hamiltonian)
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

        # set up acquire command
        acq_cmd = pulse.Acquire(duration=total_samples)

        # add commands into a schedule for first qubit
        schedule = pulse.Schedule(name='drive_pulse')
        schedule |= drive_pulse(system_model.drive(0))
        schedule |= acq_cmd(system_model.acquire(0),
                            system_model.memoryslot(0)) << schedule.duration

        return schedule

if __name__ == '__main__':
    unittest.main()
