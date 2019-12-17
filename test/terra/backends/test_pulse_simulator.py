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
import pdb
import unittest
from test.terra import common

import numpy as np
from scipy.linalg import expm
from scipy.special import erf

import qiskit
import qiskit.pulse as pulse

from qiskit.compiler import assemble
from qiskit.quantum_info import state_fidelity

from qiskit.test.mock.fake_openpulse_2q import FakeOpenPulse2Q
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

    def system_model_1q(self, omega_0, omega_a, qub_dim=2):
        """Creates system_model for 1 qubit pulse simulation
        """
        backend_mock = FakeOpenPulse2Q()
        backend_mock.configuration().hamiltonian = self.create_ham_1q(omega_0, omega_a, qub_dim)
        system_model = SystemModel.from_backend(self.backend_mock, qubit_list=[0])
        return system_model

    def backend_options_1q(self):
        backend_options = {}
        backend_options['seed'] = 9000
        return backend_options

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
        backend_options['seed'] = 9000
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



    # ---------------------------------------------------------------------
    # Test FrameChange and PersistentValue commands
    # ---------------------------------------------------------------------


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
