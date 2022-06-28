# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Configurable PulseSimulator Tests
"""

import sys
import unittest
import warnings
import numpy as np
from test.terra import common

from qiskit import QuantumCircuit, transpile, schedule
from qiskit.providers.fake_provider import FakeArmonk, FakeAthens

from qiskit.providers.aer.backends import PulseSimulator
from qiskit.pulse import (Schedule, Play, ShiftPhase, SetPhase, Delay, Acquire,
                          Waveform, DriveChannel, ControlChannel,
                          AcquireChannel, MemorySlot)
from qiskit.providers.aer.aererror import AerError

from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.providers.models.backendconfiguration import UchannelLO


class TestConfigPulseSimulator(common.QiskitAerTestCase):
    r"""PulseSimulator tests."""

    def test_from_backend(self):
        """Test that configuration, defaults, and properties are correclty imported."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        self.assertEqual(athens_backend.properties(), athens_sim.properties())
        # check that configuration is correctly imported
        backend_dict = athens_backend.configuration().to_dict()
        sim_dict = athens_sim.configuration().to_dict()
        for key in sim_dict:
            if key == 'backend_name':
                self.assertEqual(sim_dict[key], 'pulse_simulator(fake_athens)')
            elif key == 'description':
                desc = 'A Pulse-based simulator configured from the backend: fake_athens'
                self.assertEqual(sim_dict[key], desc)
            elif key == 'simulator':
                self.assertTrue(sim_dict[key])
            elif key == 'local':
                self.assertTrue(sim_dict[key])
            elif key == 'parametric_pulses':
                self.assertEqual(sim_dict[key], [])
            else:
                self.assertEqual(sim_dict[key], backend_dict[key])

        backend_dict = athens_backend.defaults().to_dict()
        sim_dict = athens_sim.defaults().to_dict()
        for key in sim_dict:
            if key == 'pulse_library':
                # need to compare pulse libraries directly due to containing dictionaries
                for idx, entry in enumerate(sim_dict[key]):
                    for entry_key in entry:
                        if entry_key == 'samples':
                            self.assertTrue(np.array_equal(entry[entry_key], backend_dict[key][idx][entry_key]))
                        else:
                            self.assertTrue(entry[entry_key] == backend_dict[key][idx][entry_key])
            else:
                self.assertEqual(sim_dict[key], backend_dict[key])

    def test_from_backend_system_model(self):
        """Test that the system model is correctly imported from the backend."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # u channel lo
        athens_attr = athens_backend.configuration().u_channel_lo
        sim_attr = athens_sim.configuration().u_channel_lo
        model_attr = athens_sim._system_model.u_channel_lo
        self.assertTrue(sim_attr == athens_attr and model_attr == athens_attr)

        # dt
        athens_attr = athens_backend.configuration().dt
        sim_attr = athens_sim.configuration().dt
        model_attr = athens_sim._system_model.dt
        self.assertTrue(sim_attr == athens_attr and model_attr == athens_attr)

    def test_set_system_model_options(self):
        """Test setting of options that need to be changed in multiple places."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # u channel lo
        set_attr = [[UchannelLO(0, 1.0 + 0.0j)]]
        athens_sim.set_options(u_channel_lo=set_attr)
        sim_attr = athens_sim.configuration().u_channel_lo
        model_attr = athens_sim._system_model.u_channel_lo
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

        # dt
        set_attr = 5.
        athens_sim.set_options(dt=set_attr)
        sim_attr = athens_sim.configuration().dt
        model_attr = athens_sim._system_model.dt
        self.assertTrue(sim_attr == set_attr and model_attr == set_attr)

    def test_from_backend_parametric_pulses(self):
        """Verify that the parametric_pulses parameter is overriden in the PulseSimulator.
        Results don't matter, just need to check that it runs.
        """

        backend = PulseSimulator.from_backend(FakeAthens())

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.measure_all()

        sched = schedule(transpile(qc, backend), backend)
        res = backend.run(sched).result()

    def test_parametric_pulses_error(self):
        """Verify error is raised if a parametric pulse makes it into the digest."""

        fake_backend = FakeAthens()
        backend = PulseSimulator.from_backend(fake_backend)

        # reset parametric_pulses option
        backend.set_option('parametric_pulses', fake_backend.configuration().parametric_pulses)

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.measure_all()
        sched = schedule(transpile(qc, backend), backend)

        with self.assertRaises(AerError):
            res = backend.run(sched).result()

    def test_set_meas_levels(self):
        """Test setting of meas_levels."""

        athens_backend = FakeAthens()
        athens_sim = PulseSimulator.from_backend(athens_backend)

        # test that a warning is thrown when meas_level 0 is attempted to be set
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            athens_sim.set_options(meas_levels=[0,1,2])

            self.assertEqual(len(w), 1)
            self.assertTrue('Measurement level 0 not supported' in str(w[-1].message))
            self.assertEqual(athens_sim.configuration().meas_levels, [1, 2])

        self.assertTrue(athens_sim.configuration().meas_levels == [1, 2])

        athens_sim.set_options(meas_levels=[2])
        self.assertTrue(athens_sim.configuration().meas_levels == [2])

    def test_set_system_model_from_backend(self):
        """Test setting system model when constructing from backend."""

        armonk_backend = FakeArmonk()
        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        armonk_sim = None

        # construct backend and catch warning
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            armonk_sim = PulseSimulator.from_backend(backend=armonk_backend,
                                                     system_model=system_model)

            self.assertEqual(len(w), 1)
            self.assertTrue('inconsistencies' in str(w[-1].message))

        # check that system model properties have been imported
        self.assertEqual(armonk_sim.configuration().dt, system_model.dt)
        self.assertEqual(armonk_sim.configuration().u_channel_lo, system_model.u_channel_lo)

    def test_set_system_model_in_constructor(self):
        """Test setting system model when constructing."""

        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        # construct directly
        test_sim = None
        # construct backend and verify no warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            test_sim = PulseSimulator(system_model=system_model)

            self.assertEqual(len(w), 0)

        # check that system model properties have been imported
        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)

    def test_set_system_model_after_construction(self):
        """Test setting the system model after construction."""

        system_model = self._system_model_1Q()

        # these are 1q systems so this doesn't make sense but can still be used to test
        system_model.u_channel_lo = [[UchannelLO(0, 1.0 + 0.0j)]]

        # first test setting after construction with no hamiltonian
        test_sim = PulseSimulator()

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            test_sim.set_options(system_model=system_model)
            self.assertEqual(len(w), 0)

        # check that system model properties have been imported
        self.assertEqual(test_sim._system_model, system_model)
        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)

        # next, construct a pulse simulator with a config containing a Hamiltonian and observe
        # warnings
        armonk_backend = FakeArmonk()
        test_sim = PulseSimulator(configuration=armonk_backend.configuration())

        # add system model and verify warning is raised
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            armonk_sim = test_sim.set_options(system_model=system_model)

            self.assertEqual(len(w), 1)
            self.assertTrue('inconsistencies' in str(w[-1].message))

        self.assertEqual(test_sim.configuration().dt, system_model.dt)
        self.assertEqual(test_sim.configuration().u_channel_lo, system_model.u_channel_lo)

    def test_validation_num_acquires(self):
        """Test that validation fails if 0 or >1 acquire is given in a schedule."""

        test_sim = PulseSimulator.from_backend(FakeArmonk())
        test_sim.set_options(
            meas_level=2,
            qubit_lo_freq=test_sim.defaults().qubit_freq_est,
            meas_return='single',
            shots=256
        )

        # check that too many acquires results in an error
        sched = self._1Q_schedule(num_acquires=2)
        try:
            test_sim.run(sched, validate=True).result()
        except AerError as error:
            self.assertTrue('does not support multiple Acquire' in error.message)

        # check that no acquires results in an error
        sched = self._1Q_schedule(num_acquires=0)
        try:
            test_sim.run(sched, validate=True).result()
        except AerError as error:
            self.assertTrue('requires at least one Acquire' in error.message)

    def test_run_simulation_from_backend(self):
        """Construct from a backend and run a simulation."""
        armonk_backend = FakeArmonk()

        # manually override parameters to insulate from future changes to FakeArmonk
        freq_est = 4.97e9
        drive_est = 6.35e7
        armonk_backend.defaults().qubit_freq_est = [freq_est]
        armonk_backend.configuration().hamiltonian['h_str']= ['wq0*0.5*(I0-Z0)', 'omegad0*X0||D0']
        armonk_backend.configuration().hamiltonian['vars'] = {'wq0': 2 * np.pi * freq_est,
                                                              'omegad0': drive_est}
        armonk_backend.configuration().hamiltonian['qub'] = {'0': 2}
        dt = 2.2222222222222221e-10
        armonk_backend.configuration().dt = dt

        armonk_sim = PulseSimulator.from_backend(armonk_backend)
        armonk_sim.set_options(
            meas_level=2,
            meas_return='single',
            shots=1
        )

        total_samples = 250
        amp = np.pi / (drive_est * dt * total_samples)

        sched = self._1Q_schedule(total_samples, amp)
        # run and verify that a pi pulse had been done
        result = armonk_sim.run(sched).result()
        final_vec = result.get_statevector()
        probabilities = np.abs(final_vec)**2
        self.assertTrue(probabilities[0] < 1e-5)
        self.assertTrue(probabilities[1] > 1 - 1e-5)

    def _system_model_1Q(self, omega_0=5., r=0.02):
        """Constructs a standard model for a 1 qubit system.

        Args:
            omega_0 (float): qubit frequency
            r (float): drive strength

        Returns:
            PulseSystemModel: model for qubit system
        """

        hamiltonian = {}
        hamiltonian['h_str'] = [
            '2*np.pi*omega0*0.5*Z0', '2*np.pi*r*0.5*X0||D0'
        ]
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

    def _1Q_schedule(self, total_samples=100, amp=1., num_acquires=1):
        """Creates a schedule for a single qubit.

        Args:
            total_samples (int): number of samples in the drive pulse
            amp (complex): amplitude of drive pulse
            num_acquires (int): number of acquire instructions to include in the schedule

        Returns:
            schedule (pulse schedule):
        """

        schedule = Schedule()
        schedule |= Play(Waveform(amp * np.ones(total_samples)), DriveChannel(0))
        for _ in range(num_acquires):
            schedule |= Acquire(total_samples, AcquireChannel(0),
                                MemorySlot(0)) << schedule.duration
        return schedule


if __name__ == '__main__':
    unittest.main()
