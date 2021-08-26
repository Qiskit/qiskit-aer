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
AerSimulator Integration Tests
"""

from ddt import ddt
from qiskit import QuantumCircuit
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)


@ddt
class TestDelayMeasure(SimulatorTestCase):
    """AerSimulator delay measure sampling optimization tests."""

    def delay_measure_circuit(self):
        """Test circuit that allows measure delay optimization"""
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.measure(0, 0)
        circuit.barrier([0, 1])
        circuit.x(1)
        circuit.measure(0, 1)
        return circuit

    def test_delay_measure_default(self):
        """Test measure sampling works with delay measure optimization"""
        run_options = {
           'optimize_ideal_threshold': 0
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertTrue(metadata.get('measure_sampling'))

    def test_delay_measure_enable(self):
        """Test enable measure sampling works"""
        run_options = {
           'delay_measure_enable': True,
           'optimize_ideal_threshold': 0
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertTrue(metadata.get('measure_sampling'))

    def test_delay_measure_disable(self):
        """Test disable measure sampling works"""
        run_options = {
           'delay_measure_enable': False,
           'optimize_ideal_threshold': 0
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertFalse(metadata.get('measure_sampling'))

    def test_delay_measure_verbose_enable(self):
        """Test delay measure with verbose option"""
        run_options = {
            'delay_measure_enable': True,
            'delay_measure_verbose': True,
            'optimize_ideal_threshold': 0
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertIn('delay_measure_verbose', metadata)

    def test_delay_measure_verbose_disable(self):
        """Test delay measure with verbose option"""
        run_options = {
            'delay_measure_enable': True,
            'delay_measure_verbose': False,
            'optimize_ideal_threshold': 0
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()

        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertNotIn('delay_measure_verbose', metadata)

    def test_delay_measure_verbose_default(self):
        """Test delay measure with verbose option"""
        run_options = {
            'delay_measure_enable': True,
            'optimize_ideal_threshold': 1,
            'optimize_noise_threshold': 1
        }
        backend = self.backend()
        circuit = self.delay_measure_circuit()
        shots = 100
        result = backend.run(circuit, shots=shots, **run_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertNotIn('delay_measure_verbose', metadata)
