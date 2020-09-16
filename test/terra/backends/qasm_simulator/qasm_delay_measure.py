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
QasmSimulator Integration Tests
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import ReadoutError, depolarizing_error
from test.benchmark.tools import quantum_volume_circuit, qft_circuit

class QasmDelayMeasureTests:
    """QasmSimulator delay measure sampling optimization tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}


    def delay_measure_circuit(self):
        """Test circuit that allows measure delay optimization"""
        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.measure(0, 0)
        circuit.barrier([0, 1])
        circuit.x(1)
        circuit.measure(0, 1)
        return circuit

    def test_delay_measure_enable(self):
        """Test measure sampling works with delay measure optimization"""

        # Circuit that allows delay measure
        circuit = self.delay_measure_circuit()
        shots = 100
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        # Delay measure default 
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['optimize_ideal_threshold'] = 0
        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertTrue(metadata.get('measure_sampling'))

        # Delay measure enabled
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['delay_measure_enable'] = True
        backend_options['optimize_ideal_threshold'] = 0
        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertTrue(metadata.get('measure_sampling'))

        # Delay measure disabled
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['delay_measure_enable'] = False
        backend_options['optimize_ideal_threshold'] = 0
        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertFalse(metadata.get('measure_sampling'))

    def test_delay_measure_verbose(self):
        """Test delay measure with verbose option"""
        circuit = self.delay_measure_circuit()
        shots = 100
        qobj = assemble([circuit], self.SIMULATOR, shots=shots, seed_simulator=1)

        # Delay measure verbose enabled
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['delay_measure_enable'] = True
        backend_options['delay_measure_verbose'] = True
        backend_options['optimize_ideal_threshold'] = 0

        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertIn('delay_measure_verbose', metadata)

        # Delay measure verbose disabled
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['delay_measure_enable'] = True
        backend_options['delay_measure_verbose'] = False
        backend_options['optimize_ideal_threshold'] = 0

        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertNotIn('delay_measure_verbose', metadata)

        # Delay measure verbose default
        backend_options = self.BACKEND_OPTS.copy()
        backend_options['delay_measure_enable'] = True
        backend_options['optimize_ideal_threshold'] = 1
        backend_options['optimize_noise_threshold'] = 1

        result = self.SIMULATOR.run(
            qobj,
            backend_options=backend_options).result()
        self.assertSuccess(result)
        metadata = result.results[0].metadata
        self.assertNotIn('delay_measure_verbose', metadata)
