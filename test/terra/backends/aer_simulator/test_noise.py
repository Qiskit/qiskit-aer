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

from ddt import ddt
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_reset_noise
from test.terra.reference import ref_kraus_noise

from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator
from qiskit import execute
from qiskit.circuit.library import QFT
import qiskit.quantum_info as qi
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)

ALL_METHODS = [
    'automatic', 'stabilizer', 'statevector', 'density_matrix',
    'matrix_product_state', 'extended_stabilizer'
]


@ddt
class QasmReadoutNoiseTests(SimulatorTestCase):
    """QasmSimulator readout error noise model tests."""

    @supported_methods(ALL_METHODS)
    def test_readout_noise(self, method, device):
        """Test simulation with classical readout error noise model."""
        backend = self.backend(method=method, device=device)
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_readout_noise.readout_error_circuits()
        noise_models = ref_readout_noise.readout_error_noise_models()
        targets = ref_readout_noise.readout_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_pauli_gate_noise(self, method, device):
        """Test simulation with Pauli gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        with self.assertWarns(DeprecationWarning):
            noise_models = ref_pauli_noise.pauli_gate_error_noise_models()
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods([
        'automatic', 'stabilizer', 'statevector', 'density_matrix',
        'matrix_product_state', 'extended_stabilizer'])
    def test_pauli_reset_noise(self, method, device):
        """Test simulation with Pauli reset error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_reset_error_circuits()
        with self.assertWarns(DeprecationWarning):
            noise_models = ref_pauli_noise.pauli_reset_error_noise_models()
        targets = ref_pauli_noise.pauli_reset_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_pauli_measure_noise(self, method, device):
        """Test simulation with Pauli measure error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_measure_error_circuits()
        with self.assertWarns(DeprecationWarning):
            noise_models = ref_pauli_noise.pauli_measure_error_noise_models()
        targets = ref_pauli_noise.pauli_measure_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_reset_gate_noise(self, method, device):
        """Test simulation with reset gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_reset_noise.reset_gate_error_circuits()
        noise_models = ref_reset_noise.reset_gate_error_noise_models()
        targets = ref_reset_noise.reset_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)


    @supported_methods([
        'automatic', 'statevector', 'density_matrix', 'matrix_product_state'])
    def test_kraus_gate_noise(self, method, device):
        """Test simulation with Kraus gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_kraus_noise.kraus_gate_error_circuits()
        noise_models = ref_kraus_noise.kraus_gate_error_noise_models()
        targets = ref_kraus_noise.kraus_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods([
        'automatic', 'statevector', 'density_matrix', 'matrix_product_state'])
    def test_kraus_gate_noise_on_QFT(self, method, device):
        """Test Kraus noise on a QFT circuit"""
        shots = 10000
        noise_model = ref_kraus_noise.kraus_gate_error_noise_models_full()
        backend = self.backend(
            method=method, device=device, noise_model=noise_model)
        circuit = QFT(3).decompose()
        circuit.measure_all()
        ref_target = ref_kraus_noise.kraus_gate_error_counts_on_QFT(shots)
        result = backend.run(circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(result, [circuit], [ref_target], delta=0.1 * shots)
