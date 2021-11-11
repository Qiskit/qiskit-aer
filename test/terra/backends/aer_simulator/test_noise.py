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
from qiskit.circuit.library import QFT
import qiskit.quantum_info as qi
from qiskit.providers.aer import noise

from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_reset_noise
from test.terra.reference import ref_kraus_noise

ALL_METHODS = [
    'automatic', 'stabilizer', 'statevector', 'density_matrix',
    'matrix_product_state', 'extended_stabilizer'
]


@ddt
class TestNoise(SimulatorTestCase):
    """AerSimulator readout error noise model tests."""

    @supported_methods(ALL_METHODS)
    def test_empty_circuit_noise(self, method, device):
        """Test simulation with empty circuit and noise model."""
        backend = self.backend(method=method, device=device)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(0.1, 1), ['x'])
        result = backend.run(
            QuantumCircuit(), shots=1, noise_model=noise_model).result()
        self.assertSuccess(result)

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

    @supported_methods(ALL_METHODS)
    def test_clifford_circuit_noise(self, method, device):
        """Test simulation with mixed Clifford quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error1 = noise.QuantumError([
            ([{"name": "id", "qubits": [0]}], 0.8),
            ([{"name": "reset", "qubits": [0]}], 0.1),
            ([{"name": "h", "qubits": [0]}], 0.1)])

        error2 = noise.QuantumError([
            ([{"name": "id", "qubits": [0]}], 0.75),
            ([{"name": "reset", "qubits": [0]}], 0.1),
            ([{"name": "reset", "qubits": [1]}], 0.1),
            ([{"name": "reset", "qubits": [0]},
              {"name": "reset", "qubits": [1]}], 0.05)])

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(error1, [0])
        qc.cx(0, 1)
        qc.append(error2, [0, 1])
        target_probs = qi.DensityMatrix(qc).probabilities_dict()

        # Add measurement
        qc.measure_all()
        result = backend.run(qc, shots=shots).result()
        self.assertSuccess(result)
        probs = {key: val / shots for key, val in result.get_counts(0).items()}
        self.assertDictAlmostEqual(target_probs, probs, delta=0.1)

    @supported_methods(['automatic', 'statevector', 'density_matrix'])
    def test_kraus_circuit_noise(self, method, device):
        """Test simulation with Kraus quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error1 = noise.amplitude_damping_error(0.05)
        error2 = error1.tensor(error1)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(error1, [0])
        qc.cx(0, 1)
        qc.append(error2, [0, 1])
        target_probs = qi.DensityMatrix(qc).probabilities_dict()

        # Add measurement
        qc.measure_all()
        result = backend.run(qc, shots=shots).result()
        self.assertSuccess(result)
        probs = {key: val / shots for key, val in result.get_counts(0).items()}
        self.assertDictAlmostEqual(target_probs, probs, delta=0.1)
