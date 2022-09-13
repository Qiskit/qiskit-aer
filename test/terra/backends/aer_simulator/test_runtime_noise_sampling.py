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
from qiskit_aer import noise

import qiskit.quantum_info as qi
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Reset
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import IGate, HGate
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)
from test.terra.reference import ref_kraus_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_reset_noise


@ddt
class TestRuntimeNoiseSampling(SimulatorTestCase):
    """AerSimulator readout error noise model tests."""

    @supported_methods([
        'statevector'])
    def test_pauli_gate_noise(self, method, device):
        """Test simulation with Pauli gate error noise model."""
        backend = self.backend(method=method, device=device)
        backend.set_options(runtime_noise_sampling_enable=True)
        backend.set_options(batched_shots_gpu=False)
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
        'statevector'])
    def test_pauli_reset_noise(self, method, device):
        """Test simulation with Pauli reset error noise model."""
        backend = self.backend(method=method, device=device)
        backend.set_options(runtime_noise_sampling_enable=True)
        backend.set_options(batched_shots_gpu=False)
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

    @supported_methods([
        'statevector'])
    def test_pauli_measure_noise(self, method, device):
        """Test simulation with Pauli measure error noise model."""
        backend = self.backend(method=method, device=device)
        backend.set_options(runtime_noise_sampling_enable=True)
        backend.set_options(batched_shots_gpu=False)
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

    @supported_methods([
        'statevector'])
    def test_reset_gate_noise(self, method, device):
        """Test simulation with reset gate error noise model."""
        backend = self.backend(method=method, device=device)
        backend.set_options(runtime_noise_sampling_enable=True)
        backend.set_options(batched_shots_gpu=False)
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
        'statevector'])
    def test_clifford_circuit_noise(self, method, device):
        """Test simulation with mixed Clifford quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        backend.set_options(runtime_noise_sampling_enable=True)
        backend.set_options(batched_shots_gpu=False)
        shots = 1000
        error1 = noise.QuantumError([
            ([(IGate(), [0])], 0.8),
            ([(Reset(), [0])], 0.1),
            ([(HGate(), [0])], 0.1)])

        error2 = noise.QuantumError([
            ([(IGate(), [0])], 0.75),
            ([(Reset(), [0])], 0.1),
            ([(Reset(), [1])], 0.1),
            ([(Reset(), [0]), (Reset(), [1])], 0.05)])

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

