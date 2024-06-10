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
import numpy as np

import qiskit.quantum_info as qi
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Reset
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import IGate, HGate
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
from test.terra.reference import ref_kraus_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_pauli_lindblad_noise
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_reset_noise

ALL_METHODS = [
    "automatic",
    "stabilizer",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "extended_stabilizer",
    "tensor_network",
]


@ddt
class TestNoise(SimulatorTestCase):
    """AerSimulator readout error noise model tests."""

    @supported_methods(ALL_METHODS)
    def test_empty_circuit_noise(self, method, device):
        """Test simulation with empty circuit and noise model."""
        backend = self.backend(method=method, device=device)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(noise.depolarizing_error(0.1, 1), ["x"])
        result = backend.run(QuantumCircuit(), shots=1, noise_model=noise_model).result()
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

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_readout_noise_without_basis_gates(self, method, device):
        """Test simulation with classical readout error noise model w/o basis gates."""
        backend = self.backend(method=method, device=device)
        noise_model = noise.NoiseModel()
        noise_model.add_readout_error(np.array([[0.9, 0.1], [0.1, 0.9]]), [0])
        backend.set_options(noise_model=noise_model)
        circ = QuantumCircuit(1, 1)
        circ.reset(0)
        circ.measure(0, 0)
        circ = transpile(circ, backend)
        result = backend.run(circ, shots=1).result()
        self.assertSuccess(result)

    @supported_methods(ALL_METHODS, [noise.QuantumError, noise.PauliError])
    def test_pauli_gate_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        noise_models = ref_pauli_noise.pauli_gate_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ],
        [noise.QuantumError, noise.PauliError],
    )
    def test_pauli_reset_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli reset error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_reset_error_circuits()
        noise_models = ref_pauli_noise.pauli_reset_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_reset_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS, [noise.QuantumError, noise.PauliError])
    def test_pauli_measure_noise(self, method, device, qerror_cls):
        """Test simulation with Pauli measure error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_noise.pauli_measure_error_circuits()
        noise_models = ref_pauli_noise.pauli_measure_error_noise_models(qerror_cls)
        targets = ref_pauli_noise.pauli_measure_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_pauli_lindblad_gate_noise(self, method, device):
        """Test simulation with Pauli gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_lindblad_noise.pauli_lindblad_gate_error_circuits()
        noise_models = ref_pauli_lindblad_noise.pauli_lindblad_gate_error_noise_models()
        targets = ref_pauli_lindblad_noise.pauli_lindblad_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(
        [
            "automatic",
            "stabilizer",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_pauli_lindblad_reset_noise(self, method, device):
        """Test simulation with Pauli reset error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_lindblad_noise.pauli_lindblad_reset_error_circuits()
        noise_models = ref_pauli_lindblad_noise.pauli_lindblad_reset_error_noise_models()
        targets = ref_pauli_lindblad_noise.pauli_lindblad_reset_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(ALL_METHODS)
    def test_pauli_lindblad_measure_noise(self, method, device):
        """Test simulation with Pauli measure error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_pauli_lindblad_noise.pauli_lindblad_measure_error_circuits()
        noise_models = ref_pauli_lindblad_noise.pauli_lindblad_measure_error_noise_models()
        targets = ref_pauli_lindblad_noise.pauli_lindblad_measure_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
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

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"]
    )
    def test_kraus_gate_noise(self, method, device):
        """Test simulation with Kraus gate error noise model."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        circuits = ref_kraus_noise.kraus_gate_error_circuits()
        noise_models = ref_kraus_noise.kraus_gate_error_noise_models()
        targets = ref_kraus_noise.kraus_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models, targets):
            backend.set_options(noise_model=noise_model)
            result = backend.run(circuit, shots=shots).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def _test_kraus_gate_noise_on_QFT(self, **options):
        shots = 10000

        # Build noise model
        error1 = noise.amplitude_damping_error(0.2)
        error2 = error1.tensor(error1)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error1, ["h"])
        noise_model.add_all_qubit_quantum_error(error2, ["cp", "swap"])

        backend = self.backend(**options, noise_model=noise_model)
        ideal_circuit = transpile(QFT(3), backend)

        # manaully build noise circuit
        noise_circuit = QuantumCircuit(3)
        for inst, qargs, cargs in ideal_circuit.data:
            noise_circuit.append(inst, qargs, cargs)
            if inst.name == "h":
                noise_circuit.append(error1, qargs)
            elif inst.name in ["cp", "swap"]:
                noise_circuit.append(error2, qargs)
        # compute target counts
        noise_state = DensityMatrix(noise_circuit)
        ref_target = {i: shots * p for i, p in noise_state.probabilities_dict().items()}

        # Run sim
        ideal_circuit.measure_all()
        result = backend.run(ideal_circuit, shots=shots).result()
        self.assertSuccess(result)
        self.compare_counts(
            result, [ideal_circuit], [ref_target], hex_counts=False, delta=0.1 * shots
        )

    @supported_methods(
        ["automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"]
    )
    def test_kraus_gate_noise_on_QFT(self, method, device):
        """Test Kraus noise on a QFT circuit"""
        self._test_kraus_gate_noise_on_QFT(method=method, device=device)

    @supported_methods(["statevector", "density_matrix"])
    def test_kraus_gate_noise_on_QFT_cache_blocking(self, method, device):
        """Test Kraus noise on a QFT circuit with caceh blocking"""
        self._test_kraus_gate_noise_on_QFT(method=method, device=device, blocking_qubits=2)

    @supported_methods(ALL_METHODS)
    def test_clifford_circuit_noise(self, method, device):
        """Test simulation with mixed Clifford quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error1 = noise.QuantumError(
            [([(IGate(), [0])], 0.8), ([(Reset(), [0])], 0.1), ([(HGate(), [0])], 0.1)]
        )

        error2 = noise.QuantumError(
            [
                ([(IGate(), [0])], 0.75),
                ([(Reset(), [0])], 0.1),
                ([(Reset(), [1])], 0.1),
                ([(Reset(), [0]), (Reset(), [1])], 0.05),
            ]
        )

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

    @supported_methods(["automatic", "statevector", "density_matrix", "tensor_network"])
    def test_kraus_circuit_noise(self, method, device):
        """Test simulation with Kraus quantum errors in circuit."""
        backend = self.backend(method=method, device=device)
        shots = 1000
        error0 = noise.amplitude_damping_error(0.05)
        error1 = noise.amplitude_damping_error(0.15)
        error01 = error1.tensor(error0)

        # Target Circuit 0
        tc0 = QuantumCircuit(2)
        tc0.h(0)
        tc0.append(qi.Kraus(error0), [0])
        tc0.cx(0, 1)
        tc0.append(qi.Kraus(error01), [0, 1])
        target_probs0 = qi.DensityMatrix(tc0).probabilities_dict()

        # Sim circuit 0
        qc0 = QuantumCircuit(2)
        qc0.h(0)
        qc0.append(error0, [0])
        qc0.cx(0, 1)
        qc0.append(error01, [0, 1])
        qc0.measure_all()

        # Target Circuit 1
        tc1 = QuantumCircuit(2)
        tc1.h(1)
        tc1.append(qi.Kraus(error0), [1])
        tc1.cx(1, 0)
        tc1.append(qi.Kraus(error01), [1, 0])
        target_probs1 = qi.DensityMatrix(tc1).probabilities_dict()

        # Sim circuit 1
        qc1 = QuantumCircuit(2)
        qc1.h(1)
        qc1.append(error0, [1])
        qc1.cx(1, 0)
        qc1.append(error01, [1, 0])
        qc1.measure_all()

        result = backend.run([qc0, qc1], shots=shots).result()
        self.assertSuccess(result)
        probs = [{key: val / shots for key, val in result.get_counts(i).items()} for i in range(2)]
        self.assertDictAlmostEqual(target_probs0, probs[0], delta=0.1)
        self.assertDictAlmostEqual(target_probs1, probs[1], delta=0.1)
