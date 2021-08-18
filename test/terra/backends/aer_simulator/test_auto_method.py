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

from ddt import ddt, data

from test.terra.reference import ref_2q_clifford
from test.terra.reference import ref_non_clifford
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import amplitude_damping_error
from .aer_simulator_test_case import (
    AerSimulatorTestCase, supported_methods)

SUPPORTED_METHODS = [
    'automatic', 'stabilizer', 'statevector', 'density_matrix',
    'matrix_product_state', 'extended_stabilizer'
]


@ddt
class TestSimulationMethod(AerSimulatorTestCase):
    """AerSimulator method option tests."""

    # ---------------------------------------------------------------------
    # Test Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------

    def test_auto_method_clifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        backend = self.backend()
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_auto_method_clifford_circuits_and_reset_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Test noise model
        noise_circs = [[{
            "name": "reset",
            "qubits": [0]
        }], [{
            "name": "id",
            "qubits": [0]
        }]]
        noise_probs = [0.5, 0.5]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_auto_method_clifford_circuits_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_auto_method_clifford_circuits_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

    def test_auto_method_clifford_circuits_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

    # ---------------------------------------------------------------------
    # Test non-Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_auto_method_nonclifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        backend = self.backend()
        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "statevector")

    def test_auto_method_nonclifford_circuit_and_reset_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Test noise model
        noise_circs = [[{
            "name": "reset",
            "qubits": [0]
        }], [{
            "name": "id",
            "qubits": [0]
        }]]
        noise_probs = [0.5, 0.5]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

    def test_auto_method_nonclifford_circuit_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

    def test_auto_method_nonclifford_circuit_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

    def test_auto_method_nonclifford_circuit_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])
        backend = self.backend(noise_model=noise_model)

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        result = backend.run(circuits, shots=shots).result()
        success = getattr(result, 'success', False)
        self.compare_result_metadata(result, circuits, 'method', "density_matrix")

