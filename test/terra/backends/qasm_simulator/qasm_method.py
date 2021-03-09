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
from qiskit.providers.aer import AerError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import amplitude_damping_error


@ddt
class QasmMethodTests:
    """QasmSimulator method option tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_backend_method_clifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method != 'automatic':
            self.compare_result_metadata(result, circuits, 'method', method)
        else:
            self.compare_result_metadata(result, circuits, 'method',
                                         'stabilizer')

    def test_backend_method_clifford_circuits_and_reset_noise(self):
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

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method != 'automatic':
            self.compare_result_metadata(result, circuits, 'method', method)
        else:
            self.compare_result_metadata(result, circuits, 'method',
                                         'stabilizer')

    def test_backend_method_clifford_circuits_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        self.assertTrue(success)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method != 'automatic':
            self.compare_result_metadata(result, circuits, 'method', method)
        else:
            self.compare_result_metadata(result, circuits, 'method',
                                         'stabilizer')

    def test_backend_method_clifford_circuits_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    def test_backend_method_clifford_circuits_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    # ---------------------------------------------------------------------
    # Test non-Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_backend_method_nonclifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(
            qobj, **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'statevector'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    def test_backend_method_nonclifford_circuit_and_reset_noise(self):
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

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    def test_backend_method_nonclifford_circuit_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    def test_backend_method_nonclifford_circuit_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    def test_backend_method_nonclifford_circuit_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(
            final_measure=True)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)

        result = self.SIMULATOR.run(qobj,
                                    noise_model=noise_model,
                                    **self.BACKEND_OPTS).result()
        success = getattr(result, 'success', False)
        # Check simulation method
        method = self.BACKEND_OPTS.get('method', 'automatic')
        if method == 'stabilizer':
            self.assertFalse(success)
        else:
            self.assertTrue(success)
            if method == 'automatic':
                target_method = 'density_matrix'
            else:
                target_method = method
            self.compare_result_metadata(result, circuits, 'method',
                                         target_method)

    @data('automatic', 'statevector', 'density_matrix', 'stabilizer',
          'matrix_product_state', 'extended_stabilizer')
    def test_option_basis_gates(self, method):
        """Test setting method and noise model has correct basis_gates"""
        config = QasmSimulator(method=method).configuration()
        noise_gates = ['id', 'sx', 'x', 'cx']
        noise_model = NoiseModel(basis_gates=noise_gates)
        target_gates = sorted(set(config.basis_gates).intersection(noise_gates).union(
            config.custom_instructions))

        sim = QasmSimulator(method=method, noise_model=noise_model)
        basis_gates = sim.configuration().basis_gates
        self.assertEqual(basis_gates, target_gates)

    @data('automatic', 'statevector', 'density_matrix', 'stabilizer',
          'matrix_product_state', 'extended_stabilizer')
    def test_option_order_basis_gates(self, method):
        """Test order of setting method and noise model gives same basis gates"""
        noise_model = NoiseModel(basis_gates=['id', 'sx', 'x', 'cx'])
        sim1 = QasmSimulator(method=method, noise_model=noise_model)
        basis_gates1 = sim1.configuration().basis_gates
        sim2 = QasmSimulator(noise_model=noise_model, method=method)
        basis_gates2 = sim2.configuration().basis_gates
        self.assertEqual(basis_gates1, basis_gates2)
