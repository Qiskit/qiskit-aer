# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_2q_clifford
from test.terra.utils import ref_non_clifford
from qiskit import compile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import AerError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.providers.aer.noise.errors import amplitude_damping_error


class QasmMethodTests(common.QiskitAerTestCase):
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
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        result = get_result()
        self.is_completed(result)
        if method == 'statevector':
            self.compare_result_metadata(result, circuits, 'method', 'statevector')
        else:
            self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_backend_method_clifford_circuits_and_reset_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Test noise model
        noise_circs = [[{"name": "reset", "qubits": [0]}],
                       [{"name": "id", "qubits": [0]}]]
        noise_probs = [0.5, 0.5]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        result = get_result()
        self.is_completed(result)
        if method == 'statevector':
            self.compare_result_metadata(result, circuits, 'method', 'statevector')
        else:
            self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_backend_method_clifford_circuits_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        if method == 'statevector':
            self.compare_result_metadata(result, circuits, 'method', 'statevector')
        else:
            self.compare_result_metadata(result, circuits, 'method', 'stabilizer')

    def test_backend_method_clifford_circuits_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    def test_backend_method_clifford_circuits_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_2q_clifford.cz_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    # ---------------------------------------------------------------------
    # Test non-Clifford circuits with clifford and non-clifford noise
    # ---------------------------------------------------------------------
    def test_backend_method_nonclifford_circuits(self):
        """Test statevector method is used for Clifford circuit"""
        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    def test_backend_method_nonclifford_circuit_and_reset_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Test noise model
        noise_circs = [[{"name": "reset", "qubits": [0]}],
                       [{"name": "id", "qubits": [0]}]]
        noise_probs = [0.5, 0.5]
        error = QuantumError(zip(noise_circs, noise_probs))
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    def test_backend_method_nonclifford_circuit_and_pauli_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=True)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    def test_backend_method_nonclifford_circuit_and_unitary_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = pauli_error([['XX', 0.5], ['II', 0.5]], standard_gates=False)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')

    def test_backend_method_nonclifford_circuit_and_kraus_noise(self):
        """Test statevector method is used for Clifford circuit"""
        # Noise Model
        error = amplitude_damping_error(0.5)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h', 's', 'sdg'])

        # Test circuits
        shots = 100
        circuits = ref_non_clifford.ccx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)

        def get_result():
            return self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS,
                                      noise_model=noise_model).result()

        # Check simulation method
        method = self.BACKEND_OPTS.get('method')
        if method == 'stabilizer':
            self.assertRaises(AerError, get_result)
        else:
            result = get_result()
            self.is_completed(result)
            self.compare_result_metadata(result, circuits, 'method', 'statevector')
