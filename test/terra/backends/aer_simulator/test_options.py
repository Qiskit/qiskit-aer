# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimualtor options tests
"""

from ddt import ddt, data
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer.noise import NoiseModel
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import state_fidelity

@ddt
class TestOptions(SimulatorTestCase):
    """Tests of AerSimulator options"""

    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer'])
    def test_seed_simulator_option_measure(self, method, device):
        """Test seed_simulator option fixes measurement outcomes"""
        backend = self.backend(method=method, device=device,
                               seed_simulator=111)
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.reset(0)
        qc.measure_all()
        qc = transpile(qc, backend)

        counts1 = backend.run(qc).result().get_counts(0)
        counts2 = backend.run(qc).result().get_counts(0)

        self.assertEqual(counts1, counts2)

    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer'])
    def test_seed_simulator_run_option_measure(self, method, device):
        """Test seed_simulator option fixes measurement outcomes"""
        backend = self.backend(method=method, device=device)
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.reset(0)
        qc.measure_all()
        qc = transpile(qc, backend)
        seed = 1234
        counts1 = backend.run(qc, seed_simulator=seed).result().get_counts(0)
        counts2 = backend.run(qc, seed_simulator=seed).result().get_counts(0)
        self.assertEqual(counts1, counts2)

    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop'])
    def test_method_option(self, method, device):
        """Test method option works"""
        backend = self.backend(method=method, device=device)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc = transpile(qc, backend)

        # Target simulation method
        if method == 'automatic':
            target = 'stabilizer'
        else:
            target = method

        result = backend.run(qc).result()
        value = result.results[0].metadata.get('method', None)
        self.assertEqual(value, target)

    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop'])
    def test_device_option(self, method, device):
        """Test device option works"""
        backend = self.backend(method=method, device=device)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc = transpile(qc, backend)

        result = backend.run(qc).result()
        value = result.results[0].metadata.get('device', None)
        self.assertEqual(value, device)

    @data('automatic', 'statevector', 'density_matrix', 'stabilizer',
          'matrix_product_state', 'extended_stabilizer')
    def test_option_basis_gates(self, method):
        """Test setting method and noise model has correct basis_gates"""
        config = self.backend(method=method).configuration()
        noise_gates = ['id', 'sx', 'x', 'cx']
        noise_model = NoiseModel(basis_gates=noise_gates)
        target_gates = (sorted(set(config.basis_gates).intersection(noise_gates))
                        + config.custom_instructions)

        sim = self.backend(method=method, noise_model=noise_model)
        basis_gates = sim.configuration().basis_gates
        self.assertEqual(sorted(basis_gates), sorted(target_gates))

    @data('automatic', 'statevector', 'density_matrix', 'stabilizer',
          'matrix_product_state', 'extended_stabilizer')
    def test_option_order_basis_gates(self, method):
        """Test order of setting method and noise model gives same basis gates"""
        noise_model = NoiseModel(basis_gates=['id', 'sx', 'x', 'cx'])
        sim1 = self.backend(method=method, noise_model=noise_model)
        basis_gates1 = sim1.configuration().basis_gates
        sim2 = self.backend(noise_model=noise_model, method=method)
        basis_gates2 = sim2.configuration().basis_gates
        self.assertEqual(sorted(basis_gates1), sorted(basis_gates2))
    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer'])
    def test_shots_option(self, method, device):
        """Test shots option is observed"""
        shots = 99
        backend = self.backend(method=method, device=device, shots=shots)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        qc = transpile(qc, backend)
        result = backend.run(qc).result()
        value = sum(result.get_counts().values())
        self.assertEqual(value, shots)

    @supported_methods(
        ['automatic', 'stabilizer', 'statevector', 'density_matrix',
         'matrix_product_state', 'extended_stabilizer'])
    def test_shots_run_option(self, method, device):
        """Test shots option is observed"""
        shots = 99
        backend = self.backend(method=method, device=device)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        qc = transpile(qc, backend)
        result = backend.run(qc, shots=shots).result()
        value = sum(result.get_counts().values())
        self.assertEqual(value, shots)

    def test_mps_approximation(self):
        """Test MPS approximation"""
        shots = 4000
        method="matrix_product_state"
        backend_exact = self.backend(method=method)
        backend_approx = self.backend(method=method,
                                     matrix_product_state_max_bond_dimension=8)
        # The test must be large enough and entangled enough so that
        # approximation actually does something
        n = 10
        circuit = QuantumCircuit(n)
        for times in range(2):
            for i in range(0, n, 2):
                circuit.unitary(random_unitary(4), [i, i+1])
            for i in range(1, n-1):
                circuit.cx(0, i)
        circuit.save_statevector('sv')

        result_exact = backend_exact.run(circuit, shots=shots).result()
        sv_exact = result_exact.data(0)['sv']
        result_approx = backend_approx.run(circuit, shots=shots).result()
        sv_approx = result_approx.data(0)['sv']
        # Check that the fidelity is reasonable
        self.assertGreaterEqual(state_fidelity(sv_exact, sv_approx), 0.88)

        # Check that the approximated result is not identical to the exact
        # result, because that could mean there was actually no approximation
        self.assertLessEqual(state_fidelity(sv_exact, sv_approx), 0.99)
