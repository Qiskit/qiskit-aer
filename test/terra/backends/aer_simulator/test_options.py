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

from ddt import ddt
from qiskit import QuantumCircuit, transpile
from .aer_simulator_test_case import (
    AerSimulatorTestCase, supported_methods)


@ddt
class TestOptions(AerSimulatorTestCase):
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
    def test_method(self, method, device):
        """Test seed_simulator option fixes measurement outcomes"""
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
    def test_device(self, method, device):
        """Test seed_simulator option fixes measurement outcomes"""
        backend = self.backend(method=method, device=device)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc = transpile(qc, backend)

        result = backend.run(qc).result()
        value = result.results[0].metadata.get('device', None)
        self.assertEqual(value, device)
