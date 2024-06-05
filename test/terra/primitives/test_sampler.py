# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sampler class tests
"""

import unittest
from test.terra.common import QiskitAerTestCase

import numpy as np
from ddt import data, ddt
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult

from qiskit_aer.primitives import Sampler

from test.terra.backends.simulator_test_case import supported_methods


@ddt
class TestSampler(QiskitAerTestCase):
    """Testing sampler class"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1)
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        self._circuit = [hadamard, bell]
        self._target = [
            {0: 0.5, 1: 0.5},
            {0: 0.5, 3: 0.5, 1: 0, 2: 0},
        ]
        self._pqc = RealAmplitudes(num_qubits=2, reps=2)
        self._pqc.measure_all()
        self._pqc_params = [[0.0] * 6, [1.0] * 6]
        self._pqc_target = [{0: 1}, {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}]

    def _generate_circuits_target(self, indices):
        if isinstance(indices, list):
            circuits = [self._circuit[j] for j in indices]
            target = [self._target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return circuits, target

    def _generate_params_target(self, indices):
        if isinstance(indices, int):
            params = self._pqc_params[indices]
            target = self._pqc_target[indices]
        elif isinstance(indices, list):
            params = [self._pqc_params[j] for j in indices]
            target = [self._pqc_target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return params, target

    def _compare_probs(self, prob, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(prob), len(target))
        for p, targ in zip(prob, target):
            for key, t_val in targ.items():
                if key in p:
                    self.assertAlmostEqual(p[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @data([0], [1], [0, 1])
    def test_sampler(self, indices):
        """test for sampler"""
        circuits, target = self._generate_circuits_target(indices)
        sampler = Sampler()
        result = sampler.run(circuits, seed=15).result()
        self._compare_probs(result.quasi_dists, target)

    @data([0], [1], [0, 1], [0, 0], [1, 1])
    def test_sampler_pqc(self, indices):
        """test for sampler with a parametrized circuit"""
        params, target = self._generate_params_target(indices)
        sampler = Sampler()
        result = sampler.run([self._pqc] * len(params), params, seed=15).result()
        self._compare_probs(result.quasi_dists, target)

    def test_sampler_param_order(self):
        """test for sampler with different parameter orders"""
        x = Parameter("x")
        y = Parameter("y")

        qc = QuantumCircuit(3, 3)
        qc.rx(x, 0)
        qc.rx(y, 1)
        qc.x(2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        sampler = Sampler(backend_options={"method": "statevector", "seed_simulator": 15})
        result = sampler.run([qc] * 4, [[0, 0], [0, 0], [np.pi / 2, 0], [0, np.pi / 2]]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[0], {4: 1})

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[1], {4: 1})

        # qc({x: pi/2, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[2], {4: 0.4990234375, 5: 0.5009765625})

        # qc({x: 0, y: pi/2})
        self.assertDictAlmostEqual(result.quasi_dists[3], {4: 0.4814453125, 6: 0.5185546875})

    def test_sampler_reverse_meas_order(self):
        """test for sampler with reverse measurement order"""
        x = Parameter("x")
        y = Parameter("y")

        qc = QuantumCircuit(3, 3)
        qc.rx(x, 0)
        qc.rx(y, 1)
        qc.x(2)
        qc.measure(0, 2)
        qc.measure(1, 1)
        qc.measure(2, 0)

        sampler = Sampler(backend_options={"method": "statevector"})
        result = sampler.run(
            [qc, qc, qc, qc], [[0, 0], [0, 0], [np.pi / 2, 0], [0, np.pi / 2]], seed=15
        ).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[0], {1: 1})

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1})

        # qc({x: pi/2, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[2], {1: 0.4990234375, 5: 0.5009765625})

        # qc({x: 0, y: pi/2})
        self.assertDictAlmostEqual(result.quasi_dists[3], {1: 0.4814453125, 3: 0.5185546875})

    def test_1qubit(self):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        sampler = Sampler()
        result = sampler.run([qc, qc2]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)
        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1})
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1})

    def test_2qubit(self):
        """test for 2-qubit cases"""
        qc0 = QuantumCircuit(2)
        qc0.measure_all()
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.measure_all()
        qc3 = QuantumCircuit(2)
        qc3.x([0, 1])
        qc3.measure_all()

        sampler = Sampler()
        result = sampler.run([qc0, qc1, qc2, qc3]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1})
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1})
        self.assertDictAlmostEqual(result.quasi_dists[2], {2: 1})
        self.assertDictAlmostEqual(result.quasi_dists[3], {3: 1})

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        sampler = Sampler()
        with self.subTest("one circuit"):
            result = sampler.run([qc], shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 1)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            result = sampler.run([qc] * 2, shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 2)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 2)

    def test_with_shots_option(self):
        """test with shots option."""
        params, target = self._generate_params_target([1])
        sampler = Sampler()
        result = sampler.run(
            circuits=[self._pqc], parameter_values=params, shots=1024, seed=15
        ).result()
        self._compare_probs(result.quasi_dists, target)

    def test_with_shots_none(self):
        """test with shots None."""
        sampler = Sampler()
        result = sampler.run(
            circuits=[self._pqc], parameter_values=[self._pqc_params[1]], shots=None
        ).result()
        self.assertDictAlmostEqual(
            result.quasi_dists[0],
            {
                0: 0.01669499556655749,
                1: 0.3363966103502914,
                2: 0.04992359174946462,
                3: 0.596984802333687,
            },
        )

    @data(8192, None)
    def test_num_clbits(self, shots):
        """test of QuasiDistribution"""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.measure_all()

        result = Sampler().run(qc, shots=shots, seed=20).result()
        quasis = result.quasi_dists[0]
        bin_probs = quasis.binary_probabilities()
        self.assertDictAlmostEqual(bin_probs, {"0000": 0.5, "0001": 0.5}, delta=1e-2)

    def test_multiple_cregs(self):
        """Test for the circuit with multipe cregs"""
        qc = QuantumCircuit(2)
        cr1 = ClassicalRegister(1, "cr1")
        cr2 = ClassicalRegister(1, "cr2")
        qc.add_register(cr1)
        qc.add_register(cr2)
        qc.measure(0, 0)
        qc.measure(1, 1)

        result = Sampler().run(qc, shots=100).result()
        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1})

    @supported_methods(
        [
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "stabilizer",
            "extended_stabilizer",
            "tensor_network",
        ]
    )
    def test_truncate_large_circuit(self, method, device):
        """Test trancate large circuit in transplier"""
        options = {"method": method, "device": device}
        sampler = Sampler(backend_options=options)
        qc = QuantumCircuit(100, 2)
        qc.h(98)
        qc.cx(98, 99)
        qc.measure([98, 99], [0, 1])
        result = sampler.run(qc).result()
        self.assertIsInstance(result, SamplerResult)


if __name__ == "__main__":
    unittest.main()
