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
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult

from qiskit import QuantumCircuit
from qiskit.providers.aer.primitives import Sampler


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
        with Sampler(circuits=circuits) as sampler:
            result = sampler(
                list(range(len(indices))), parameter_values=[[] for _ in indices], seed=15
            )
            self._compare_probs(result.quasi_dists, target)

    @data([0], [1], [0, 1])
    def test_sampler_pqc(self, indices):
        """test for sampler with a parametrized circuit"""
        params, target = self._generate_params_target(indices)
        with Sampler(circuits=self._pqc) as sampler:
            result = sampler([0] * len(params), params, seed=15)
            self._compare_probs(result.quasi_dists, target)

    @data([0, 0], [0, 1], [1, 1])
    def test_evaluate_two_pqcs(self, indices):
        """test for sampler with two parametrized circuits"""
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with Sampler(circuits=circs) as sampler:
            result = sampler(indices, parameter_values=params, seed=15)
            self._compare_probs(result.quasi_dists, target)

    def test_sampler_example(self):
        """test for Sampler example"""

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        # executes a Bell circuit
        with Sampler(circuits=[bell], parameters=[[]]) as sampler:
            result = sampler(parameter_values=[[]], circuits=[0], seed=15)
            self.assertIsInstance(result, SamplerResult)
            self.assertEqual(len(result.quasi_dists), 1)
            self.assertDictAlmostEqual(result.quasi_dists[0], {0: 0.5244140625, 3: 0.4755859375})
            self.assertEqual(len(result.metadata), 1)

        # executes three Bell circuits
        with Sampler([bell] * 3, [[]] * 3) as sampler:
            result = sampler([0, 1, 2], [[]] * 3, seed=15)
            self.assertIsInstance(result, SamplerResult)
            self.assertEqual(len(result.quasi_dists), 3)
            self.assertEqual(len(result.metadata), 3)
            self.assertDictAlmostEqual(result.quasi_dists[0], {0: 0.5244140625, 3: 0.4755859375})
            self.assertDictAlmostEqual(result.quasi_dists[1], {0: 0.5126953125, 3: 0.4873046875})
            self.assertDictAlmostEqual(result.quasi_dists[2], {0: 0.5048828125, 3: 0.4951171875})

        with Sampler([bell]) as sampler:
            result = sampler([bell, bell, bell], seed=15)
            self.assertIsInstance(result, SamplerResult)
            self.assertEqual(len(result.quasi_dists), 3)
            self.assertEqual(len(result.metadata), 3)
            self.assertDictAlmostEqual(result.quasi_dists[0], {0: 0.5244140625, 3: 0.4755859375})
            self.assertDictAlmostEqual(result.quasi_dists[1], {0: 0.5126953125, 3: 0.4873046875})
            self.assertDictAlmostEqual(result.quasi_dists[2], {0: 0.5048828125, 3: 0.4951171875})

        # parametrized circuit
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [1, 2, 3, 4, 5, 6]
        theta3 = [0, 1, 2, 3, 4, 5, 6, 7]

        with Sampler(circuits=[pqc, pqc2], parameters=[pqc.parameters, pqc2.parameters]) as sampler:
            result = sampler([0, 0, 1], [theta1, theta2, theta3], seed=15)

        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)
        self.assertDictAlmostEqual(
            result.quasi_dists[0], {0: 0.14453125, 1: 0.3583984375, 2: 0.1083984375, 3: 0.388671875}
        )
        self.assertDictAlmostEqual(
            result.quasi_dists[1],
            {0: 0.0478515625, 1: 0.033203125, 2: 0.6162109375, 3: 0.302734375},
        )
        self.assertDictAlmostEqual(
            result.quasi_dists[2], {0: 0.1787109375, 1: 0.7060546875, 2: 0.08984375, 3: 0.025390625}
        )

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

        with Sampler([qc, qc], [[x, y], [y, x]], backend_options={"seed_simulator": 15}) as sampler:
            result = sampler([0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]])
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

        with Sampler([qc, qc], [[x, y], [y, x]]) as sampler:
            result = sampler(
                [0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]], seed=15
            )
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

        with Sampler([qc, qc2], [qc.parameters, qc2.parameters]) as sampler:
            result = sampler([0, 1], [[]] * 2)
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

        with Sampler(
            [qc0, qc1, qc2, qc3], [qc0.parameters, qc1.parameters, qc2.parameters, qc3.parameters]
        ) as sampler:
            result = sampler([0, 1, 2, 3], [[]] * 4)
            self.assertIsInstance(result, SamplerResult)
            self.assertEqual(len(result.quasi_dists), 4)

            self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1})
            self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1})
            self.assertDictAlmostEqual(result.quasi_dists[2], {2: 1})
            self.assertDictAlmostEqual(result.quasi_dists[3], {3: 1})

    def test_errors(self):
        """Test for errors"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        with Sampler([qc1, qc2], [qc1.parameters, qc2.parameters]) as sampler:
            with self.assertRaises(QiskitError):
                sampler([0], [[1e2]])
            with self.assertRaises(QiskitError):
                sampler([1], [[]])
            with self.assertRaises(QiskitError):
                sampler([1], [[1e2]])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        with Sampler(circuits=[qc] * 10) as sampler:
            with self.subTest("one circuit"):
                result = sampler([0], shots=1000)
                self.assertEqual(len(result.quasi_dists), 1)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})
                self.assertEqual(len(result.metadata), 1)

            with self.subTest("two circuits"):
                result = sampler([2, 4], shots=1000)
                self.assertEqual(len(result.quasi_dists), 2)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})
                self.assertEqual(len(result.metadata), 2)

    def test_passing_objects(self):
        """Test passing objects for Sampler."""

        params, target = self._generate_params_target([0])

        with self.subTest("Valid test"):
            with Sampler(circuits=self._pqc) as sampler:
                result = sampler(circuits=[self._pqc], parameter_values=params, seed=15)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("Invalid circuit test"):
            circuit = QuantumCircuit(2)
            with Sampler(circuits=self._pqc) as sampler:
                with self.assertRaises(QiskitError):
                    sampler(circuits=[circuit], parameter_values=params)

    def test_with_shots_option(self):
        """test with shots option."""
        params, target = self._generate_params_target([1])
        with Sampler(circuits=self._pqc) as sampler:
            result = sampler(circuits=[0], parameter_values=params, shots=1024, seed=15)
            self._compare_probs(result.quasi_dists, target)


if __name__ == "__main__":
    unittest.main()
