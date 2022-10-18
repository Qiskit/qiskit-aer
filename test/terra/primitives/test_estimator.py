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
Estimator class tests
"""


import unittest
from test.terra.common import QiskitAerTestCase
from test.terra.decorators import deprecated

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import EstimatorResult
from qiskit.quantum_info import Operator, SparsePauliOp

from qiskit_aer.primitives import Estimator


class TestEstimator(QiskitAerTestCase):
    """Testing estimator class"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.014918456829035, -1.2922526095793785

    @deprecated
    def test_estimator(self):
        """(Deprecated) test for a simple use case"""
        lst = [("XX", 1), ("YY", 2), ("ZZ", 3)]
        with self.subTest("PauliSumOp"):
            observable = PauliSumOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with Estimator([ansatz], [observable]) as est:
                result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15)
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.728515625])

        with self.subTest("SparsePauliOp"):
            observable = SparsePauliOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            with Estimator([ansatz], [observable]) as est:
                result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15)
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.728515625])

    @deprecated
    def test_estimator_param_reverse(self):
        """(Deprecated) test for the reverse parameter"""
        observable = PauliSumOp.from_list([("XX", 1), ("YY", 2), ("ZZ", 3)])
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        with Estimator([ansatz], [observable], [ansatz.parameters[::-1]]) as est:
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5][::-1]], seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.728515625])

    @deprecated
    def test_init_observable_from_operator(self):
        """(Deprecated) test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        matrix = Operator(
            [
                [-1.06365335, 0.0, 0.0, 0.1809312],
                [0.0, -1.83696799, 0.1809312, 0.0],
                [0.0, 0.1809312, -0.24521829, 0.0],
                [0.1809312, 0.0, 0.0, -1.06365335],
            ]
        )
        with Estimator([circuit], [matrix]) as est:
            result = est([0], [0], seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284607781171875])

    @deprecated
    def test_evaluate(self):
        """(Deprecated) test for evaluate"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])

    @deprecated
    def test_evaluate_multi_params(self):
        """(Deprecated) test for evaluate with multiple parameters"""
        with Estimator([self.ansatz], [self.observable]) as est:
            result = est(
                [0] * 2,
                [0] * 2,
                parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
                seed=15,
            )
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598, -1.3237023178807785])

    @deprecated
    def test_evaluate_no_params(self):
        """(Deprecated) test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with Estimator([circuit], [self.observable]) as est:
            result = est([0], [0], seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])

    @deprecated
    def test_run_with_multiple_observables_and_none_parameters(self):
        """(Deprecated) test for evaluate without parameters"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with Estimator(circuit, ["ZZZ", "III"]) as est:
            result = est(circuits=[0, 0], observables=[0, 1], seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [0.00390625, 1.0])

    @deprecated
    def test_estimator_example(self):
        """(Deprecated) test for Estimator example"""
        psi1 = RealAmplitudes(num_qubits=2, reps=2)
        psi2 = RealAmplitudes(num_qubits=2, reps=3)

        params1 = psi1.parameters
        params2 = psi2.parameters

        op1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        op2 = SparsePauliOp.from_list([("IZ", 1)])
        op3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

        with Estimator([psi1, psi2], [op1, op2, op3], [params1, params2]) as est:
            theta1 = [0, 1, 1, 2, 3, 5]
            theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
            theta3 = [1, 2, 3, 4, 5, 6]

            with self.subTest("test circuit 0, observable 0"):
                # calculate [ <psi1(theta1)|op1|psi1(theta1)> ]
                result = est([0], [0], [theta1], seed=15)
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1.57421875])
                self.assertEqual(len(result.metadata), 1)

            with self.subTest("test circuit [0, 0], observable [1, 2]"):
                # calculate [ <psi1(theta1)|op2|psi1(theta1)>, <psi1(theta1)|op3|psi1(theta1)> ]
                result = est([0, 0], [1, 2], [theta1] * 2, seed=15)
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [-0.5234375, 0.037109375])
                self.assertEqual(len(result.metadata), 2)

            with self.subTest("test circuit 1, observable 1"):
                # calculate [ <psi2(theta2)|op2|psi2(theta2)> ]
                result = est([1], [1], [theta2], seed=15)
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [0.232421875])
                self.assertEqual(len(result.metadata), 1)

            with self.subTest("test circuit [0, 0], observable [0, 0]"):
                # calculate [ <psi1(theta1)|op1|psi1(theta1)>, <psi1(theta3)|op1|psi1(theta3)> ]
                result = est([0, 0], [0, 0], [theta1, theta3], seed=15)
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1.57421875, 1.0859375])
                self.assertEqual(len(result.metadata), 2)

            with self.subTest("test circuit [0, 1, 0], observable [0, 1, 2]"):
                # calculate [ <psi1(theta1)|op1|psi1(theta1)>,
                #             <psi2(theta2)|op2|psi2(theta2)>,
                #             <psi1(theta3)|op3|psi1(theta3)> ]
                result = est([0, 1, 0], [0, 1, 2], [theta1, theta2, theta3], seed=15)
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1.57421875, 0.138671875, -1.078125])
                self.assertEqual(len(result.metadata), 3)

            with self.subTest("test circuit psi2, observable op2"):
                # It is possible to pass objects.
                # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
                result = est([psi2], [op2], [theta2], seed=15)
                np.testing.assert_allclose(result.values, [0.232421875])
                self.assertEqual(len(result.metadata), 1)

    @deprecated
    def test_1qubit(self):
        """(Deprecated) Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        with Estimator([qc, qc2], [op, op2], [[]] * 2) as est:
            with self.subTest("test circuit 0, observable 0"):
                result = est([0], [0], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 0, observable 1"):
                result = est([0], [1], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 1, observable 0"):
                result = est([1], [0], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 1, observable 1"):
                result = est([1], [1], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [-1])

    @deprecated
    def test_2qubits(self):
        """(Deprecated) Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        with Estimator([qc, qc2], [op, op2, op3], [[]] * 2) as est:
            with self.subTest("test circuit 0, observable 0"):
                result = est([0], [0], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 1, observable 0"):
                result = est([1], [0], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 0, observable 1"):
                result = est([0], [1], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 1, observable 1"):
                result = est([1], [1], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 0, observable 2"):
                result = est([0], [2], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [1])

            with self.subTest("test circuit 1, observable 2"):
                result = est([1], [2], [[]])
                self.assertIsInstance(result, EstimatorResult)
                np.testing.assert_allclose(result.values, [-1])

    @deprecated
    def test_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        with Estimator([qc, qc2], [op, op2], [[]] * 2) as est:
            with self.assertRaises(ValueError):
                est([0], [1], [[]])
            with self.assertRaises(ValueError):
                est([1], [0], [[]])
            with self.assertRaises(ValueError):
                est([0], [0], [[1e4]])
            with self.assertRaises(ValueError):
                est([1], [1], [[1, 2]])
            with self.assertRaises(ValueError):
                est([0, 1], [1], [[1]])
            with self.assertRaises(ValueError):
                est([0], [0, 1], [[1]])

    @deprecated
    def test_empty_parameter(self):
        """(Deprecated) Test for empty parameter"""
        n = 2
        qc = QuantumCircuit(n)
        op = SparsePauliOp.from_list([("I" * n, 1)])
        with Estimator(circuits=[qc] * 10, observables=[op] * 10) as estimator:
            with self.subTest("one circuit"):
                result = estimator([0], [1], shots=1000)
                np.testing.assert_allclose(result.values, [1])
                self.assertEqual(len(result.metadata), 1)

            with self.subTest("two circuits"):
                result = estimator([2, 4], [3, 5], shots=1000)
                np.testing.assert_allclose(result.values, [1, 1])
                self.assertEqual(len(result.metadata), 2)

    @deprecated
    def test_numpy_params(self):
        """(Deprecated) Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        with Estimator(circuits=qc, observables=op) as estimator:
            target = estimator([0] * k, [0] * k, params_list, seed=15)

            with self.subTest("ndarrary"):
                result = estimator([0] * k, [0] * k, params_array, seed=15)
                self.assertEqual(len(result.metadata), k)
                np.testing.assert_allclose(result.values, target.values)

            with self.subTest("list of ndarray"):
                result = estimator([0] * k, [0] * k, params_list_array, seed=15)
                self.assertEqual(len(result.metadata), k)
                np.testing.assert_allclose(result.values, target.values)

    @deprecated
    def test_passing_objects(self):
        """(Deprecated) Test passsing object for Estimator."""

        with self.subTest("Valid test"):
            with Estimator([self.ansatz], [self.observable]) as estimator:
                result = estimator(
                    circuits=[self.ansatz, self.ansatz],
                    observables=[self.observable, self.observable],
                    parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                    seed=15,
                )
            self.assertAlmostEqual(result.values[0], self.expvals[0])
            self.assertAlmostEqual(result.values[1], self.expvals[1])

        with self.subTest("Invalid circuit test"):
            circuit = QuantumCircuit(2)
            with Estimator([self.ansatz], [self.observable]) as estimator:
                with self.assertRaises(ValueError):
                    estimator(
                        circuits=[self.ansatz, circuit],
                        observables=[self.observable, self.observable],
                        parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                    )

        with self.subTest("Invalid observable test"):
            observable = SparsePauliOp(["ZX"])
            with Estimator([self.ansatz], [self.observable]) as estimator:
                with self.assertRaises(ValueError):
                    estimator(
                        circuits=[self.ansatz, self.ansatz],
                        observables=[observable, self.observable],
                        parameter_values=[list(range(6)), [0, 1, 1, 2, 3, 5]],
                    )

    @deprecated
    def test_with_shots_option_with_approximation(self):
        """(Deprecated) test with shots option."""
        with Estimator([self.ansatz], [self.observable], approximation=True) as est:
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.3088991960117797])

    @deprecated
    def test_with_shots_option_without_approximation(self):
        """(Deprecated) test with shots option."""
        with Estimator([self.ansatz], [self.observable], approximation=False) as est:
            result = est([0], [0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed=15)
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])

    def test_estimator(self):
        """test for a simple use case"""
        lst = [("XX", 1), ("YY", 2), ("ZZ", 3)]
        with self.subTest("PauliSumOp"):
            observable = PauliSumOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            est = Estimator()
            result = est.run(
                ansatz, observable, parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15
            ).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.728515625])

        with self.subTest("SparsePauliOp"):
            observable = SparsePauliOp.from_list(lst)
            ansatz = RealAmplitudes(num_qubits=2, reps=2)
            est = Estimator()
            result = est.run(
                ansatz, observable, parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15
            ).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1.728515625])

    def test_init_observable_from_operator(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        matrix = Operator(
            [
                [-1.06365335, 0.0, 0.0, 0.1809312],
                [0.0, -1.83696799, 0.1809312, 0.0],
                [0.0, 0.1809312, -0.24521829, 0.0],
                [0.1809312, 0.0, 0.0, -1.06365335],
            ]
        )
        est = Estimator()
        result = est.run([circuit], [matrix], seed=15).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284607781171875])

    def test_evaluate(self):
        """test for evaluate"""
        est = Estimator()
        result = est.run(
            self.ansatz, self.observable, parameter_values=[[0, 1, 1, 2, 3, 5]], seed=15
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])

    def test_evaluate_multi_params(self):
        """test for evaluate with multiple parameters"""
        est = Estimator()
        result = est.run(
            [self.ansatz] * 2,
            [self.observable] * 2,
            parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
            seed=15,
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598, -1.3237023178807785])

    def test_evaluate_no_params(self):
        """test for evaluate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        est = Estimator()
        result = est.run(circuit, self.observable, seed=15).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])

    def test_run_with_multiple_observables_and_none_parameters(self):
        """test for evaluate without parameters"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        est = Estimator()
        result = est.run(
            [circuit] * 2, [SparsePauliOp("ZZZ"), SparsePauliOp("III")], seed=15
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [0.00390625, 1.0])

    def test_1qubit(self):
        """Test for 1-qubit cases"""
        qc0 = QuantumCircuit(1)
        qc1 = QuantumCircuit(1)
        qc1.x(0)

        op0 = SparsePauliOp.from_list([("I", 1)])
        op1 = SparsePauliOp.from_list([("Z", 1)])

        est = Estimator()
        with self.subTest("test circuit 0, observable 0"):
            result = est.run(qc0, op0).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 0, observable 1"):
            result = est.run(qc0, op1).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 1, observable 0"):
            result = est.run(qc1, op0).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 1, observable 1"):
            result = est.run(qc1, op1).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1])

    def test_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc0 = QuantumCircuit(2)
        qc1 = QuantumCircuit(2)
        qc1.x(0)

        op0 = SparsePauliOp.from_list([("II", 1)])
        op1 = SparsePauliOp.from_list([("ZI", 1)])
        op2 = SparsePauliOp.from_list([("IZ", 1)])

        est = Estimator()
        with self.subTest("test circuit 0, observable 0"):
            result = est.run(qc0, op0).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 1, observable 0"):
            result = est.run(qc1, op0).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 0, observable 1"):
            result = est.run(qc0, op1).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 1, observable 1"):
            result = est.run(qc1, op1).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 0, observable 2"):
            result = est.run(qc0, op2).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [1])

        with self.subTest("test circuit 1, observable 2"):
            result = est.run(qc1, op2).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 2
        qc = QuantumCircuit(n)
        op = SparsePauliOp.from_list([("I" * n, 1)])
        estimator = Estimator()
        with self.subTest("one circuit"):
            result = estimator.run(qc, op, shots=1000).result()
            np.testing.assert_allclose(result.values, [1])
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            result = estimator.run([qc] * 2, [op] * 2, shots=1000).result()
            np.testing.assert_allclose(result.values, [1, 1])
            self.assertEqual(len(result.metadata), 2)

    def test_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = Estimator()
        target = estimator.run([qc] * k, [op] * k, params_list, seed=15).result()

        with self.subTest("ndarrary"):
            result = estimator.run([qc] * k, [op] * k, params_array, seed=15).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

        with self.subTest("list of ndarray"):
            result = estimator.run([qc] * k, [op] * k, params_list_array, seed=15).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values)

    def test_with_shots_option_with_approximation(self):
        """test with shots option."""
        est = Estimator(approximation=True)
        result = est.run(
            self.ansatz, self.observable, parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed=15
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.3088991960117797])

    def test_with_shots_option_without_approximation(self):
        """test with shots option."""
        est = Estimator(approximation=False)
        result = est.run(
            self.ansatz, self.observable, parameter_values=[[0, 1, 1, 2, 3, 5]], shots=1024, seed=15
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.2895828299114598])


if __name__ == "__main__":
    unittest.main()
