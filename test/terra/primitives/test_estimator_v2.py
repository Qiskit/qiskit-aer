# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Estimator V2."""

from __future__ import annotations

import unittest
from test.terra.common import QiskitAerTestCase

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2


class TestEstimatorV2(QiskitAerTestCase):
    """Test Estimator V2"""

    def setUp(self):
        super().setUp()
        self._precision = 5e-3
        self._rtol = 3e-1
        self._seed = 15
        self._rng = np.random.default_rng(self._seed)
        self._options = {
            "run_options": {"seed_simulator": self._seed},
            "default_precision": self._precision,
        }
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
        self.expvals = -1.0284380963435145, -1.284366511861733

        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )
        self.backend = AerSimulator()

    def test_estimator_run(self):
        """Test Estimator.run()"""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        psi1, psi2 = pm.run([psi1, psi2])
        estimator = EstimatorV2(options=self._options)
        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        ham1 = hamiltonian1.apply_layout(psi1.layout)
        job = estimator.run([(psi1, ham1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956], rtol=self._rtol)
        self.assertIn("simulator_metadata", result[0].metadata)

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H1|psi2(theta2)> ]
        ham1 = hamiltonian1.apply_layout(psi2.layout)
        result2 = estimator.run([(psi2, ham1, theta2)]).result()
        np.testing.assert_allclose(result2[0].data.evs, [2.97797666], rtol=self._rtol)
        self.assertIn("simulator_metadata", result2[0].metadata)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        ham2 = hamiltonian2.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        result3 = estimator.run([(psi1, [ham2, ham3], theta1)]).result()
        np.testing.assert_allclose(result3[0].data.evs, [-0.551653, 0.07535239], rtol=self._rtol)
        self.assertIn("simulator_metadata", result3[0].metadata)

        # calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
        #              <psi1(theta3)|H3|psi1(theta3)>],
        #             [<psi2(theta2)|H2|psi2(theta2)>] ]
        ham1 = hamiltonian1.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        ham2 = hamiltonian2.apply_layout(psi2.layout)
        result4 = estimator.run(
            [
                (psi1, [ham1, ham3], [theta1, theta3]),
                (psi2, ham2, theta2),
            ]
        ).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318], rtol=self._rtol)
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238], rtol=self._rtol)
        self.assertIn("simulator_metadata", result4[0].metadata)
        self.assertIn("simulator_metadata", result4[1].metadata)

    def test_estimator_with_pub(self):
        """Test estimator with explicit EstimatorPubs."""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        psi1, psi2 = pm.run([psi1, psi2])

        ham1 = hamiltonian1.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        obs1 = ObservablesArray.coerce([ham1, ham3])
        bind1 = BindingsArray.coerce({tuple(psi1.parameters): [theta1, theta3]})
        pub1 = EstimatorPub(psi1, obs1, bind1)

        ham2 = hamiltonian2.apply_layout(psi2.layout)
        obs2 = ObservablesArray.coerce(ham2)
        bind2 = BindingsArray.coerce({tuple(psi2.parameters): theta2})
        pub2 = EstimatorPub(psi2, obs2, bind2)

        estimator = EstimatorV2(options=self._options)
        result4 = estimator.run([pub1, pub2]).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318], rtol=self._rtol)
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238], rtol=self._rtol)

    def test_estimator_run_no_params(self):
        """test for estimator without parameters"""
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        circuit = pm.run(circuit)
        est = EstimatorV2(options=self._options)
        observable = self.observable.apply_layout(circuit.layout)
        result = est.run([(circuit, observable)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733], rtol=self._rtol)

    def test_run_single_circuit_observable(self):
        """Test for single circuit and single observable case."""
        est = EstimatorV2(options=self._options)
        pm = generate_preset_pass_manager(optimization_level=0, target=self.backend.target)

        with self.subTest("No parameter"):
            qc = QuantumCircuit(1)
            qc.x(0)
            qc = pm.run(qc)
            op = SparsePauliOp("Z")
            op = op.apply_layout(qc.layout)
            param_vals = [None, [], [[]], np.array([]), np.array([[]]), [np.array([])]]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

        with self.subTest("One parameter"):
            param = Parameter("x")
            qc = QuantumCircuit(1)
            qc.ry(param, 0)
            qc = pm.run(qc)
            op = SparsePauliOp("Z")
            op = op.apply_layout(qc.layout)
            param_vals = [
                [np.pi],
                np.array([np.pi]),
            ]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

        with self.subTest("More than one parameter"):
            qc = self.psi[0]
            qc = pm.run(qc)
            op = self.hamiltonian[0]
            op = op.apply_layout(qc.layout)
            param_vals = [
                self.theta[0],
                [self.theta[0]],
                np.array(self.theta[0]),
                np.array([self.theta[0]]),
                [np.array(self.theta[0])],
            ]
            target = [1.5555572817900956]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

    def test_run_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        qc, qc2 = pm.run([qc, qc2])

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = EstimatorV2(options=self._options)
        op_1 = op.apply_layout(qc.layout)
        result = est.run([(qc, op_1)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_2 = op2.apply_layout(qc.layout)
        result = est.run([(qc, op_2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_3 = op.apply_layout(qc2.layout)
        result = est.run([(qc2, op_3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_4 = op2.apply_layout(qc2.layout)
        result = est.run([(qc2, op_4)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1], rtol=self._rtol)

    def test_run_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        qc, qc2 = pm.run([qc, qc2])

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = EstimatorV2(options=self._options)
        op_1 = op.apply_layout(qc.layout)
        result = est.run([(qc, op_1)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_2 = op.apply_layout(qc2.layout)
        result = est.run([(qc2, op_2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_3 = op2.apply_layout(qc.layout)
        result = est.run([(qc, op_3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_4 = op2.apply_layout(qc2.layout)
        result = est.run([(qc2, op_4)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_5 = op3.apply_layout(qc.layout)
        result = est.run([(qc, op_5)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_6 = op3.apply_layout(qc2.layout)
        result = est.run([(qc2, op_6)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1], rtol=self._rtol)

    def test_run_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = EstimatorV2(options=self._options)
        with self.assertRaises(ValueError):
            est.run([(qc, op2)]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op, [[1e4]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc2, op2, [[1, 2]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, [op, op2], [[1]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op)], precision=-1).result()
        with self.assertRaises(ValueError):
            est.run([(qc, 1j * op)], precision=0.1).result()
        # precision < 0
        with self.assertRaises(ValueError):
            est.run([(qc, op, None, -1)]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op)], precision=-1).result()
        with self.subTest("missing []"):
            with self.assertRaisesRegex(ValueError, "An invalid Estimator pub-like was given"):
                _ = est.run((qc, op)).result()

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        qc = pm.run(qc)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        op = op.apply_layout(qc.layout)
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        statevector_estimator = StatevectorEstimator(seed=123)
        target = statevector_estimator.run([(qc, op, params_list)]).result()

        estimator = EstimatorV2(options=self._options)

        with self.subTest("ndarrary"):
            result = estimator.run([(qc, op, params_array)]).result()
            self.assertEqual(result[0].data.evs.shape, (k,))
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs, rtol=self._rtol)

        with self.subTest("list of ndarray"):
            result = estimator.run([(qc, op, params_list_array)]).result()
            self.assertEqual(result[0].data.evs.shape, (k,))
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs, rtol=self._rtol)

    def test_precision(self):
        """Test for precision"""
        estimator = EstimatorV2(options=self._options)
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        psi1 = pm.run(self.psi[0])
        hamiltonian1 = self.hamiltonian[0].apply_layout(psi1.layout)
        theta1 = self.theta[0]
        job = estimator.run([(psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        # The result of the second run is the same
        job = estimator.run([(psi1, hamiltonian1, [theta1]), (psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [1.901141473854881], rtol=self._rtol)
        # apply smaller precision value
        job = estimator.run([(psi1, hamiltonian1, [theta1])], precision=self._precision * 0.5)
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956], rtol=self._rtol)

    def test_diff_precision(self):
        """Test for running different precisions at once"""
        estimator = EstimatorV2(options=self._options)
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        psi1 = pm.run(self.psi[0])
        hamiltonian1 = self.hamiltonian[0].apply_layout(psi1.layout)
        theta1 = self.theta[0]
        job = estimator.run(
            [(psi1, hamiltonian1, [theta1]), (psi1, hamiltonian1, [theta1], self._precision * 0.8)]
        )
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [1.901141473854881], rtol=self._rtol)

    def test_iter_pub(self):
        """test for an iterable of pubs"""
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)
        circuit = pm.run(circuit)
        estimator = EstimatorV2(options=self._options)
        observable = self.observable.apply_layout(circuit.layout)
        result = estimator.run(iter([(circuit, observable), (circuit, observable)])).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [-1.284366511861733], rtol=self._rtol)


if __name__ == "__main__":
    unittest.main()
