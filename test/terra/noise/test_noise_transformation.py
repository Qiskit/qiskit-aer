# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
NoiseTransformer class tests
"""
from test.terra.common import QiskitAerTestCase

import unittest

import numpy

from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate, HGate, SGate
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.quantum_info.operators.channel import Kraus
from qiskit.quantum_info.random import random_unitary
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.quantum_error import QuantumError
from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit_aer.noise.errors.standard_errors import pauli_error
from qiskit_aer.noise.errors.standard_errors import reset_error
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer.utils import approximate_noise_model
from qiskit_aer.utils import approximate_quantum_error

try:
    import cvxpy

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@unittest.skipUnless(HAS_CVXPY, "cvxpy is required to run these tests")
class TestNoiseTransformer(QiskitAerTestCase):
    def setUp(self):
        super().setUp()
        self.ops = {"X": XGate(), "Y": YGate(), "Z": ZGate(), "H": HGate(), "S": SGate()}

    def assertNoiseModelsAlmostEqual(self, lhs, rhs, places=3):
        self.assertNoiseDictsAlmostEqual(
            lhs._local_quantum_errors, rhs._local_quantum_errors, places=places
        )
        self.assertNoiseDictsAlmostEqual(
            lhs._default_quantum_errors, rhs._default_quantum_errors, places=places
        )
        self.assertNoiseDictsAlmostEqual(
            lhs._local_readout_errors, rhs._local_readout_errors, places=places
        )
        if lhs._default_readout_error is not None:
            self.assertTrue(rhs._default_readout_error is not None)
            self.assertErrorsAlmostEqual(
                lhs._default_readout_error, rhs._default_readout_error, places=places
            )
        else:
            self.assertTrue(rhs._default_readout_error is None)

    def assertNoiseDictsAlmostEqual(self, lhs, rhs, places=3):
        keys = set(lhs.keys()).union(set(rhs.keys()))
        for key in keys:
            self.assertTrue(key in lhs.keys(), msg="Key {} is missing from lhs".format(key))
            self.assertTrue(key in rhs.keys(), msg="Key {} is missing from rhs".format(key))
            if isinstance(lhs[key], dict):
                self.assertNoiseDictsAlmostEqual(lhs[key], rhs[key], places=places)
            else:
                self.assertErrorsAlmostEqual(lhs[key], rhs[key], places=places)

    def assertErrorsAlmostEqual(self, lhs, rhs, places=3):
        self.assertMatricesAlmostEqual(
            lhs.to_quantumchannel()._data, rhs.to_quantumchannel()._data, places
        )

    def assertDictAlmostEqual(self, lhs, rhs, places=None):
        keys = set(lhs.keys()).union(set(rhs.keys()))
        for key in keys:
            self.assertAlmostEqual(
                lhs.get(key),
                rhs.get(key),
                msg="Not almost equal for key {}: {} !~ {}".format(key, lhs.get(key), rhs.get(key)),
                places=places,
            )

    def assertListAlmostEqual(self, lhs, rhs, places=None):
        self.assertEqual(
            len(lhs), len(rhs), msg="List lengths differ: {} != {}".format(len(lhs), len(rhs))
        )
        for i in range(len(lhs)):
            if isinstance(lhs[i], numpy.ndarray) and isinstance(rhs[i], numpy.ndarray):
                self.assertMatricesAlmostEqual(lhs[i], rhs[i], places=places)
            else:
                self.assertAlmostEqual(lhs[i], rhs[i], places=places)

    def assertMatricesAlmostEqual(self, lhs, rhs, places=None):
        self.assertEqual(lhs.shape, rhs.shape, "Marix shapes differ: {} vs {}".format(lhs, rhs))
        n, m = lhs.shape
        for x in range(n):
            for y in range(m):
                self.assertAlmostEqual(
                    lhs[x, y],
                    rhs[x, y],
                    places=places,
                    msg="Matrices {} and {} differ on ({}, {})".format(lhs, rhs, x, y),
                )

    def test_transformation_by_pauli(self):
        # polarization in the XY plane; we represent via Kraus operators
        X = self.ops["X"]
        Y = self.ops["Y"]
        Z = self.ops["Z"]
        p = 0.22
        theta = numpy.pi / 5
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * numpy.array(X) + numpy.sin(theta) * numpy.array(Y))
        results = approximate_quantum_error(Kraus([E0, E1]), operator_dict={"X": X, "Y": Y, "Z": Z})
        expected_results = pauli_error(
            [
                ("X", p * numpy.cos(theta) * numpy.cos(theta)),
                ("Y", p * numpy.sin(theta) * numpy.sin(theta)),
                ("Z", 0),
                ("I", 1 - p),
            ]
        )
        self.assertErrorsAlmostEqual(expected_results, results)

    def test_transformation_by_kraus(self):
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        reset_to_0 = [numpy.array([[1, 0], [0, 0]]), numpy.array([[0, 1], [0, 0]])]
        reset_to_1 = [numpy.array([[0, 0], [1, 0]]), numpy.array([[0, 0], [0, 1]])]
        reset_kraus = [Kraus(reset_to_0), Kraus(reset_to_1)]

        actual = approximate_quantum_error(error, operator_list=reset_kraus)

        p = (1 + gamma - numpy.sqrt(1 - gamma)) / 2
        expected_probs = [1 - p, p, 0]
        self.assertListAlmostEqual(expected_probs, actual.probabilities)

    def test_reset(self):
        # approximating amplitude damping using relaxation operators
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        q = 0
        expected_results = reset_error(p, q)
        results = approximate_quantum_error(error, operator_string="reset")
        self.assertErrorsAlmostEqual(results, expected_results)

    def test_transform(self):
        X = self.ops["X"]
        Y = self.ops["Y"]
        Z = self.ops["Z"]
        p = 0.34
        theta = numpy.pi / 7
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * numpy.array(X) + numpy.sin(theta) * numpy.array(Y))

        results_dict = approximate_quantum_error(
            Kraus([E0, E1]), operator_dict={"X": X, "Y": Y, "Z": Z}
        )
        results_string = approximate_quantum_error(Kraus([E0, E1]), operator_string="pauli")
        results_list = approximate_quantum_error(Kraus([E0, E1]), operator_list=[X, Y, Z])
        results_tuple = approximate_quantum_error(Kraus([E0, E1]), operator_list=(X, Y, Z))

        self.assertErrorsAlmostEqual(results_dict, results_string)
        self.assertErrorsAlmostEqual(results_string, results_list)
        self.assertErrorsAlmostEqual(results_list, results_tuple)

    def test_approx_noise_model(self):
        noise_model = NoiseModel()
        gamma = 0.23
        p = 0.4
        q = 0.33
        ad_error = amplitude_damping_error(gamma)
        r_error = reset_error(p, q)  # should be approximated as-is
        noise_model.add_all_qubit_quantum_error(ad_error, "iden x y s")
        noise_model.add_all_qubit_quantum_error(r_error, "iden z h")

        result = approximate_noise_model(noise_model, operator_string="reset")

        expected_result = NoiseModel()
        gamma_p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        gamma_q = 0
        ad_error_approx = reset_error(gamma_p, gamma_q)
        expected_result.add_all_qubit_quantum_error(ad_error_approx, "iden x y s")
        expected_result.add_all_qubit_quantum_error(r_error, "iden z h")

        self.assertNoiseModelsAlmostEqual(expected_result, result)

    def test_approx_names(self):
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        results_1 = approximate_quantum_error(error, operator_string="pauli")
        results_2 = approximate_quantum_error(error, operator_string="Pauli")
        self.assertErrorsAlmostEqual(results_1, results_2)

    def test_paulis_1_and_2_qubits(self):
        probs = [0.5, 0.3, 0.2]
        paulis_1q = ["X", "Y", "Z"]
        paulis_2q = ["XI", "YI", "ZI"]

        error_1q = pauli_error(zip(paulis_1q, probs))
        error_2q = pauli_error(zip(paulis_2q, probs))

        results_1q = approximate_quantum_error(error_1q, operator_string="pauli")
        results_2q = approximate_quantum_error(error_2q, operator_string="pauli")

        self.assertErrorsAlmostEqual(error_1q, results_1q)
        self.assertErrorsAlmostEqual(error_2q, results_2q, places=2)

        paulis_2q = ["XY", "ZZ", "YI"]
        error_2q = pauli_error(zip(paulis_2q, probs))
        results_2q = approximate_quantum_error(error_2q, operator_string="pauli")
        self.assertErrorsAlmostEqual(error_2q, results_2q, places=2)

    def test_reset_2_qubit(self):
        # approximating amplitude damping using relaxation operators
        gamma = 0.23
        p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        A0 = [[1, 0], [0, numpy.sqrt(1 - gamma)]]
        A1 = [[0, numpy.sqrt(gamma)], [0, 0]]
        error_1 = QuantumError([([(Kraus([A0, A1]), [0]), (IGate(), [1])], 1)])
        error_2 = QuantumError([([(Kraus([A0, A1]), [1]), (IGate(), [0])], 1)])

        expected_results_1 = QuantumError(
            [([(IGate(), [0]), (IGate(), [1])], 1 - p), ([(Reset(), [0]), (IGate(), [1])], p)]
        )
        expected_results_2 = QuantumError(
            [([(IGate(), [1]), (IGate(), [0])], 1 - p), ([(Reset(), [1]), (IGate(), [0])], p)]
        )

        results_1 = approximate_quantum_error(error_1, operator_string="reset")
        results_2 = approximate_quantum_error(error_2, operator_string="reset")

        self.assertErrorsAlmostEqual(results_1, expected_results_1)
        self.assertErrorsAlmostEqual(results_2, expected_results_2)

    def test_clifford(self):
        x_p = 0.1
        y_p = 0.2
        z_p = 0.3
        error = pauli_error([("X", x_p), ("Y", y_p), ("Z", z_p), ("I", 1 - (x_p + y_p + z_p))])
        results = approximate_quantum_error(error, operator_string="clifford")
        self.assertErrorsAlmostEqual(error, results, places=1)

    def test_errors(self):
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        # kraus error is legit, transform_channel_operators are not
        with self.assertRaisesRegex(TypeError, "takes 1 positional argument but 2 were given"):
            approximate_quantum_error(error, 7)
        with self.assertRaises(NoiseError):
            approximate_quantum_error(error, operator_string="seven")

    def test_approx_random_unitary_channel(self):
        # run without raising any error
        noise = Kraus(random_unitary(2, seed=123))
        for opstr in ["pauli", "reset", "clifford"]:
            approximate_quantum_error(noise, operator_string=opstr)

        noise = Kraus(random_unitary(4, seed=123))
        for opstr in ["pauli", "reset"]:
            approximate_quantum_error(noise, operator_string=opstr)

    def test_approx_random_mixed_unitary_channel_1q(self):
        # run without raising any error
        noise1 = UnitaryGate(random_unitary(2, seed=123))
        noise2 = UnitaryGate(random_unitary(2, seed=456))
        noise = QuantumError([(noise1, 0.7), (noise2, 0.3)])
        for opstr in ["pauli", "reset", "clifford"]:
            approximate_quantum_error(noise, operator_string=opstr)

    def test_approx_random_mixed_unitary_channel_2q(self):
        # run without raising any error
        noise1 = UnitaryGate(random_unitary(4, seed=123))
        noise2 = UnitaryGate(random_unitary(4, seed=456))
        noise = QuantumError([(noise1, 0.7), (noise2, 0.3)])
        for opstr in ["pauli", "reset"]:
            approximate_quantum_error(noise, operator_string=opstr)


if __name__ == "__main__":
    unittest.main()
