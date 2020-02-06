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
NoiseTransformer class tests
"""

import unittest
import numpy
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.utils import NoiseTransformer
from qiskit.providers.aer.utils import approximate_quantum_error
from qiskit.providers.aer.utils import approximate_noise_model
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import reset_error
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError


class TestNoiseTransformer(unittest.TestCase):
    def setUp(self):
        self.ops = {
            'X': standard_gate_unitary('x'),
            'Y': standard_gate_unitary('y'),
            'Z': standard_gate_unitary('z'),
            'H': standard_gate_unitary('h'),
            'S': standard_gate_unitary('s')
        }
        self.n = NoiseTransformer()

    def assertNoiseModelsAlmostEqual(self, lhs, rhs, places=3):
        self.assertNoiseDictsAlmostEqual(
            lhs._nonlocal_quantum_errors,
            rhs._nonlocal_quantum_errors,
            places=places)
        self.assertNoiseDictsAlmostEqual(
            lhs._local_quantum_errors,
            rhs._local_quantum_errors,
            places=places)
        self.assertNoiseDictsAlmostEqual(
            lhs._default_quantum_errors,
            rhs._default_quantum_errors,
            places=places)
        self.assertNoiseDictsAlmostEqual(
            lhs._local_readout_errors,
            rhs._local_readout_errors,
            places=places)
        if lhs._default_readout_error is not None:
            self.assertTrue(rhs._default_readout_error is not None)
            self.assertErrorsAlmostEqual(
                lhs._default_readout_error,
                rhs._default_readout_error,
                places=places)
        else:
            self.assertTrue(rhs._default_readout_error is None)

    def assertNoiseDictsAlmostEqual(self, lhs, rhs, places=3):
        keys = set(lhs.keys()).union(set(rhs.keys()))
        for key in keys:
            self.assertTrue(
                key in lhs.keys(),
                msg="Key {} is missing from lhs".format(key))
            self.assertTrue(
                key in rhs.keys(),
                msg="Key {} is missing from rhs".format(key))
            if isinstance(lhs[key], dict):
                self.assertNoiseDictsAlmostEqual(lhs[key], rhs[key], places=places)
            else:
                self.assertErrorsAlmostEqual(lhs[key], rhs[key], places=places)

    def assertErrorsAlmostEqual(self, lhs, rhs, places=3):
        self.assertMatricesAlmostEqual(lhs.to_quantumchannel()._data,
                                       rhs.to_quantumchannel()._data, places)

    def assertDictAlmostEqual(self, lhs, rhs, places=None):
        keys = set(lhs.keys()).union(set(rhs.keys()))
        for key in keys:
            self.assertAlmostEqual(
                lhs.get(key),
                rhs.get(key),
                msg="Not almost equal for key {}: {} !~ {}".format(
                    key, lhs.get(key), rhs.get(key)),
                places=places)

    def assertListAlmostEqual(self, lhs, rhs, places=None):
        self.assertEqual(
            len(lhs),
            len(rhs),
            msg="List lengths differ: {} != {}".format(len(lhs), len(rhs)))
        for i in range(len(lhs)):
            if isinstance(lhs[i], numpy.ndarray) and isinstance(
                    rhs[i], numpy.ndarray):
                self.assertMatricesAlmostEqual(lhs[i], rhs[i], places=places)
            else:
                self.assertAlmostEqual(lhs[i], rhs[i], places=places)

    def assertMatricesAlmostEqual(self, lhs, rhs, places=None):
        self.assertEqual(lhs.shape, rhs.shape,
                         "Marix shapes differ: {} vs {}".format(lhs, rhs))
        n, m = lhs.shape
        for x in range(n):
            for y in range(m):
                self.assertAlmostEqual(
                    lhs[x, y],
                    rhs[x, y],
                    places=places,
                    msg="Matrices {} and {} differ on ({}, {})".format(
                        lhs, rhs, x, y))

    def test_transformation_by_pauli(self):
        n = NoiseTransformer()
        # polarization in the XY plane; we represent via Kraus operators
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.22
        theta = numpy.pi / 5
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)
        results = approximate_quantum_error((E0, E1),
                                            operator_dict={
                                                "X": X,
                                                "Y": Y,
                                                "Z": Z})
        expected_results = pauli_error(
            [('X', p * numpy.cos(theta) * numpy.cos(theta)),
             ('Y', p * numpy.sin(theta) * numpy.sin(theta)), ('Z', 0),
             ('I', 1 - p)])
        self.assertErrorsAlmostEqual(expected_results, results)

        # now try again without fidelity; should be the same
        n.use_honesty_constraint = False
        results = approximate_quantum_error((E0, E1),
                                            operator_dict={
                                                "X": X,
                                                "Y": Y,
                                                "Z": Z})
        self.assertErrorsAlmostEqual(expected_results, results)

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
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.34
        theta = numpy.pi / 7
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)

        results_dict = approximate_quantum_error((E0, E1),
                                                 operator_dict={
                                                     "X": X,
                                                     "Y": Y,
                                                     "Z": Z})
        results_string = approximate_quantum_error((E0, E1),
                                                   operator_string='pauli')
        results_list = approximate_quantum_error((E0, E1),
                                                 operator_list=[X, Y, Z])
        results_tuple = approximate_quantum_error((E0, E1),
                                                  operator_list=(X, Y, Z))

        self.assertErrorsAlmostEqual(results_dict, results_string)
        self.assertErrorsAlmostEqual(results_string, results_list)
        self.assertErrorsAlmostEqual(results_list, results_tuple)

    def test_fidelity(self):
        n = NoiseTransformer()
        expected_fidelity = {'X': 0, 'Y': 0, 'Z': 0, 'H': 0, 'S': 2}
        for key in expected_fidelity:
            self.assertAlmostEqual(
                expected_fidelity[key],
                n.fidelity([self.ops[key]]),
                msg="Wrong fidelity for {}".format(key))

    def test_approx_noise_model(self):
        noise_model = NoiseModel()
        gamma = 0.23
        p = 0.4
        q = 0.33
        ad_error = amplitude_damping_error(gamma)
        r_error = reset_error(p, q)  # should be approximated as-is
        noise_model.add_all_qubit_quantum_error(ad_error, 'iden x y s')
        noise_model.add_all_qubit_quantum_error(r_error, 'iden z h')

        result = approximate_noise_model(noise_model, operator_string="reset")

        expected_result = NoiseModel()
        gamma_p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        gamma_q = 0
        ad_error_approx = reset_error(gamma_p, gamma_q)
        expected_result.add_all_qubit_quantum_error(ad_error_approx,
                                                    'iden x y s')
        expected_result.add_all_qubit_quantum_error(r_error, 'iden z h')

        self.assertNoiseModelsAlmostEqual(expected_result, result)

    def test_clifford(self):
        x_p = 0.17
        y_p = 0.13
        z_p = 0.34
        error = pauli_error([('X', x_p), ('Y', y_p), ('Z', z_p),
                             ('I', 1 - (x_p + y_p + z_p))])
        results = approximate_quantum_error(error, operator_string="clifford")
        self.assertErrorsAlmostEqual(error, results)

    def test_approx_names(self):
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        results_1 = approximate_quantum_error(error, operator_string="pauli")
        results_2 = approximate_quantum_error(error, operator_string="Pauli")
        self.assertErrorsAlmostEqual(results_1, results_2)

    def test_paulis_1_and_2_qubits(self):
        probs = [0.5, 0.3, 0.2]
        paulis_1q = ['X', 'Y', 'Z']
        paulis_2q = ['XI', 'YI', 'ZI']

        error_1q = pauli_error(zip(paulis_1q, probs))
        error_2q = pauli_error(zip(paulis_2q, probs))

        results_1q = approximate_quantum_error(error_1q, operator_string="pauli")
        results_2q = approximate_quantum_error(error_2q, operator_string="pauli")

        self.assertErrorsAlmostEqual(error_1q, results_1q)
        self.assertErrorsAlmostEqual(error_2q, results_2q, places = 2)

        paulis_2q = ['XY', 'ZZ', 'YI']
        error_2q = pauli_error(zip(paulis_2q, probs))
        results_2q = approximate_quantum_error(error_2q, operator_string="pauli")
        self.assertErrorsAlmostEqual(error_2q, results_2q, places=2)

    def test_reset_2_qubit(self):
        # approximating amplitude damping using relaxation operators
        gamma = 0.23
        p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        q = 0
        A0 = [[1, 0], [0, numpy.sqrt(1 - gamma)]]
        A1 = [[0, numpy.sqrt(gamma)], [0, 0]]
        error_1 = QuantumError([([{'name': 'kraus', 'qubits': [0], 'params': [A0, A1]},
                                  {'name': 'id', 'qubits': [1]}
                                  ], 1)])
        error_2 = QuantumError([([{'name': 'kraus', 'qubits': [1], 'params': [A0, A1]},
                                  {'name': 'id', 'qubits': [0]}
                                  ], 1)])

        expected_results_1 = QuantumError([
            ([{'name': 'id', 'qubits': [0]}, {'name': 'id', 'qubits': [1]}], 1-p),
            ([{'name': 'reset', 'qubits': [0]}, {'name': 'id', 'qubits': [1]}],p),
        ])
        expected_results_2 = QuantumError([
            ([{'name': 'id', 'qubits': [1]}, {'name': 'id', 'qubits': [0]}], 1 - p),
            ([{'name': 'reset', 'qubits': [1]}, {'name': 'id', 'qubits': [0]}], p),
        ])

        results_1 = approximate_quantum_error(error_1, operator_string="reset")
        results_2 = approximate_quantum_error(error_2, operator_string="reset")

        self.assertErrorsAlmostEqual(results_1, expected_results_1)
        self.assertErrorsAlmostEqual(results_2, expected_results_2)



    def test_errors(self):
        gamma = 0.23
        error = amplitude_damping_error(gamma)
        # kraus error is legit, transform_channel_operators are not
        with self.assertRaisesRegex(
                TypeError, "takes 1 positional argument but 2 were given"):
            approximate_quantum_error(error, 7)
        with self.assertRaisesRegex(RuntimeError,
                                    "No information about noise type seven"):
            approximate_quantum_error(error, operator_string="seven")


if __name__ == '__main__':
    unittest.main()
