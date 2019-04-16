import unittest
import numpy
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary
from qiskit.providers.aer.noise import NoiseTransformer
from qiskit.providers.aer.noise import approximate_quantum_error
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error
from qiskit.providers.aer.noise.errors.standard_errors import reset_error
from qiskit.providers.aer.noise.errors.standard_errors import pauli_error

#TODO: skip tests if CVXOPT is not present

class TestNoiseTransformer(unittest.TestCase):
    def setUp(self):
        self.ops = {'X': standard_gate_unitary('x'),
                    'Y': standard_gate_unitary('y'),
                    'Z': standard_gate_unitary('z'),
                    'H': standard_gate_unitary('h'),
                    'S': standard_gate_unitary('s')
                    }
        self.n = NoiseTransformer()

    def assertErrorsAlmostEqual(self, lhs, rhs, places = 3):
        self.assertMatricesAlmostEqual(lhs.to_channel()._data, rhs.to_channel()._data, places)


    def assertDictAlmostEqual(self, lhs, rhs, places = None):
        keys = set(lhs.keys()).union(set(rhs.keys()))
        for key in keys:
            self.assertAlmostEqual(lhs.get(key), rhs.get(key), msg = "Not almost equal for key {}: {} !~ {}".format(key, lhs.get(key), rhs.get(key)), places = places)

    def assertListAlmostEqual(self, lhs, rhs, places = None):
        self.assertEqual(len(lhs), len(rhs), msg = "List lengths differ: {} != {}".format(len(lhs), len(rhs)))
        for i in range(len(lhs)):
            if isinstance(lhs[i], numpy.ndarray) and isinstance(rhs[i], numpy.ndarray):
                self.assertMatricesAlmostEqual(lhs[i], rhs[i], places = places)
            else:
                self.assertAlmostEqual(lhs[i], rhs[i], places = places)

    def assertMatricesAlmostEqual(self, lhs, rhs, places = None):
        self.assertEqual(lhs.shape, rhs.shape, "Marix shapes differ: {} vs {}".format(lhs, rhs))
        n, m = lhs.shape
        for x in range(n):
            for y in range(m):
                self.assertAlmostEqual(lhs[x,y], rhs[x,y], places = places, msg="Matrices {} and {} differ on ({}, {})".format(lhs, rhs, x, y))


    def test_transformation_by_pauli(self):
        n = NoiseTransformer()
        #polarization in the XY plane; we represent via Kraus operators
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.22
        theta = numpy.pi / 5
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)
        results = approximate_quantum_error((E0, E1), operator_dict={"X": X, "Y": Y, "Z": Z})
        expected_results = pauli_error([('X', p*numpy.cos(theta)*numpy.cos(theta)),
                                        ('Y', p*numpy.sin(theta)*numpy.sin(theta)),
                                        ('Z', 0),
                                        ('I', 1-p)])
        self.assertErrorsAlmostEqual(expected_results, results)


        #now try again without fidelity; should be the same
        n.use_honesty_constraint = False
        results = approximate_quantum_error((E0, E1), operator_dict={"X": X, "Y": Y, "Z": Z})
        self.assertErrorsAlmostEqual(expected_results, results)

    def test_reset(self):
        # approximating amplitude damping using relaxation operators
        gamma = 0.5
        error = amplitude_damping_error(gamma)
        p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        q = 0
        expected_results = reset_error(p,q)
        results = approximate_quantum_error(error, "reset")
        self.assertErrorsAlmostEqual(results, expected_results)

    def test_transform(self):
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.34
        theta = numpy.pi / 7
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)

        results_dict = approximate_quantum_error((E0, E1), operator_dict={"X": X, "Y": Y, "Z": Z})
        results_string = approximate_quantum_error((E0, E1), operator_string='pauli')
        results_list = approximate_quantum_error((E0, E1), operator_list=[X, Y, Z])
        results_tuple = approximate_quantum_error((E0, E1), operator_list=(X, Y, Z))

        self.assertErrorsAlmostEqual(results_dict, results_string)
        self.assertErrorsAlmostEqual(results_string, results_list)
        self.assertErrorsAlmostEqual(results_list, results_tuple)

    # def test_fidelity(self):
    #     n = NoiseTransformer()
    #     expected_fidelity = {'X': 0, 'Y': 0, 'Z': 0, 'H': 0, 'S': 2}
    #     for key in expected_fidelity:
    #         self.assertAlmostEqual(expected_fidelity[key], n.fidelity([self.ops[key]]), msg = "Wrong fidelity for {}".format(key))
    #
    # def test_numeric_channel_matrix_representation(self):
    #     n = NoiseTransformer()
    #     gamma = 0.5
    #     E0 = numpy.array([[1, 0], [0, numpy.sqrt(1 - gamma)]])
    #     E1 = numpy.array([[0, numpy.sqrt(gamma)], [0, 0]])
    #     numeric_matrix = n.numeric_channel_matrix_representation((E0, E1))
    #     expected_numeric_matrix = numpy.array([[1, 0, 0, 0], [0, 0.707106781186548, 0, 0],[0,0,0.707106781186548,0],[0.500000000000000, 0, 0, 0.500000000000000]])
    #     self.assertMatricesAlmostEqual(expected_numeric_matrix, numeric_matrix)
    #
    # def test_op_name_to_matrix(self):
    #     X = self.ops['X']
    #     Y = self.ops['Y']
    #     Z = self.ops['Z']
    #     H = self.ops['H']
    #     S = self.ops['S']
    #     self.assertTrue((X*Y*Z == self.n.op_name_to_matrix('XYZ', self.ops)).all())
    #     self.assertTrue((S*X*S*Y*H*Z*H*S*X*Z == self.n.op_name_to_matrix('SXSYHZHSXZ', self.ops)).all())
    #
    # def test_qobj_noise_to_kraus_operators(self):
    #     amplitude_damping_kraus_noise = {
    #         "type": "qerror",
    #         "operations": ["h"],
    #         "probabilities": [1.0],
    #         "instructions": [
    #             [{"name": "kraus", "qubits": [0], "params": [
    #                 [[[1, 0], [0, 0]], [[0, 0], [0.5, 0]]],
    #                 [[[0, 0], [0.86602540378, 0]], [[0, 0], [0, 0]]]]}]
    #         ]
    #     }
    #     n = NoiseTransformer()
    #     kraus_operators = n.qobj_noise_to_kraus_operators(amplitude_damping_kraus_noise)
    #     matrices = amplitude_damping_kraus_noise['instructions'][0][0]['params']
    #     expected_kraus_operators = [n.qobj_matrix_to_numpy_matrix(m) for m in matrices]
    #     self.assertListAlmostEqual(expected_kraus_operators, kraus_operators, places=4)
    #
    #     #now for malformed data
    #     amplitude_damping_kraus_noise['instructions'][0][0] = {"name": "TTG", "qubits": [0]}
    #     with self.assertRaises(RuntimeError):
    #         kraus_operators = n.qobj_noise_to_kraus_operators(amplitude_damping_kraus_noise)
    #
    # def test_errors(self):
    #     n = NoiseTransformer()
    #     gamma = 0.5
    #     E0 = numpy.array([[1, 0], [0, numpy.sqrt(1 - gamma)]])
    #     E1 = numpy.array([[0, numpy.sqrt(gamma)], [0, 0]])
    #     # kraus error is legit, transform_channel_operators are not
    #     with self.assertRaisesRegex(RuntimeError, "7 is not an appropriate input to transform"):
    #         n.transform(7, (E0, E1))
    #     with self.assertRaisesRegex(RuntimeError, "No information about noise type seven"):
    #         n.transform("seven", (E0, E1))
    #
    #     #let's pretend cvxopt does not exist; the script should raise ImportError with proper message
    #     import unittest.mock
    #     import sys
    #     with unittest.mock.patch.dict(sys.modules, {'cvxopt': None}):
    #         with self.assertRaisesRegex(ImportError, "The CVXOPT library is required to use this module"):
    #             n.transform("relaxation", (E0, E1))



if __name__ == '__main__':
    unittest.main()