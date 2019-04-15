import unittest
from aer.noise import NoiseTransformer
import numpy

#TODO: skip tests if CVXOPT is not present

class TestNoiseTransformer(unittest.TestCase):
    def setUp(self):
        #TODO: replace with Qiskit based defs
        X = numpy.array([[0, 1], [1, 0]])
        Y = numpy.array([[0, -1j], [1j, 0]])
        Z = numpy.array([[1, 0], [0, -1]])
        S = numpy.array([[1, 0], [0, 1j]])
        H = numpy.sqrt(2) * numpy.array([[1, 1], [1, -1]])
        self.ops = {'X': X, 'Y': Y, 'Z': Z, 'H': H, 'S': S}
        self.n = NoiseTransformer()

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
        #polarization
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.22
        theta = numpy.pi / 5
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)
        results = n.transform({"X": X, "Y": Y, "Z": Z}, (E0, E1))
        expected_results = {'X': p*numpy.cos(theta)*numpy.cos(theta), 'Y': p*numpy.sin(theta)*numpy.sin(theta), 'Z': 0}
        for op in results.keys():
            self.assertAlmostEqual(expected_results[op], results[op], 3)

        #now try again without fidelity; should be the same
        n.use_honesty_constraint = False
        results = n.transform({"X": X, "Y": Y, "Z": Z}, (E0, E1))
        for op in results.keys():
            self.assertAlmostEqual(expected_results[op], results[op], 3)

    def test_relaxation(self):
        # amplitude damping
        gamma = 0.5
        E0 = numpy.array([[1, 0], [0, numpy.sqrt(1 - gamma)]])
        E1 = numpy.array([[0, numpy.sqrt(gamma)], [0, 0]])
        results = self.n.transform("relaxation", (E0, E1))
        expected_results = {'p': (gamma - numpy.sqrt(1 - gamma) + 1) / 2, 'q': 0}
        for op in results.keys():
            self.assertAlmostEqual(expected_results[op], results[op], 3)

    def test_transform(self):
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        p = 0.34
        theta = numpy.pi / 7
        E0 = numpy.sqrt(1 - p) * numpy.array(numpy.eye(2))
        E1 = numpy.sqrt(p) * (numpy.cos(theta) * X + numpy.sin(theta) * Y)

        results_dict = self.n.transform({"X": X, "Y": Y, "Z": Z}, (E0, E1))
        results_string = self.n.transform('pauli', (E0, E1))
        results_list = list(self.n.transform([X, Y, Z], (E0, E1)))
        results_tuple = list(self.n.transform((X, Y, Z), (E0, E1)))
        results_list_as_dict = dict(zip(['X', 'Y', 'Z'],results_list))

        self.assertDictAlmostEqual(results_dict, results_string)
        self.assertListAlmostEqual(results_list, results_tuple)
        self.assertDictAlmostEqual(results_list_as_dict, results_dict)

    def test_fidelity(self):
        n = NoiseTransformer()
        expected_fidelity = {'X': 0, 'Y': 0, 'Z': 0, 'H': 0, 'S': 2}
        for key in expected_fidelity:
            self.assertAlmostEqual(expected_fidelity[key], n.fidelity([self.ops[key]]), msg = "Wrong fidelity for {}".format(key))

    def test_numeric_channel_matrix_representation(self):
        n = NoiseTransformer()
        gamma = 0.5
        E0 = numpy.array([[1, 0], [0, numpy.sqrt(1 - gamma)]])
        E1 = numpy.array([[0, numpy.sqrt(gamma)], [0, 0]])
        numeric_matrix = n.numeric_channel_matrix_representation((E0, E1))
        expected_numeric_matrix = numpy.array([[1, 0, 0, 0], [0, 0.707106781186548, 0, 0],[0,0,0.707106781186548,0],[0.500000000000000, 0, 0, 0.500000000000000]])
        self.assertMatricesAlmostEqual(expected_numeric_matrix, numeric_matrix)

    def test_op_name_to_matrix(self):
        X = self.ops['X']
        Y = self.ops['Y']
        Z = self.ops['Z']
        H = self.ops['H']
        S = self.ops['S']
        self.assertTrue((X*Y*Z == self.n.op_name_to_matrix('XYZ', self.ops)).all())
        self.assertTrue((S*X*S*Y*H*Z*H*S*X*Z == self.n.op_name_to_matrix('SXSYHZHSXZ', self.ops)).all())

    def test_qobj_noise_to_kraus_operators(self):
        amplitude_damping_kraus_noise = {
            "type": "qerror",
            "operations": ["h"],
            "probabilities": [1.0],
            "instructions": [
                [{"name": "kraus", "qubits": [0], "params": [
                    [[[1, 0], [0, 0]], [[0, 0], [0.5, 0]]],
                    [[[0, 0], [0.86602540378, 0]], [[0, 0], [0, 0]]]]}]
            ]
        }
        n = NoiseTransformer()
        kraus_operators = n.qobj_noise_to_kraus_operators(amplitude_damping_kraus_noise)
        matrices = amplitude_damping_kraus_noise['instructions'][0][0]['params']
        expected_kraus_operators = [n.qobj_matrix_to_numpy_matrix(m) for m in matrices]
        self.assertListAlmostEqual(expected_kraus_operators, kraus_operators, places=4)

        #now for malformed data
        amplitude_damping_kraus_noise['instructions'][0][0] = {"name": "TTG", "qubits": [0]}
        with self.assertRaises(RuntimeError):
            kraus_operators = n.qobj_noise_to_kraus_operators(amplitude_damping_kraus_noise)

    def test_qobj_conversion(self):
        import json
        with open("../data/qobj_noise_kraus.json") as f:
            qobj = json.load(f)
        n = NoiseTransformer()
        result_qobj = n.transform_qobj('relaxation', qobj)

        gamma = 0.75  # can be seen from the json file
        expected_p = (gamma - numpy.sqrt(1 - gamma) + 1) / 2
        expected_q = 0

        expected_matrices = [
            numpy.sqrt(1-(expected_p + expected_q)) * numpy.array([[1, 0], [0, 1]]),
            numpy.sqrt(expected_p) * numpy.array([[1, 0], [0, 0]]),
            numpy.sqrt(expected_p) * numpy.array([[0, 1], [0, 0]]),
            numpy.sqrt(expected_q) * numpy.array([[0, 0], [0, 1]]),
            numpy.sqrt(expected_q) * numpy.array([[0, 0], [1, 0]])
        ]
        matrices = result_qobj['config']['noise_model']['errors'][0]['instructions'][0][0]['params']
        matrices = [n.qobj_matrix_to_numpy_matrix(m) for m in matrices]
        self.assertListAlmostEqual(expected_matrices, matrices, places = 3)

        #let's also run on something without noise and verify nothing changes
        with open("../data/qobj_snapshot_expval_pauli.json") as f:
            qobj = json.load(f)
        result_qobj = n.transform_qobj('relaxation', qobj)
        self.assertEqual(qobj, result_qobj)

    def test_errors(self):
        n = NoiseTransformer()
        gamma = 0.5
        E0 = numpy.array([[1, 0], [0, numpy.sqrt(1 - gamma)]])
        E1 = numpy.array([[0, numpy.sqrt(gamma)], [0, 0]])
        # kraus error is legit, transform_channel_operators are not
        with self.assertRaisesRegex(RuntimeError, "7 is not an appropriate input to transform"):
            n.transform(7, (E0, E1))
        with self.assertRaisesRegex(RuntimeError, "No information about noise type seven"):
            n.transform("seven", (E0, E1))

        #let's pretend cvxopt does not exist; the script should raise ImportError with proper message
        import unittest.mock
        import sys
        with unittest.mock.patch.dict(sys.modules, {'cvxopt': None}):
            with self.assertRaisesRegex(ImportError, "The CVXOPT library is required to use this module"):
                n.transform("relaxation", (E0, E1))



if __name__ == '__main__':
    unittest.main()