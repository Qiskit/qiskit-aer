# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""tests for operator_models.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.operator_models import FrameFreqHelper, OperatorModel, vector_apply_diag_frame
from qiskit.providers.aer.pulse_new.models.signals import Constant, ConstantSignal, VectorSignal


class TestOperatorModel(unittest.TestCase):
    """Tests for OperatorModel."""

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

        # define a basic model
        w = 2.
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi *  r * self.X / 2]
        signals = [Constant(w), ConstantSignal(1., w)]

        self.w = 2
        self.r = r
        self.basic_model = OperatorModel(operators=operators, signals=signals)

    def test_frame_operator_errors(self):
        """Check different modes of error raising for frame setting."""

        # 1d array
        try:
            self.basic_model.frame_operator = np.array([1., 1.])
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # 2d array
        try:
            self.basic_model.frame_operator = np.array([[1., 0.], [0., 1.]])
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # Operator
        try:
            self.basic_model.frame_operator = self.Z
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

    def test_diag_frame_operator(self):
        """Test setting a diagonal frame operator."""

        self._frame_operator_test(np.array([1j, -1j]), 1.123)
        self._frame_operator_test(np.array([1j, -1j]), np.pi)


    def test_non_diag_frame_operator(self):
        """Test setting a diagonal frame operator."""
        self._frame_operator_test(-1j * (self.Y + self.Z), 1.123)
        self._frame_operator_test(-1j * (self.Y + self.Z), np.pi)


    def _frame_operator_test(self, frame_operator, t):
        """Routine for testing setting of valid frame operators."""

        self.basic_model.frame_operator = frame_operator

        # convert to 2d array
        if isinstance(frame_operator, Operator):
            frame_operator = frame_operator.data
        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:
            frame_operator = np.diag(frame_operator)

        value = self.basic_model.evaluate(t)

        i2pi = -1j * 2 * np.pi

        U = expm(- frame_operator * t)

        # drive coefficient
        d_coeff = self.r * np.cos(2 * np.pi * self.w * t)

        # manually evaluate frame
        expected = (i2pi * self.w * U @ self.Z.data @ U.conj().transpose() / 2 +
                    d_coeff * i2pi * U @ self.X.data @ U.conj().transpose() / 2 -
                    frame_operator)

        self.assertAlmostEqual(value, expected)


    def assertAlmostEqual(self, A, B, tol=10**-12):
        self.assertTrue(np.abs(A - B).max() < tol)


class TestFrameFreqHelper(unittest.TestCase):

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')


    def test_evaluate_no_cutoff(self):
        """test evaluate with a non-diagonal frame and no cutoff freq."""

        frame_op = -1j * np.pi * self.X
        operators = [Operator(-1j * np.pi * self.Z), Operator(-1j * self.X / 2)]
        carrier_freqs = np.array([0., 1.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        t = np.pi * 0.02
        coeffs = np.array([1., 1.])
        val = helper.evaluate(t, coeffs)
        U = expm(frame_op.data * t)
        U_adj = U.conj().transpose()
        expected = (U_adj @ (-1j * np.pi * self.Z.data +
                             1j * np.pi * self.X.data +
                            -1j * np.cos(2 * np.pi * t) * self.X.data / 2) @ U)

        self.assertAlmostEqual(val, expected)

        # with complex envelope
        t = np.pi * 0.02
        coeffs = np.array([1., 1. + 2 * 1j])
        val = helper.evaluate(t, coeffs)
        U = expm(frame_op.data * t)
        U_adj = U.conj().transpose()
        expected = (U_adj @ (-1j * np.pi * self.Z.data +
                             1j * np.pi * self.X.data +
                            -1j * np.cos(2 * np.pi * t) * self.X.data / 2 +
                            1j * 2 * np.sin(2 * np.pi * t) * self.X.data / 2) @ U)

        self.assertAlmostEqual(val, expected)

    def test_evaluate_diag_frame_no_cutoff(self):
        """test evaluate with a diagonal frame and no cutoff freq."""

        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [Operator(-1j * np.pi * self.Z), Operator(-1j * self.X / 2)]
        carrier_freqs = np.array([0., 1.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        t = np.pi * 0.02
        coeffs = np.array([1., 1.])
        val = helper.evaluate(t, coeffs)
        U = np.diag(np.exp(frame_op * t))
        U_adj = U.conj().transpose()
        expected = -1j * np.cos(2 * np.pi * t) * U_adj @ self.X.data @ U / 2

        self.assertAlmostEqual(val, expected)

        # with complex envelope
        t = np.pi * 0.02
        coeffs = np.array([1., 1. + 2 * 1j])
        val = helper.evaluate(t, coeffs)
        U = np.diag(np.exp(frame_op * t))
        U_adj = U.conj().transpose()
        expected = -1j * (np.cos(2 * np.pi * t) * U_adj @ self.X.data @ U -
                    2 * np.sin(2 * np.pi * t) * U_adj @ self.X.data @ U ) / 2

        self.assertAlmostEqual(val, expected)

    def test_evaluate_no_frame(self):
        """Test FrameFreqHelper.evaluate with no frame or cutoff."""

        operators = [self.X, self.Y, self.Z]
        carrier_freqs = np.array([1., 2., 3.])

        ffhelper = FrameFreqHelper(operators, carrier_freqs)

        t = 0.123
        coeffs = np.array([1., 1j, 1 + 1j])

        out = ffhelper.evaluate(t, coeffs)
        sig_vals = np.real(coeffs * np.exp(1j * 2 * np.pi * carrier_freqs * t))
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(sig_vals, ops_as_arrays, axes=1)

        self.assertAlmostEqual(out, expected_out)

        t = 0.123 * np.pi
        coeffs = np.array([4.131, 3.23, 2.1 + 3.1j])

        out = ffhelper.evaluate(t, coeffs)
        sig_vals = np.real(coeffs * np.exp(1j * 2 * np.pi * carrier_freqs * t))
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(sig_vals, ops_as_arrays, axes=1)

        self.assertAlmostEqual(out, expected_out)

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        operators = [self.X]
        carrier_freqs = np.array([1.])

        ffhelper = FrameFreqHelper(operators, carrier_freqs)

        t = 0.123
        y = np.array([1., 1j])
        out = ffhelper.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = ffhelper.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

        t = 100.12498
        y = np.eye(2)
        out = ffhelper.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = ffhelper.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

    def test_internal_helper_mats_no_cutoff(self):
        """Test internal setup steps for helper matrices with no cutoff freq."""

        # no cutoff with already diagonal frame
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = None

        self.assertTrue(helper._M_cutoff == M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

        # same test but with frame given as a 2d array
        # in this case diagonalization will occur, causing eigenvalues to
        # be sorted in ascending order
        frame_op = -1j * np.pi * np.array([[-1., 0], [0, 1.]])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])

        helper = FrameFreqHelper(operators, carrier_freqs, frame_op)

        D_diff = -1j * np.pi * np.array([[0, 2.], [-2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = None

        self.assertTrue(helper._M_cutoff == M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

    def test_internal_helper_mats_with_cutoff(self):
        """Test internal setup steps for helper matrices with cutoff freq."""

        # cutoff test
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [self.X, self.X, self.X]
        carrier_freqs = np.array([1., 2., 3.])
        cutoff_freq = 3.

        helper = FrameFreqHelper(operators,
                                 carrier_freqs,
                                 frame_op,
                                 cutoff_freq)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = np.array([[[1, 1],
                              [1, 1]],
                             [[1, 0],
                              [1, 1]],
                             [[0, 0],
                              [1, 0]]
                             ])

        self.assertAlmostEqual(helper._M_cutoff, M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

        # same test with lower cutoff
        cutoff_freq = 2.

        helper = FrameFreqHelper(operators,
                                 carrier_freqs,
                                 frame_op,
                                 cutoff_freq)

        D_diff = -1j * np.pi * np.array([[0, -2.], [2., 0.]])
        im_freqs = 1j * 2 * np.pi * carrier_freqs
        S_expect = np.array([w + D_diff for w in im_freqs])
        M_expect = np.array([[[1, 0],
                              [1, 1]],
                             [[0, 0],
                              [1, 0]],
                             [[0, 0],
                              [0, 0]]
                             ])

        self.assertAlmostEqual(helper._M_cutoff, M_expect)
        self.assertAlmostEqual(helper._S, S_expect)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
