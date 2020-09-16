"""tests for DE_Methods.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse.de.DE_Options import DE_Options
from qiskit.providers.aer.pulse.de.DE_Methods import (ODE_Method,
                                                      RK4,
                                                      ScipyODE,
                                                      QiskitZVODE,
                                                      method_from_string)

class TestDE_Methods(unittest.TestCase):

    def setUp(self):
        # set up a 2d test problem
        self.t0 = 0.
        self.y0 = np.eye(2)

        self.X = np.array([[0., 1.], [1., 0.]], dtype=complex)
        self.Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
        self.Z = np.array([[1., 0.], [0., -1.]], dtype=complex)

        # define rhs in terms of constant a constant generator
        def generator(t):
            return -1j * 2 * np.pi * self.X / 2

        def rhs(t, y):
            return generator(t) @ y

        self.rhs = {'rhs': rhs}

    def test_method_from_string(self):
        """Test method_from_string"""

        method = method_from_string('scipy-RK45')
        self.assertTrue(method == ScipyODE)

        method = method_from_string('zvode-adams')
        self.assertTrue(method == QiskitZVODE)

    def test_ScipyODE_options_and_defaults(self):
        """Test option handling for ScipyODE solver."""

        options = DE_Options(method='scipy-RK45')
        solver = ScipyODE(options=options)

        # test restructuring/default handling for this solver
        self.assertTrue(solver.options.method == 'RK45')
        self.assertTrue(solver.options.max_step == np.inf)

    def test_QiskitZVODE_options_and_defaults(self):
        """Test option handling for QiskitZVODE solver."""

        options = DE_Options(method='zvode-adams')
        solver = QiskitZVODE(t0 = 0., y0=np.array([1.]), rhs=self.rhs, options=options)

        # test restructuring/default handling for this solver
        self.assertTrue(solver.options.method == 'adams')
        self.assertTrue(solver.options.first_step == 0)
        self.assertTrue(solver.options.max_step == 0)
        self.assertTrue(solver.options.min_step == 0)

    def test_QiskitZVODE_instantiation_error(self):
        """Test option handling for QiskitZVODE solver."""

        expected_message = 'QiskitZVODE solver requires t0, y0, and rhs at instantiation.'

        options = DE_Options(method='zvode-adams')
        try:
            solver = QiskitZVODE(options=options)
        except Exception as exception:
            self.assertEqual(str(exception), expected_message)

    def test_standard_problems_ScipyODE_RK45(self):
        """Run standard variable step tests for scipy-RK45."""
        self._test_variable_step_method('scipy-RK45')

    def test_standard_problems_ScipyODE_RK23(self):
        """Run standard variable step tests for scipy-RK23."""
        self._test_variable_step_method('scipy-RK23')

    def test_standard_problems_ScipyODE_BDF(self):
        """Run standard variable step tests for scipy-BDF."""
        self._test_variable_step_method('scipy-BDF')

    def test_standard_problems_QiskitZVODE_adams(self):
        """Run standard variable step tests for zvode-adams."""
        self._test_variable_step_method('zvode-adams')

    def _test_variable_step_method(self, method_str):
        """Some tests for a variable step method."""

        # get method and set general options
        ode_method = method_from_string(method_str)
        options = DE_Options(method=method_str, atol=10**-10, rtol=10**-10)

        # run on matrix problem
        solver = ode_method(t0=self.t0, y0=self.y0, rhs=self.rhs, options=options)
        solver.integrate(1.)
        expected = expm(-1j * np.pi * self.X)

        # set the comparison tolerance to be somewhat lenient
        self.assertAlmostEqual(solver.y, expected, tol=10**-8)

        # test with an arbitrary problem
        def rhs(t, y):
            return np.array([t**2])

        solver = ode_method(t0=0., y0=np.array(0.), rhs={'rhs': rhs}, options=options)

        solver.integrate(1.)
        expected = 1./3
        self.assertAlmostEqual(solver.y, expected, tol=10**-9)

    def test_RK4(self):
        """Run tests on RK4 fixed-step solver."""
        ode_method = method_from_string('RK4')
        options = DE_Options(max_dt=10**-3)

        # run on matrix problem
        solver = ode_method(t0=self.t0, y0=self.y0, rhs=self.rhs, options=options)
        solver.integrate(1.)
        expected = expm(-1j * np.pi * self.X)

        # set the comparison tolerance to be somewhat lenient
        self.assertAlmostEqual(solver.y, expected, tol=10**-8)

        # test with an arbitrary problem
        def rhs(t, y):
            return np.array([t**2])

        solver = ode_method(t0=0., y0=np.array(0.), rhs={'rhs': rhs}, options=options)

        solver.integrate(1.)
        expected = 1./3
        self.assertAlmostEqual(solver.y, expected, tol=10**-8)


    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
