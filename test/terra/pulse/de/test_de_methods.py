"""tests for DE_Methods.py"""

import unittest
import numpy as np
from scipy.linalg import expm

import sys
sys.path.insert(1, '../../de/')
from DE_Methods import ODE_Method, BMDE_Method, RK4, Expm, ScipyODE, method_from_string

class TestDE_Methods(unittest.TestCase):

    def setUp(self):
        # set up a 2d test problem
        self.t0 = 0.
        self.y0 = np.eye(2)

        self.X = np.array([[0., 1.], [1., 0.]], dtype=complex)
        self.Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
        self.Z = np.array([[1., 0.], [0., -1.]], dtype=complex)

        def generator(t):
            return -1j * 2 * np.pi * self.X / 2

        def rhs(t, y):
            return generator(t) @ y

        self.rhs = {'rhs': rhs, 'generator': generator}

    def test_method_from_string(self):
        """Test method_from_string"""

        # test for scipy wrapper
        method, options = method_from_string('scipy-RK45')
        self.assertTrue(method == ScipyODE)
        self.assertTrue(options == {'method': 'RK45'})

        # test for Expm
        method, options = method_from_string('Expm')
        self.assertTrue(method == Expm)
        self.assertTrue(options == {})

    def test_ScipyODE(self):
        """Test ScipyODE method"""

        # test with the default problem
        scipy_options = {'atol': 10**-10, 'rtol': 10**-10}
        solver_options = {'method': 'RK45', 'scipy_options': scipy_options}

        scipy_solver = ScipyODE(t0=self.t0, y0=self.y0, rhs=self.rhs, solver_options=solver_options)
        scipy_solver.integrate(1.)

        expected = expm(-1j * np.pi * self.X)
        self.assertAlmostEqual(scipy_solver.y, expected, tol=10**-9)

        # test with an arbitrary problem
        def rhs(t, y):
            return np.array([t**2])

        scipy_solver = ScipyODE(t0=0., y0=np.array(0.), rhs={'rhs': rhs}, solver_options=solver_options)

        scipy_solver.integrate(1.)
        expected = 1./3
        self.assertAlmostEqual(scipy_solver.y, expected, tol=10**-9)

    def test_Expm(self):
        """Test Expm method"""

        # test with the default problem
        solver_options = {'max_dt': 0.01}

        expm_solver = Expm(t0=self.t0, y0=self.y0, rhs=self.rhs, solver_options=solver_options)
        expm_solver.integrate(1.)

        expected = expm(-1j * np.pi * self.X)
        self.assertAlmostEqual(expm_solver.y, expected, tol=10**-9)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
