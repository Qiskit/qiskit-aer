"""tests for ODEs cpp wrappers"""

import unittest
import numpy as np
from qiskit.providers.aer.pulse.de_solvers.pulse_utils import (create_sundials_integrator,
                                                               create_sundials_sens_integrator)

class TestODEsCppWrappers(unittest.TestCase):

    def setUp(self):
        self.y = [3.5]
        self.par = [5]

        def f(t, y):
            return y * self.par[0]

        self.f = f

    def test_sundials_1D(self):
        """Test simple one dimensional ODE"""
        odes = create_sundials_integrator(0.0, self.y, self.f)
        odes.integrate(1.)
        self.assertAlmostEqual(odes.y[0], self.y[0] * np.exp(self.par[0] * 1.))

    def test_sundials_sens_1D(self):
        """Test simple one dimensional ODE with sens"""
        def pf(p):
            self.par[0] = p[0]

        odes = create_sundials_sens_integrator(0.0, self.y, self.f, pf, self.par)
        odes.set_tolerances(1e-9, 1e-9)
        #odes.set_max_nsteps(50000)
        odes.integrate(1.)
        self.assertAlmostEqual(odes.y[0], self.y[0] * np.exp(self.par[0] * 1.), 5)
        self.assertAlmostEqual(odes.get_sens(0)[0], self.y[0] * np.exp(self.par[0] * 1.), 5)