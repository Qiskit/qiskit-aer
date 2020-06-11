# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Options for DE_Methods"""

class DE_Options:
    """
    Container class for options and defaults specific for DE_Methods.  Options can be specified
    either as arguments to the constructor::

        opts = DE_Options(order=10, ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts.order = 10

    Notes:
        - These options are solely related to DE solving.
        - Not all options are used by all methods; see each method description for a list of
          relevant options.

    Attributes:
        method (str, 'zvode-adams'): Integration method.
        atol (float, 1e-8): Absolute tolerance.
        rtol (float, 1e-6): Relative tolerance.
        order (int, 12): Order of integrator
        nsteps (int, 50000): Max. number of internal steps per time interval.
        first_step (float, 0): Size of initial step (0 = automatic).
        min_step (float, 0): Minimum step size (0 = automatic).
        max_step (float, 0): Maximum step size (0 = automatic).
        max_dt (float, 1e-3):
    """

    def __init__(self,
                 method='zvode-adams',
                 atol=1e-8,
                 rtol=1e-6,
                 order=12,
                 nsteps=50000,
                 first_step=None,
                 max_step=None,
                 min_step=None,
                 max_dt=10**-3):

        # Integration method (default = 'zvode-adams')
        self.method = method
        # Absolute tolerance (default = 1e-8)
        self.atol = atol
        # Relative tolerance (default = 1e-6)
        self.rtol = rtol
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order = order
        # Max. number of internal steps/call
        self.nsteps = nsteps
        # Size of initial step (0 = determined by solver)
        self.first_step = first_step
        # Max step size (0 = determined by solver)
        self.max_step = max_step
        # Minimal step size (0 = determined by solver)
        self.min_step = min_step
        # max step size for fixed step-size solvers
        self.max_dt = max_dt

    def copy(self):
        return DE_Options(method=self.method,
                          atol=self.atol,
                          rtol=self.rtol,
                          order=self.order,
                          nsteps=self.nsteps,
                          first_step=self.first_step,
                          max_step=self.max_step,
                          min_step=self.min_step,
                          max_dt=self.max_dt)

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return self.__str__()
