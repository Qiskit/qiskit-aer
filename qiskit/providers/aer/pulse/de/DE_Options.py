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
# pylint: disable=invalid-name

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
        - Not all options are used by all methods; see each method description for a list of
          relevant options.

    Attributes:
        method (str, 'zvode-adams'): Integration method.
        atol (float, 1e-8): Absolute tolerance for variable step solvers.
        rtol (float, 1e-6): Relative tolerance for variable step solvers.
        order (int, 12): Order of integrator.
        nsteps (int, 10**6): Max. number of internal steps per time interval for variable step
                             solvers.
        first_step (float, None): Size of initial step for variable step solvers.
        min_step (float, None): Minimum step size for variable step solvers.
        max_step (float, None): Maximum step size for variable step solvers.
        max_dt (float, 1e-3): Max step size for fixed step solver.
    """

    def __init__(self,
                 method='zvode-adams',
                 atol=1e-8,
                 rtol=1e-6,
                 order=12,
                 nsteps=10**6,
                 first_step=None,
                 max_step=None,
                 min_step=None,
                 max_dt=10**-3):

        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.order = order
        self.nsteps = nsteps
        self.first_step = first_step
        self.max_step = max_step
        self.min_step = min_step
        self.max_dt = max_dt

    def copy(self):
        """Create a copy of the object."""
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
