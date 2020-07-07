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

"""Pulse solver options"""

from ..de.DE_Options import DE_Options


class PulseSimOptions():
    """
    Class of options for pulse solver routines.  Options can be specified either as
    arguments to the constructor::

        opts = Options(order=10, ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

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
        num_cpus (int): Number of cpus used by mcsolver (default = # of cpus).
        norm_tol (float, 1e-3): Tolerance used when finding wavefunction norm.
        norm_steps (int, 5): Max. number of steps used to find wavefunction norm
                            to within norm_tol
        shots (int, 1024): Number of shots to run.
        seeds (ndarray, None): Array containing random number seeds for
                                repeatible shots.
        reuse_seeds (bool, False): Reuse seeds, if already generated.
        store_final_state (bool, False): Whether or not to store the final state
                                        of the evolution.
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
                 max_dt=10**-3,
                 num_cpus=0,
                 norm_tol=1e-3,
                 norm_steps=5,
                 progress_bar=True,
                 shots=1024,
                 store_final_state=False,
                 seeds=None,
                 reuse_seeds=False):

        # set DE specific options
        self.de_options = DE_Options(method=method,
                                     atol=atol,
                                     rtol=rtol,
                                     order=order,
                                     nsteps=nsteps,
                                     first_step=first_step,
                                     max_step=max_step,
                                     min_step=min_step,
                                     max_dt=max_dt)

        self.shots = shots
        self.seeds = seeds
        self.reuse_seeds = reuse_seeds
        self.progress_bar = progress_bar
        self.num_cpus = num_cpus
        self.norm_tol = norm_tol
        self.norm_steps = norm_steps
        self.store_final_state = store_final_state

    def copy(self):
        """Create a copy."""
        return PulseSimOptions(method=self.de_options.method,
                               atol=self.de_options.atol,
                               rtol=self.de_options.rtol,
                               order=self.de_options.order,
                               nsteps=self.de_options.nsteps,
                               first_step=self.de_options.first_step,
                               max_step=self.de_options.max_step,
                               min_step=self.de_options.min_step,
                               max_dt=self.de_options.max_dt,
                               num_cpus=self.num_cpus,
                               norm_tol=self.norm_tol,
                               norm_steps=self.norm_steps,
                               progress_bar=self.progress_bar,
                               shots=self.shots,
                               store_final_state=self.store_final_state,
                               seeds=self.seeds,
                               reuse_seeds=self.reuse_seeds)

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return self.__str__()
