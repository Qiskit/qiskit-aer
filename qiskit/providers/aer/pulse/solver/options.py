# -*- coding: utf-8 -*-

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

"""OpenPulse options"""


class OPoptions():
    """
    Class of options for opsolver.  Options can be specified either as
    arguments to the constructor::

        opts = Options(order=10, ...)

    or by changing the class attributes after creation::

        opts = Options()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    Attributes:
        atol (float, 1e-8): Absolute tolerance.
        rtol (float, 1e-/6): Relative tolerance.
        method (str, 'adams'): Integration method, 'adams' or 'bdf'.
        order (int, 12): Order of integrator (<=12 'adams', <=5 'bdf').
        nsteps (int, 50000): Max. number of internal steps per time interval.
        first_step (float, 0): Size of initial step (0 = automatic).
        min_step (float, 0): Minimum step size (0 = automatic).
        max_step (float, 0): Maximum step size (0 = automatic)
        num_cpus (int): Number of cpus used by mcsolver (default = # of cpus).
        norm_tol (float, 1e-3): Tolerance used when finding wavefunction norm.
        norm_steps (int, 5): Max. number of steps used to find wavefunction norm
                            to within norm_tol
        shots (int, 1024): Number of shots to run.
        rhs_reuse (bool, False): Reuse RHS compiled function.
        rhs_filename (str): Name of compiled Cython module.
        seeds (ndarray, None): Array containing random number seeds for
                                repeatible shots.
        reuse_seeds (bool, False): Reuse seeds, if already generated.
        store_final_state (bool, False): Whether or not to store the final state
                                        of the evolution.
    """

    def __init__(self, atol=1e-8, rtol=1e-6, method='adams', order=12,
                 nsteps=50000, first_step=0, max_step=0, min_step=0,
                 num_cpus=0, norm_tol=1e-3, norm_steps=5,
                 progress_bar=True, rhs_reuse=False,
                 rhs_filename=None, shots=1024,
                 store_final_state=False, seeds=None,
                 reuse_seeds=False):

        # Absolute tolerance (default = 1e-8)
        self.atol = atol
        # Relative tolerance (default = 1e-6)
        self.rtol = rtol
        # Integration method (default = 'adams', for stiff 'bdf')
        self.method = method
        # Max. number of internal steps/call
        self.nsteps = nsteps
        # Size of initial step (0 = determined by solver)
        self.first_step = first_step
        # Minimal step size (0 = determined by solver)
        self.min_step = min_step
        # Max step size (0 = determined by solver)
        self.max_step = max_step
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        self.order = order
        # Number of shots to run (default=500)
        self.shots = shots
        # Holds seeds for rand num gen
        self.seeds = seeds
        # reuse seeds
        self.reuse_seeds = reuse_seeds
        # Use preexisting RHS function for time-dependent solvers
        self.rhs_reuse = rhs_reuse
        # Track progress
        self.progress_bar = progress_bar
        # Use filename for preexisting RHS function (will default to last
        # compiled function if None & rhs_exists=True)
        self.rhs_filename = rhs_filename
        # Number of processors to use
        if num_cpus:
            self.num_cpus = num_cpus
        else:
            self.num_cpus = 0
        # Tolerance for wavefunction norm (mcsolve only)
        self.norm_tol = norm_tol
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        self.norm_steps = norm_steps
        # store final state?
        self.store_final_state = store_final_state

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return self.__str__()
