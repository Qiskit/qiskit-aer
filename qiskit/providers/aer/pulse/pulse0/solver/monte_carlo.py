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

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
# pylint: disable=no-name-in-module, import-error, unused-variable, invalid-name

"""Monte carlo wave function solver."""

from math import log
import logging
import numpy as np
from scipy.integrate import ode
from scipy.linalg.blas import get_blas_funcs

from qiskit.providers.aer.pulse.solver.zvode import qiskit_zvode
from qiskit.providers.aer.pulse.cy.measure import occ_probabilities, write_shots_memory
from ..qutip_lite.cy.spmatfuncs import cy_expect_psi_csr, spmv_csr

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


def monte_carlo(seed, exp, op_system):
    """
    Monte Carlo algorithm returning state-vector or expectation values
    at times tlist for a single trajectory.
    """

    global_data = op_system.global_data
    ode_options = op_system.ode_options

    cy_rhs_func = global_data['rhs_func']
    rng = np.random.RandomState(seed)
    tlist = exp['tlist']
    snapshots = []
    # Init memory
    memory = np.zeros((1, global_data['memory_slots']), dtype=np.uint8)
    # Init register
    register = np.zeros(global_data['n_registers'], dtype=np.uint8)

    # Get number of acquire, snapshots, and conditionals
    num_acq = len(exp['acquire'])
    acq_idx = 0
    num_snap = len(exp['snapshot'])
    snap_idx = 0
    num_cond = len(exp['cond'])
    cond_idx = 0

    collapse_times = []
    collapse_operators = []

    # first rand is collapse norm, second is which operator
    rand_vals = rng.rand(2)

    ODE = ode(cy_rhs_func)

    if op_system.use_cpp_ode_func:
        # Don't know how to use OrderedDict type on Cython, so transforming it to dict
        channels = dict(op_system.channels)
        ODE.set_f_params(global_data, exp, op_system.system, channels, register)
    else:
        _inst = 'ODE.set_f_params(%s)' % global_data['string']
        logging.debug("Monte Carlo: %s\n\n", _inst)
        code = compile(_inst, '<string>', 'exec')
        # pylint: disable=exec-used
        exec(code)

    # initialize ODE solver for RHS
    ODE._integrator = qiskit_zvode(method=ode_options.method,
                                   order=ode_options.order,
                                   atol=ode_options.atol,
                                   rtol=ode_options.rtol,
                                   nsteps=ode_options.nsteps,
                                   first_step=ode_options.first_step,
                                   min_step=ode_options.min_step,
                                   max_step=ode_options.max_step
                                   )
    # Forces complex ODE solving
    if not any(ODE._y):
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)

    ODE.set_initial_value(global_data['initial_state'], 0)

    # make array for collapse operator inds
    cinds = np.arange(global_data['c_num'])
    n_dp = np.zeros(global_data['c_num'], dtype=float)

    # RUN ODE UNTIL EACH TIME IN TLIST
    for stop_time in tlist:
        # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.t < stop_time:
            t_prev = ODE.t
            y_prev = ODE.y
            norm2_prev = dznrm2(ODE._y) ** 2
            # integrate up to stop_time, one step at a time.
            ODE.integrate(stop_time, step=1)
            if not ODE.successful():
                raise Exception("ZVODE step failed!")
            norm2_psi = dznrm2(ODE._y) ** 2
            if norm2_psi <= rand_vals[0]:
                # collapse has occured:
                # find collapse time to within specified tolerance
                # ------------------------------------------------
                ii = 0
                t_final = ODE.t
                while ii < ode_options.norm_steps:
                    ii += 1
                    t_guess = t_prev + \
                        log(norm2_prev / rand_vals[0]) / \
                        log(norm2_prev / norm2_psi) * (t_final - t_prev)
                    ODE._y = y_prev
                    ODE.t = t_prev
                    ODE._integrator.call_args[3] = 1
                    ODE.integrate(t_guess, step=0)
                    if not ODE.successful():
                        raise Exception(
                            "ZVODE failed after adjusting step size!")
                    norm2_guess = dznrm2(ODE._y)**2
                    if (abs(rand_vals[0] - norm2_guess) <
                            ode_options.norm_tol * rand_vals[0]):
                        break

                    if norm2_guess < rand_vals[0]:
                        # t_guess is still > t_jump
                        t_final = t_guess
                        norm2_psi = norm2_guess
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        y_prev = ODE.y
                        norm2_prev = norm2_guess
                if ii > ode_options.norm_steps:
                    raise Exception("Norm tolerance not reached. " +
                                    "Increase accuracy of ODE solver or " +
                                    "Options.norm_steps.")

                collapse_times.append(ODE.t)
                # all constant collapse operators.
                for i in range(n_dp.shape[0]):
                    n_dp[i] = cy_expect_psi_csr(global_data['n_ops_data'][i],
                                                global_data['n_ops_ind'][i],
                                                global_data['n_ops_ptr'][i],
                                                ODE._y, 1)

                # determine which operator does collapse and store it
                _p = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[_p >= rand_vals[1]][0]
                collapse_operators.append(j)

                state = spmv_csr(global_data['c_ops_data'][j],
                                 global_data['c_ops_ind'][j],
                                 global_data['c_ops_ptr'][j],
                                 ODE._y)

                state /= dznrm2(state)
                ODE._y = state
                ODE._integrator.call_args[3] = 1
                rand_vals = rng.rand(2)

        # after while loop (Do measurement or conditional)
        # ------------------------------------------------
        out_psi = ODE._y / dznrm2(ODE._y)
        for aind in range(acq_idx, num_acq):
            if exp['acquire'][aind][0] == stop_time:
                current_acq = exp['acquire'][aind]
                qubits = current_acq[1]
                memory_slots = current_acq[2]
                probs = occ_probabilities(qubits, out_psi, global_data['measurement_ops'])
                rand_vals = rng.rand(memory_slots.shape[0])
                write_shots_memory(memory, memory_slots, probs, rand_vals)
                acq_idx += 1

    return memory
