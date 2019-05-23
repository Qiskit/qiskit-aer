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

import numpy as np
from scipy.integrate import ode
from scipy.linalg.blas import get_blas_funcs
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv, spmv_csr
from openpulse.solver.zvode import qiskit_zvode

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)

def monte_carlo(pid, ophandler, ode_options):
    """
    Monte Carlo algorithm returning state-vector or expectation values
    at times tlist for a single trajectory.
    """

    global _cy_rhs_func

    tlist = ophandler._data['tlist']
    memory = [
        [
            0 for _l in range(ophandler.qobj_config['memory_slot_size'])
        ] for _r in range(ophandler.qobj_config['memory_slots'])
    ]
    register = bytearray(ophandler.backend_config['n_registers'])

    opt = ophandler._options

    collapse_times = []
    collapse_operators = []

    # SEED AND RNG AND GENERATE
    prng = RandomState(ophandler._options.seeds[pid])
    # first rand is collapse norm, second is which operator
    rand_vals = prng.rand(2)

    ODE = ode(_cy_rhs_func)

    _inst = 'ODE.set_f_params(%s)' % ophandler._data['string']
    code = compile(_inst, '<string>', 'exec')
    exec(code)
    psi = ophandler._data['psi0']

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

    if not len(ODE._y):
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)

    # make array for collapse operator inds
    cinds = np.arange(ophandler._data['c_num'])
    n_dp = np.zeros(ophandler._data['c_num'], dtype=float)

    kk_prev = 0
    # RUN ODE UNTIL EACH TIME IN TLIST
    for kk in tlist:
        ODE.set_initial_value(psi, kk_prev)
        # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.t < kk:
            t_prev = ODE.t
            y_prev = ODE.y
            norm2_prev = dznrm2(ODE._y) ** 2
            # integrate up to kk, one step at a time.
            ODE.integrate(kk, step=1)
            if not ODE.successful():
                raise Exception("ZVODE step failed!")
            norm2_psi = dznrm2(ODE._y) ** 2
            if norm2_psi <= rand_vals[0]:
                # collapse has occured:
                # find collapse time to within specified tolerance
                # ------------------------------------------------
                ii = 0
                t_final = ODE.t
                while ii < ophandler._options.norm_steps:
                    ii += 1
                    t_guess = t_prev + \
                        math.log(norm2_prev / rand_vals[0]) / \
                        math.log(norm2_prev / norm2_psi) * (t_final - t_prev)
                    ODE._y = y_prev
                    ODE.t = t_prev
                    ODE._integrator.call_args[3] = 1
                    ODE.integrate(t_guess, step=0)
                    if not ODE.successful():
                        raise Exception(
                            "ZVODE failed after adjusting step size!")
                    norm2_guess = dznrm2(ODE._y)**2
                    if (abs(rand_vals[0] - norm2_guess) <
                            ophandler._options.norm_tol * rand_vals[0]):
                        break
                    elif (norm2_guess < rand_vals[0]):
                        # t_guess is still > t_jump
                        t_final = t_guess
                        norm2_psi = norm2_guess
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        y_prev = ODE.y
                        norm2_prev = norm2_guess
                if ii > ophandler._options.norm_steps:
                    raise Exception("Norm tolerance not reached. " +
                                    "Increase accuracy of ODE solver or " +
                                    "Options.norm_steps.")

                collapse_times.append(ODE.t)
                # all constant collapse operators.
                for i in range(n_dp.shape[0]):
                    n_dp[i] = cy_expect_psi_csr(ophandler._data['n_ops_data'][i],
                                                ophandler._data['n_ops_ind'][i],
                                                ophandler._data['n_ops_ptr'][i],
                                                ODE._y, 1)

                # determine which operator does collapse and store it
                _p = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[_p >= rand_vals[1]][0]
                collapse_operators.append(j)

                state = spmv_csr(ophandler._data['c_ops_data'][j],
                                 ophandler._data['c_ops_ind'][j],
                                 ophandler._data['c_ops_ptr'][j],
                                 ODE._y)

                state /= dznrm2(state)
                ODE._y = state
                ODE._integrator.call_args[3] = 1
                rand_vals = prng.rand(2)

        # after while loop (Do measurement or conditional)
        # ----------------
        out_psi = ODE._y / dznrm2(ODE._y)

        # measurement
        psi = _proj_measurement(pid, ophandler, kk, out_psi, memory, register)

        kk_prev = kk

    return psi, memory