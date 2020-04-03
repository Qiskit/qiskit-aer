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
# pylint: disable=no-name-in-module, import-error, invalid-name

"""
Controller for Monte Carlo state-vector solver method.
"""

from math import log
import time
import numpy as np
from scipy.linalg.blas import get_blas_funcs
from qiskit.tools.parallel import parallel_map, CPU_COUNT
from ..de_solvers.pulse_de_solver import construct_pulse_zvode_solver
from ..de_solvers.pulse_utils import (cy_expect_psi_csr, occ_probabilities,
                                      write_shots_memory, spmv_csr)

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


def run_monte_carlo_experiments(op_system):
    """ Runs monte carlo experiments for a given op_system

    Parameters:
        op_system (PulseSimDescription): container for simulation information

    Returns:
        tuple: two lists with experiment results

    Raises:
        Exception: if initial state is of incorrect format
    """

    if not op_system.initial_state.isket:
        raise Exception("Initial state must be a state vector.")

    # set num_cpus to the value given in settings if none in Options
    if not op_system.ode_options.num_cpus:
        op_system.ode_options.num_cpus = CPU_COUNT

    # setup seeds array
    seed = op_system.global_data.get('seed', np.random.randint(np.iinfo(np.int32).max - 1))
    prng = np.random.RandomState(seed)
    for exp in op_system.experiments:
        exp['seed'] = prng.randint(np.iinfo(np.int32).max - 1)

    map_kwargs = {'num_processes': op_system.ode_options.num_cpus}

    exp_results = []
    exp_times = []
    for exp in op_system.experiments:
        start = time.time()
        rng = np.random.RandomState(exp['seed'])
        seeds = rng.randint(np.iinfo(np.int32).max - 1,
                            size=op_system.global_data['shots'])
        exp_res = parallel_map(monte_carlo_evolution,
                               seeds,
                               task_args=(exp, op_system,),
                               **map_kwargs)

        # exp_results is a list for each shot
        # so transform back to an array of shots
        exp_res2 = []
        for exp_shot in exp_res:
            exp_res2.append(exp_shot[0].tolist())

        end = time.time()
        exp_times.append(end - start)
        exp_results.append(np.array(exp_res2))

    return exp_results, exp_times


def monte_carlo_evolution(seed, exp, op_system):
    """ Performs a single monte carlo run for the given op_system, experiment, and seed

    Parameters:
        seed (int): seed for random number generation
        exp (dict): dictionary containing experiment description
        op_system (PulseSimDescription): container for information required for simulation

    Returns:
        array: results of experiment

    Raises:
        Exception: if ODE solving has errors
    """

    global_data = op_system.global_data
    ode_options = op_system.ode_options

    rng = np.random.RandomState(seed)
    tlist = exp['tlist']
    # Init memory
    memory = np.zeros((1, global_data['memory_slots']), dtype=np.uint8)

    # Get number of acquire
    num_acq = len(exp['acquire'])
    acq_idx = 0

    collapse_times = []
    collapse_operators = []

    # first rand is collapse norm, second is which operator
    rand_vals = rng.rand(2)

    # make array for collapse operator inds
    cinds = np.arange(global_data['c_num'])
    n_dp = np.zeros(global_data['c_num'], dtype=float)

    ODE = construct_pulse_zvode_solver(exp, op_system)

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
                                                ODE._y, True)

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
