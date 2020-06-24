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
from .pulse_sim_options import PulseSimOptions
from .pulse_de_solver import setup_de_solver
from .pulse_utils import (cy_expect_psi_csr, occ_probabilities, write_shots_memory, spmv_csr)

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


def run_monte_carlo_experiments(pulse_sim_desc, pulse_de_model, solver_options=None):
    """ Runs monte carlo experiments for a given op_system

    Parameters:
        pulse_sim_desc (PulseSimDescription): description of pulse simulation
        pulse_de_model (PulseInternalDEModel): description of de model
        solver_options (PulseSimOptions): options

    Returns:
        tuple: two lists with experiment results

    Raises:
        Exception: if initial state is of incorrect format
    """

    solver_options = PulseSimOptions() if solver_options is None else solver_options

    if not pulse_sim_desc.initial_state.isket:
        raise Exception("Initial state must be a state vector.")

    y0 = pulse_sim_desc.initial_state.full().ravel()

    # set num_cpus to the value given in settings if none in Options
    if not solver_options.num_cpus:
        solver_options.num_cpus = CPU_COUNT

    # setup seeds array
    seed = pulse_sim_desc.seed or np.random.randint(np.iinfo(np.int32).max - 1)
    prng = np.random.RandomState(seed)
    for exp in pulse_sim_desc.experiments:
        exp['seed'] = prng.randint(np.iinfo(np.int32).max - 1)

    map_kwargs = {'num_processes': solver_options.num_cpus}

    exp_results = []
    exp_times = []

    # needs to be configured ahead of time
    pulse_de_model._config_internal_data()

    for exp in pulse_sim_desc.experiments:
        start = time.time()
        rng = np.random.RandomState(exp['seed'])
        seeds = rng.randint(np.iinfo(np.int32).max - 1, size=pulse_sim_desc.shots)
        exp_res = parallel_map(monte_carlo_evolution,
                               seeds,
                               task_args=(exp,
                                          y0,
                                          pulse_sim_desc,
                                          pulse_de_model,
                                          solver_options, ),
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


def monte_carlo_evolution(seed,
                          exp,
                          y0,
                          pulse_sim_desc,
                          pulse_de_model,
                          solver_options=None):
    """ Performs a single monte carlo run for the given op_system, experiment, and seed

    Parameters:
        seed (int): seed for random number generation
        exp (dict): dictionary containing experiment description
        y0 (array): initial state
        pulse_sim_desc (PulseSimDescription): container for simulation description
        pulse_de_model (PulseInternalDEModel): container for de model
        solver_options (PulseSimOptions): options

    Returns:
        array: results of experiment

    Raises:
        Exception: if ODE solving has errors
    """

    solver_options = PulseSimOptions() if solver_options is None else solver_options

    rng = np.random.RandomState(seed)
    tlist = exp['tlist']
    # Init memory
    memory = np.zeros((1, pulse_sim_desc.memory_slots), dtype=np.uint8)

    # Get number of acquire
    num_acq = len(exp['acquire'])
    acq_idx = 0

    collapse_times = []
    collapse_operators = []

    # first rand is collapse norm, second is which operator
    rand_vals = rng.rand(2)

    # make array for collapse operator inds
    cinds = np.arange(pulse_de_model.c_num)
    n_dp = np.zeros(pulse_de_model.c_num, dtype=float)

    ODE = setup_de_solver(exp, y0, pulse_de_model, solver_options.de_options)

    # RUN ODE UNTIL EACH TIME IN TLIST
    for stop_time in tlist:
        # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.t < stop_time:
            t_prev = ODE.t
            y_prev = ODE.y
            norm2_prev = dznrm2(ODE.y) ** 2
            # integrate up to stop_time, one step at a time.
            ODE.integrate(stop_time, step=1)
            if not ODE.successful():
                raise Exception("Integration step failed!")
            norm2_psi = dznrm2(ODE.y) ** 2
            if norm2_psi <= rand_vals[0]:
                # collapse has occured:
                # find collapse time to within specified tolerance
                # ------------------------------------------------
                ii = 0
                t_final = ODE.t
                while ii < solver_options.norm_steps:
                    ii += 1
                    t_guess = t_prev + \
                        log(norm2_prev / rand_vals[0]) / \
                        log(norm2_prev / norm2_psi) * (t_final - t_prev)
                    ODE.y = y_prev
                    ODE.t = t_prev
                    ODE.integrate(t_guess, step=0)
                    if not ODE.successful():
                        raise Exception(
                            "Integration failed after adjusting step size!")
                    norm2_guess = dznrm2(ODE.y)**2
                    if (abs(rand_vals[0] - norm2_guess) <
                            solver_options.norm_tol * rand_vals[0]):
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
                if ii > solver_options.norm_steps:
                    raise Exception("Norm tolerance not reached. " +
                                    "Increase accuracy of ODE solver or " +
                                    "Options.norm_steps.")

                collapse_times.append(ODE.t)
                # all constant collapse operators.
                for i in range(n_dp.shape[0]):
                    n_dp[i] = cy_expect_psi_csr(pulse_de_model.n_ops_data[i],
                                                pulse_de_model.n_ops_ind[i],
                                                pulse_de_model.n_ops_ptr[i],
                                                ODE.y, True)

                # determine which operator does collapse and store it
                _p = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[_p >= rand_vals[1]][0]
                collapse_operators.append(j)

                state = spmv_csr(pulse_de_model.c_ops_data[j],
                                 pulse_de_model.c_ops_ind[j],
                                 pulse_de_model.c_ops_ptr[j],
                                 ODE.y)

                state /= dznrm2(state)
                ODE.y = state
                rand_vals = rng.rand(2)

        # after while loop (Do measurement or conditional)
        # ------------------------------------------------
        out_psi = ODE.y / dznrm2(ODE.y)
        for aind in range(acq_idx, num_acq):
            if exp['acquire'][aind][0] == stop_time:
                current_acq = exp['acquire'][aind]
                qubits = current_acq[1]
                memory_slots = current_acq[2]
                probs = occ_probabilities(qubits, out_psi, pulse_sim_desc.measurement_ops)
                rand_vals = rng.rand(memory_slots.shape[0])
                write_shots_memory(memory, memory_slots, probs, rand_vals)
                acq_idx += 1

    return memory
