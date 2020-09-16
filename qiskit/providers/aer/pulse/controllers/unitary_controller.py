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
Controller for solving unitary evolution of a state-vector.
"""

import time
import numpy as np
from scipy.linalg.blas import get_blas_funcs
from qiskit.tools.parallel import parallel_map, CPU_COUNT
from .pulse_sim_options import PulseSimOptions
from .pulse_de_solver import setup_de_solver

# Imports from qutip_extra_lite
from .pulse_utils import occ_probabilities, write_shots_memory

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


def _full_simulation(exp, y0, pulse_sim_desc, pulse_de_model, solver_options=None):
    """
    Set up full simulation, i.e. combining different (ideally modular) computational
    resources into one function.
    """

    solver_options = PulseSimOptions() if solver_options is None else solver_options

    psi, ode_t = unitary_evolution(exp, y0, pulse_de_model, solver_options)

    # ###############
    # do measurement
    # ###############
    rng = np.random.RandomState(exp['seed'])

    shots = pulse_sim_desc.shots
    # Init memory
    memory = np.zeros((shots, pulse_sim_desc.memory_slots), dtype=np.uint8)

    qubits = []
    memory_slots = []
    tlist = exp['tlist']
    for acq in exp['acquire']:
        if acq[0] == tlist[-1]:
            qubits += list(acq[1])
            memory_slots += list(acq[2])
    qubits = np.array(qubits, dtype='uint32')
    memory_slots = np.array(memory_slots, dtype='uint32')

    probs = occ_probabilities(qubits, psi, pulse_sim_desc.measurement_ops)
    rand_vals = rng.rand(memory_slots.shape[0] * shots)
    write_shots_memory(memory, memory_slots, probs, rand_vals)

    return [memory, psi, ode_t]


def run_unitary_experiments(pulse_sim_desc, pulse_de_model, solver_options=None):
    """ Runs unitary experiments for a given op_system

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

    # run simulation on each experiment in parallel
    start = time.time()
    exp_results = parallel_map(_full_simulation,
                               pulse_sim_desc.experiments,
                               task_args=(y0, pulse_sim_desc, pulse_de_model, solver_options, ),
                               **map_kwargs
                               )
    end = time.time()
    exp_times = (np.ones(len(pulse_sim_desc.experiments)) *
                 (end - start) / len(pulse_sim_desc.experiments))

    return exp_results, exp_times


def unitary_evolution(exp, y0, pulse_de_model, solver_options=None):
    """
    Calculates evolution when there is no noise, or any measurements that are not at the end
    of the experiment.

    Parameters:
        exp (dict): dictionary containing experiment description
        y0 (array): initial state
        pulse_de_model (PulseInternalDEModel): container for de model
        solver_options (PulseSimOptions): options

    Returns:
        array: results of experiment

    Raises:
        Exception: if ODE solving has errors
    """

    solver_options = PulseSimOptions() if solver_options is None else solver_options

    ODE = setup_de_solver(exp, y0, pulse_de_model, solver_options.de_options)

    tlist = exp['tlist']

    for t in tlist[1:]:
        ODE.integrate(t)
        if ODE.successful():
            psi = ODE.y / dznrm2(ODE.y)
        else:
            err_msg = 'ODE method exited with status: %s' % ODE.return_code()
            raise Exception(err_msg)

    # apply final rotation to come out of rotating frame
    psi_rot = np.exp(-1j * pulse_de_model.h_diag_elems * ODE.t)
    psi *= psi_rot

    return psi, ODE.t
