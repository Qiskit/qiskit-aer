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
from ..de_solvers.pulse_de_solver import construct_pulse_zvode_solver

# Imports from qutip_extra_lite
from ..de_solvers.pulse_utils import occ_probabilities, write_shots_memory

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


def _full_simulation(exp, op_system):
    """
    Set up full simulation, i.e. combining different (ideally modular) computational
    resources into one function.
    """
    psi, ode_t = unitary_evolution(exp, op_system)

    # ###############
    # do measurement
    # ###############
    rng = np.random.RandomState(exp['seed'])

    shots = op_system.global_data['shots']
    # Init memory
    memory = np.zeros((shots, op_system.global_data['memory_slots']),
                      dtype=np.uint8)

    qubits = []
    memory_slots = []
    tlist = exp['tlist']
    for acq in exp['acquire']:
        if acq[0] == tlist[-1]:
            qubits += list(acq[1])
            memory_slots += list(acq[2])
    qubits = np.array(qubits, dtype='uint32')
    memory_slots = np.array(memory_slots, dtype='uint32')

    probs = occ_probabilities(qubits, psi, op_system.global_data['measurement_ops'])
    rand_vals = rng.rand(memory_slots.shape[0] * shots)
    write_shots_memory(memory, memory_slots, probs, rand_vals)

    return [memory, psi, ode_t]


def run_unitary_experiments(op_system):
    """ Runs unitary experiments for a given op_system

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

    # run simulation on each experiment in parallel
    start = time.time()
    exp_results = parallel_map(_full_simulation,
                               op_system.experiments,
                               task_args=(op_system, ),
                               **map_kwargs
                               )
    end = time.time()
    exp_times = (np.ones(len(op_system.experiments)) *
                 (end - start) / len(op_system.experiments))

    return exp_results, exp_times


def unitary_evolution(exp, op_system):
    """
    Calculates evolution when there is no noise,
    or any measurements that are not at the end
    of the experiment.

    Args:
        exp (dict): Dictionary of experimental pulse and fc
        op_system (OPSystem): Global OpenPulse system settings

    Returns:
        array: Memory of shots.

    Raises:
        Exception: Error in ODE solver.
    """

    ODE = construct_pulse_zvode_solver(exp, op_system)

    tlist = exp['tlist']

    for t in tlist[1:]:
        ODE.integrate(t, step=0)
        if ODE.successful():
            psi = ODE.y / dznrm2(ODE.y)
        else:
            err_msg = 'ZVODE exited with status: %s' % ODE.get_return_code()
            raise Exception(err_msg)

    # apply final rotation to come out of rotating frame
    psi_rot = np.exp(-1j * op_system.global_data['h_diag_elems'] * ODE.t)
    psi *= psi_rot

    return psi, ODE.t
