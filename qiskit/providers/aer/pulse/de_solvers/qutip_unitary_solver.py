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
# pylint: disable=no-name-in-module, import-error, invalid-name

"""
Solver from qutip.
"""

import time
import logging
import numpy as np
from scipy.integrate import ode
from scipy.linalg.blas import get_blas_funcs
from qiskit.tools.parallel import parallel_map, CPU_COUNT
from ..pulse0.qutip_lite.cy.spmatfuncs import cy_expect_psi_csr
from ..pulse0.qutip_lite.cy.utilities import _cython_build_cleanup
from ..pulse0.qobj.operators import apply_projector
from ..pulse0.solver.rhs_utils import _op_generate_rhs, _op_func_load
from ..pulse0.solver.data_config import op_data_config
from ..pulse0.cy.measure import occ_probabilities, write_shots_memory

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)

#
# Internal, global variables for storing references to dynamically loaded
# cython functions
#
_cy_rhs_func = None


def qutip_unitary_solver(op_system):
    """ unitary solver
    """

    if not op_system.initial_state.isket:
        raise Exception("Initial state must be a state vector.")

    # set num_cpus to the value given in settings if none in Options
    if not op_system.ode_options.num_cpus:
        op_system.ode_options.num_cpus = CPU_COUNT

    # build Hamiltonian data structures
    op_data_config(op_system)
    if not op_system.use_cpp_ode_func:
        # compile Cython RHS
        _op_generate_rhs(op_system)
    # Load cython function
    _op_func_load(op_system)

    results = run_unitary_experiments(op_system)
    # Results are stored in ophandler.result
    return results

def run_unitary_experiments(op_system):


    """unitary evolution requires no seeds, so move this out of this deterministic DE
    solving class once measurements are moved
    """
    # setup seeds array
    if op_system.global_data['seed']:
        prng = np.random.RandomState(op_system.global_data['seed'])
    else:
        prng = np.random.RandomState(
            np.random.randint(np.iinfo(np.int32).max - 1))
    for exp in op_system.experiments:
        exp['seed'] = prng.randint(np.iinfo(np.int32).max - 1)


    map_kwargs = {'num_processes': op_system.ode_options.num_cpus}


    start = time.time()
    exp_results = parallel_map(unitary_evolution,
                               op_system.experiments,
                               task_args=(op_system,),
                               **map_kwargs
                               )
    end = time.time()
    exp_times = (np.ones(len(op_system.experiments)) *
                 (end - start) / len(op_system.experiments))


    # format the data into the proper output
    all_results = []
    for idx_exp, exp in enumerate(op_system.experiments):

        m_lev = op_system.global_data['meas_level']
        m_ret = op_system.global_data['meas_return']

        # populate the results dictionary
        results = {'seed_simulator': exp['seed'],
                   'shots': op_system.global_data['shots'],
                   'status': 'DONE',
                   'success': True,
                   'time_taken': exp_times[idx_exp],
                   'header': exp['header'],
                   'meas_level': m_lev,
                   'meas_return': m_ret,
                   'data': {}}

        memory = exp_results[idx_exp][0]
        results['data']['statevector'] = []
        for coef in exp_results[idx_exp][1]:
            results['data']['statevector'].append([np.real(coef),
                                                   np.imag(coef)])
        results['header']['ode_t'] = exp_results[idx_exp][2]

        # meas_level 2 return the shots
        if m_lev == 2:

            # convert the memory **array** into a n
            # integer
            # e.g. [1,0] -> 2
            int_mem = memory.dot(np.power(2.0,
                                          np.arange(memory.shape[1]))).astype(int)

            # if the memory flag is set return each shot
            if op_system.global_data['memory']:
                hex_mem = [hex(val) for val in int_mem]
                results['data']['memory'] = hex_mem

            # Get hex counts dict
            unique = np.unique(int_mem, return_counts=True)
            hex_dict = {}
            for kk in range(unique[0].shape[0]):
                key = hex(unique[0][kk])
                hex_dict[key] = unique[1][kk]

            results['data']['counts'] = hex_dict

        # meas_level 1 returns the <n>
        elif m_lev == 1:

            if m_ret == 'avg':

                memory = [np.mean(memory, 0)]

            # convert into the right [real, complex] pair form for json
            # this should be cython?
            results['data']['memory'] = []

            for mem_shot in memory:
                results['data']['memory'].append([])
                for mem_slot in mem_shot:
                    results['data']['memory'][-1].append(
                        [np.real(mem_slot), np.imag(mem_slot)])

            if m_ret == 'avg':
                results['data']['memory'] = results['data']['memory'][0]

        all_results.append(results)

    if not op_system.use_cpp_ode_func:
        _cython_build_cleanup(op_system.global_data['rhs_file_name'])

    return all_results


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

    global_data = op_system.global_data
    ode_options = op_system.ode_options

    cy_rhs_func = global_data['rhs_func']
    rng = np.random.RandomState(exp['seed'])
    tlist = exp['tlist']
    snapshots = []
    shots = global_data['shots']
    # Init memory
    memory = np.zeros((shots, global_data['memory_slots']),
                      dtype=np.uint8)
    # Init register
    register = np.zeros(global_data['n_registers'], dtype=np.uint8)

    num_channels = len(exp['channels'])

    ODE = ode(cy_rhs_func)
    if op_system.use_cpp_ode_func:
        # Don't know how to use OrderedDict type on Cython, so transforming it to dict
        channels = dict(op_system.channels)
        ODE.set_f_params(global_data, exp, op_system.system, channels, register)
    else:
        _inst = 'ODE.set_f_params(%s)' % global_data['string']
        logging.debug("Unitary Evolution: %s\n\n", _inst)
        code = compile(_inst, '<string>', 'exec')
        exec(code)  # pylint disable=exec-used

    ODE.set_integrator('zvode',
                       method=ode_options.method,
                       order=ode_options.order,
                       atol=ode_options.atol,
                       rtol=ode_options.rtol,
                       nsteps=ode_options.nsteps,
                       first_step=ode_options.first_step,
                       min_step=ode_options.min_step,
                       max_step=ode_options.max_step)

    if not ODE._y:
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)

    # Since all experiments are defined to start at zero time.
    ODE.set_initial_value(global_data['initial_state'], 0)
    for time in tlist[1:]:
        ODE.integrate(time, step=0)
        if ODE.successful():
            psi = ODE.y / dznrm2(ODE.y)
        else:
            err_msg = 'ZVODE exited with status: %s' % ODE.get_return_code()
            raise Exception(err_msg)


    # Do final measurement at end, only take acquire channels at the end
    psi_rot = np.exp(-1j * global_data['h_diag_elems'] * ODE.t)
    psi *= psi_rot
    qubits = []
    memory_slots = []
    for acq in exp['acquire']:
        if acq[0] == tlist[-1]:
            qubits += list(acq[1])
            memory_slots += list(acq[2])
    qubits = np.array(qubits, dtype='uint32')
    memory_slots = np.array(memory_slots, dtype='uint32')

    probs = occ_probabilities(qubits, psi, global_data['measurement_ops'])
    rand_vals = rng.rand(memory_slots.shape[0] * shots)
    write_shots_memory(memory, memory_slots, probs, rand_vals)
    return [memory, psi, ODE.t]
