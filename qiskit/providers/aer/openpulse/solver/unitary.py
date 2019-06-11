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
# pylint: disable = unused-variable, no-name-in-module

import numpy as np
from scipy.integrate import ode
from scipy.linalg.blas import get_blas_funcs
from openpulse.cython.memory import write_memory
from openpulse.cython.measure import occ_probabilities, write_shots_memory

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)

def unitary_evolution(exp, global_data, ode_options):
    """
    Calculates evolution when there is no noise,
    or any measurements that are not at the end
    of the experiment.

    Args:
        exp (dict): Dictionary of experimental pulse and fc
            data.
        global_data (dict): Data that applies to all experiments.
        ode_options (OPoptions): Options for the underlying ODE solver.

    Returns:
        Stuff
    """
    cy_rhs_func  = global_data['rhs_func']
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
    ODE.set_integrator('zvode',
                       method=ode_options.method,
                       order=ode_options.order,
                       atol=ode_options.atol,
                       rtol=ode_options.rtol,
                       nsteps=ode_options.nsteps,
                       first_step=ode_options.first_step,
                       min_step=ode_options.min_step,
                       max_step=ode_options.max_step)

    _inst = 'ODE.set_f_params(%s)' % global_data['string']
    code = compile(_inst, '<string>', 'exec')
    exec(code)

    if not len(ODE._y):
        ODE.t = 0.0
        ODE._y = np.array([0.0], complex)
    ODE._integrator.reset(len(ODE._y), ODE.jac is not None)

    # Since all experiments are defined to start at zero time.
    ODE.set_initial_value(global_data['initial_state'], 0) 
    for kk in tlist[1:]:
        ODE.integrate(kk, step=0)
        if ODE.successful():
            psi = ODE.y / dznrm2(ODE.y)
        else:
            err_msg = 'ZVODE exited with status: %s' % ODE.get_return_code()
            raise Exception(err_msg)

        # Do any snapshots here

        # set channel and frame change indexing arrays
    
    
    # Do final measurement at end
    qubits = exp['acquire'][0][1]
    memory_slots = exp['acquire'][0][2]
    probs = occ_probabilities(qubits, psi, global_data['measurement_ops'])
    rand_vals = rng.rand(memory_slots.shape[0]*shots)
    write_shots_memory(memory, memory_slots, probs, rand_vals)
    int_mem = memory.dot(np.power(2.0,
                         np.arange(memory.shape[1]-1,-1,-1))).astype(int)
    if global_data['memory']:
        hex_mem = [hex(val) for val in int_mem]
        return hex_mem
    # Get hex counts dict
    unique = np.unique(int_mem, return_counts = True)
    hex_dict = {}
    for kk in range(unique[0].shape[0]):
        key = hex(unique[0][kk])
        hex_dict[key] = unique[1][kk]
    return hex_dict