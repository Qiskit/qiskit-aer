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
# pylint: disable=unused-variable, no-name-in-module, protected-access,
# pylint: disable=invalid-name, import-error, exec-used

"""Module for unitary pulse evolution.
"""

import logging
import numpy as np
from scipy.integrate import ode
from scipy.linalg.blas import get_blas_funcs
from ..cy.measure import occ_probabilities, write_shots_memory

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)


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

        # Do any snapshots here

        # set channel and frame change indexing arrays

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
