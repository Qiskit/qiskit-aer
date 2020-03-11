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
# pylint: disable=invalid-name

"""Data configuration module"""

import numpy as np
pi = np.pi


def op_data_config(op_system):
    """ Preps the data for the opsolver.

    Everything is stored in the passed op_system.

    Args:
        op_system (OPSystem): An openpulse system.
    """

    num_h_terms = len(op_system.system)
    H = [hpart[0] for hpart in op_system.system]
    op_system.global_data['num_h_terms'] = num_h_terms

    # take care of collapse operators, if any
    op_system.global_data['c_num'] = 0
    if op_system.noise:
        op_system.global_data['c_num'] = len(op_system.noise)
        op_system.global_data['num_h_terms'] += 1

    op_system.global_data['c_ops_data'] = []
    op_system.global_data['c_ops_ind'] = []
    op_system.global_data['c_ops_ptr'] = []
    op_system.global_data['n_ops_data'] = []
    op_system.global_data['n_ops_ind'] = []
    op_system.global_data['n_ops_ptr'] = []

    op_system.global_data['h_diag_elems'] = op_system.h_diag

    # if there are any collapse operators
    H_noise = 0
    for kk in range(op_system.global_data['c_num']):
        c_op = op_system.noise[kk]
        n_op = c_op.dag() * c_op
        # collapse ops
        op_system.global_data['c_ops_data'].append(c_op.data.data)
        op_system.global_data['c_ops_ind'].append(c_op.data.indices)
        op_system.global_data['c_ops_ptr'].append(c_op.data.indptr)
        # norm ops
        op_system.global_data['n_ops_data'].append(n_op.data.data)
        op_system.global_data['n_ops_ind'].append(n_op.data.indices)
        op_system.global_data['n_ops_ptr'].append(n_op.data.indptr)
        # Norm ops added to time-independent part of
        # Hamiltonian to decrease norm
        H_noise -= 0.5j * n_op

    if H_noise:
        H = H + [H_noise]

    # construct data sets
    op_system.global_data['h_ops_data'] = [-1.0j * hpart.data.data
                                           for hpart in H]
    op_system.global_data['h_ops_ind'] = [hpart.data.indices for hpart in H]
    op_system.global_data['h_ops_ptr'] = [hpart.data.indptr for hpart in H]

    # setup ode args string
    ode_var_str = ""

    # diagonal elements
    ode_var_str += "global_data['h_diag_elems'], "

    # Hamiltonian data
    for kk in range(op_system.global_data['num_h_terms']):
        h_str = "global_data['h_ops_data'][%s], " % kk
        h_str += "global_data['h_ops_ind'][%s], " % kk
        h_str += "global_data['h_ops_ptr'][%s], " % kk
        ode_var_str += h_str

    # Add pulse array and pulse indices
    ode_var_str += "global_data['pulse_array'], "
    ode_var_str += "global_data['pulse_indices'], "

    var_list = list(op_system.vars.keys())
    final_var = var_list[-1]

    freq_list = list(op_system.freqs.keys())
    final_freq = freq_list[-1]

    # Now add channel variables
    chan_list = list(op_system.channels.keys())
    final_chan = chan_list[-1]
    for chan in chan_list:
        ode_var_str += "exp['channels']['%s'][0], " % chan
        ode_var_str += "exp['channels']['%s'][1]" % chan
        if chan != final_chan or var_list:
            ode_var_str += ', '

    # now do the variables
    for idx, var in enumerate(var_list):
        ode_var_str += "global_data['vars'][%s]" % idx
        if var != final_var or freq_list:
            ode_var_str += ', '

    # now do the freq
    for idx, freq in enumerate(freq_list):
        ode_var_str += "global_data['freqs'][%s]" % idx
        if freq != final_freq:
            ode_var_str += ', '

    # Add register
    ode_var_str += ", register"
    op_system.global_data['string'] = ode_var_str

    # Convert inital state to flat array in global_data
    op_system.global_data['initial_state'] = \
        op_system.initial_state.full().ravel()
