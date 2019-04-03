# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
import numpy as np

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

    op_system.global_data['c_ops_data'] = []
    op_system.global_data['c_ops_ind'] = []
    op_system.global_data['c_ops_ptr'] = []
    op_system.global_data['n_ops_data'] = []
    op_system.global_data['n_ops_ind'] = []
    op_system.global_data['n_ops_ind'] = []

    # if there are any collapse operators
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
        op_system.global_data['n_ops_ind'].append(n_op.data.indptr)
        # Norm ops added to time-independent part of 
        # Hamiltonian to decrease norm
        H[0] -= 0.5j * n_op

    # construct data sets
    op_system.global_data['h_ops_data'] = [-1.0j* hpart.data.data for hpart in H]
    op_system.global_data['h_ops_ind'] = [hpart.data.indices for hpart in H]
    op_system.global_data['h_ops_ptr'] = [hpart.data.indptr for hpart in H]

    # setup ode args string
    ode_var_str = ""
    # Hamiltonian data
    for kk in range(num_h_terms):
        h_str = "global_data['h_ops_data'][%s], " % kk
        h_str += "global_data['h_ops_ind'][%s], " % kk
        h_str += "global_data['h_ops_ptr'][%s], " % kk
        ode_var_str += h_str
    
    # Add pulse array and pulse indices
    ode_var_str += "global_data['pulse_array'], "
    ode_var_str += "global_data['pulse_indices'], "

    var_list = list(op_system.vars.keys())
    final_var = var_list[-1]

    # Now add channel variables
    chan_list = list(op_system.channels.keys())
    final_chan = chan_list[-1]
    for chan in chan_list:
        ode_var_str += "exp['channels']['%s'][0], " % chan
        ode_var_str += "exp['channels']['%s'][1]" % chan
        if chan != final_chan or var_list:
            ode_var_str+= ', '
    
    #now do the variables
    for idx, var in enumerate(var_list):
        ode_var_str += "global_data['vars'][%s]" % idx
        if var != final_var:
             ode_var_str+= ', '
    # Add register
    ode_var_str += ", register"
    op_system.global_data['string'] = ode_var_str

    #Convert inital state to flat array in global_data
    op_system.global_data['initial_state'] = \
        op_system.initial_state.full().ravel()