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
import os
import sys
import math
import numpy as np
from numpy.random import RandomState, randint
from scipy.integrate import ode
import scipy.sparse as sp
from scipy.integrate._ode import zvode
from scipy.linalg.blas import get_blas_funcs
from collections import OrderedDict
import qutip as qt
from qutip.qobj import Qobj
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv, spmv_csr
from qutip.cy.utilities import _cython_build_cleanup
from openpulse.qobj.operators import apply_projector
import qutip.settings
from openpulse.solver.codegen import OPCodegen
from openpulse.solver.rhs_utils import _op_generate_rhs, _op_func_load
from openpulse.solver.data_config import op_data_config
import openpulse.solver.settings as settings
from openpulse.solver.unitary import unitary_evolution
from qiskit.tools.parallel import parallel_map

dznrm2 = get_blas_funcs("znrm2", dtype=np.float64)

#
# Internal, global variables for storing references to dynamically loaded
# cython functions
#
_cy_rhs_func = None

def opsolve(op_system):
    """Opsolver
    """

    if not op_system.initial_state.isket:
        raise Exception("Initial state must be a state vector.")

    # set num_cpus to the value given in qutip.settings if none in Options
    if not op_system.ode_options.num_cpus:
        op_system.ode_options.num_cpus = qutip.settings.num_cpus

     # compile Cython RHS
    _op_generate_rhs(op_system)
    # Load cython function
    _op_func_load(op_system)
    # build Hamiltonian data structures
    op_data_config(op_system)
    # load monte carlo class
    mc = OP_mcwf(op_system)
    # Run the simulation
    out = mc.run()
    # Results are stored in ophandler.result
    return out

# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class OP_mcwf(object):
    """
    Private class for solving Monte Carlo evolution
    """

    def __init__(self, op_system):

        self.op_system = op_system
        # set output variables, even if they are not used to simplify output
        # code.
        self.output = None
        self.collapse_times = None
        self.collapse_operator = None

        # FOR EVOLUTION WITH COLLAPSE OPERATORS
        if not op_system.can_sample:
            # preallocate ntraj arrays for state vectors, collapse times, and
            # which operator
            self.collapse_times = [[] for kk in 
                range(op_system.global_data['shots'])]
            self.collapse_operators = [[] for kk in 
                range(op_system.global_data['shots'])]
        # setup seeds array
        if op_system.global_data['seed']:
            prng = np.random.RandomState(op_system.global_data['seed'])
        else:
            prng = np.random.RandomState(
                np.random.randint(np.iinfo(np.int32).max-1))
        for exp in op_system.experiments:
            exp['seed'] = prng.randint(np.iinfo(np.int32).max-1)

    def run(self):

        # If no collapse terms, and only measurements at end
        # can do a single shot.
        map_kwargs = {'num_processes': self.op_system.ode_options.num_cpus}

        if self.op_system.can_sample:
            results = parallel_map(unitary_evolution,
                                   self.op_system.experiments,
                                   task_args=(self.op_system.global_data,
                                   self.op_system.ode_options
                                   ), **map_kwargs
                                  )
        
        
        _cython_build_cleanup(self.op_system.global_data['rhs_file_name'])
        return results


# Measurement
def _proj_measurement(pid, ophandler, tt, state, memory, register=None):
    """
    Projection measurement of quantum state
    """
    prng = np.random.RandomState(np.random.randint(np.iinfo(np.int32).max-1))
    qubits = []
    results = []

    for key, acq in ophandler._acqs.items():
        if pid < 0:
            if len(acq.m_slot) > 0:
                mem_slot_id = acq.m_slot[-1]
            else:
                continue
            reg_slot_id = None
        else:
            if tt in acq.t1:
                mem_slot_id = acq.m_slot[acq.t1.index(tt)]
                reg_slot_id = acq.r_slot[acq.t1.index(tt)]
            else:
                continue
        oper = ophandler._measure_ops[acq.name]
        p_q = cy_expect_psi_csr(oper.data.data,
                                oper.data.indices,
                                oper.data.indptr,
                                state,
                                1)
        # level2 measurement
        rnd = prng.rand()
        if rnd <= p_q:
            outcome = 1
        else:
            outcome = 0
        memory[mem_slot_id][0] = outcome
        if reg_slot_id is not None and register is not None:
            register[reg_slot_id] = outcome
        qubits.append(key)
        results.append(outcome)

    # projection
    if len(qubits) > 0:
        psi_proj = apply_projector(qubits, results, ophandler.h_qub, ophandler.h_osc, state)
    else:
        psi_proj = state

    return psi_proj


# -----------------------------------------------------------------------------
# single-trajectory for monte carlo
# -----------------------------------------------------------------------------


