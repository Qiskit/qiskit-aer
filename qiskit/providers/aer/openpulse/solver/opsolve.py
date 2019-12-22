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

"""The main OpenPulse solver routine.
"""

import time
import numpy as np
from scipy.linalg.blas import get_blas_funcs
from qiskit.tools.parallel import parallel_map, CPU_COUNT
from ..qutip_lite.cy.spmatfuncs import cy_expect_psi_csr
from ..qutip_lite.cy.utilities import _cython_build_cleanup
from ..qobj.operators import apply_projector
from .rhs_utils import _op_generate_rhs, _op_func_load
from .data_config import op_data_config
from .unitary import unitary_evolution
from .monte_carlo import monte_carlo


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
    # load monte carlo class
    montecarlo = OP_mcwf(op_system)
    # Run the simulation
    out = montecarlo.run()
    # Results are stored in ophandler.result
    return out


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class OP_mcwf():
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
                np.random.randint(np.iinfo(np.int32).max - 1))
        for exp in op_system.experiments:
            exp['seed'] = prng.randint(np.iinfo(np.int32).max - 1)

    def run(self):
        """Runs the solver.
        """
        map_kwargs = {'num_processes': self.op_system.ode_options.num_cpus}

        # exp_results from the solvers return the values of the measurement
        # operators

        # If no collapse terms, and only measurements at end
        # can do a single shot.

        # exp_results is a list of '0' and '1'
        # where '0' occurs with probability 1-<M>
        # and '1' occurs with probability <M>
        # M is the measurement operator, which is a projector
        # into one of the qubit states (usually |1>)
        if self.op_system.can_sample:
            start = time.time()
            exp_results = parallel_map(unitary_evolution,
                                       self.op_system.experiments,
                                       task_args=(self.op_system,),
                                       **map_kwargs
                                       )
            end = time.time()
            exp_times = (np.ones(len(self.op_system.experiments)) *
                         (end - start) / len(self.op_system.experiments))

        # need to simulate each trajectory, so shots*len(experiments) times
        # Do a for-loop over experiments, and do shots in parallel_map
        else:
            exp_results = []
            exp_times = []
            for exp in self.op_system.experiments:
                start = time.time()
                rng = np.random.RandomState(exp['seed'])
                seeds = rng.randint(np.iinfo(np.int32).max - 1,
                                    size=self.op_system.global_data['shots'])
                exp_res = parallel_map(monte_carlo,
                                       seeds,
                                       task_args=(exp, self.op_system,),
                                       **map_kwargs)

                # exp_results is a list for each shot
                # so transform back to an array of shots
                exp_res2 = []
                for exp_shot in exp_res:
                    exp_res2.append(exp_shot[0].tolist())

                end = time.time()
                exp_times.append(end - start)
                exp_results.append(np.array(exp_res2))

        # format the data into the proper output
        all_results = []
        for idx_exp, exp in enumerate(self.op_system.experiments):

            m_lev = self.op_system.global_data['meas_level']
            m_ret = self.op_system.global_data['meas_return']

            # populate the results dictionary
            results = {'seed_simulator': exp['seed'],
                       'shots': self.op_system.global_data['shots'],
                       'status': 'DONE',
                       'success': True,
                       'time_taken': exp_times[idx_exp],
                       'header': exp['header'],
                       'meas_level': m_lev,
                       'meas_return': m_ret,
                       'data': {}}

            if self.op_system.can_sample:
                memory = exp_results[idx_exp][0]
                results['data']['statevector'] = []
                for coef in exp_results[idx_exp][1]:
                    results['data']['statevector'].append([np.real(coef),
                                                           np.imag(coef)])
                results['header']['ode_t'] = exp_results[idx_exp][2]
            else:
                memory = exp_results[idx_exp]

            # meas_level 2 return the shots
            if m_lev == 2:

                # convert the memory **array** into a n
                # integer
                # e.g. [1,0] -> 2
                int_mem = memory.dot(np.power(2.0,
                                              np.arange(memory.shape[1]))).astype(int)

                # if the memory flag is set return each shot
                if self.op_system.global_data['memory']:
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

        if not self.op_system.use_cpp_ode_func:
            _cython_build_cleanup(self.op_system.global_data['rhs_file_name'])

        return all_results


# Measurement
def _proj_measurement(pid, ophandler, tt, state, memory, register=None):
    """
    Projection measurement of quantum state
    """
    prng = np.random.RandomState(np.random.randint(np.iinfo(np.int32).max - 1))
    qubits = []
    results = []

    for key, acq in ophandler._acqs.items():
        if pid < 0:
            if any(acq.m_slot):
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
    if any(qubits):
        psi_proj = apply_projector(qubits, results, ophandler.h_qub,
                                   ophandler.h_osc, state)
    else:
        psi_proj = state

    return psi_proj
