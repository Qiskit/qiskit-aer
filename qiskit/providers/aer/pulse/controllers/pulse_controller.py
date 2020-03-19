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
# pylint: disable=invalid-name, missing-return-type-doc

"""
Initial attempt to clarify simulation flow.

This should:
    - interpret the qobj (ultimately eliminating full_digest)
        - For now just call digest_pulse_qobj and accept the current output format
        - Eventually have internal descriptions of: signals, "events" (e.g. measurements), etc
    - set up the DE solving portion (either using OPSystem or eliminating OPSystem)
        - this is a mix of pieces from digest and opsolve
    - call the DE solver
        - the DE solver will be pieces of unitary.py, but also the parts of digest that
          make decisions about things having to do with DE solving
    - given the output, perform measurement
        - This is currently also in unitary.py
        - ultimately measurements should probably consist of calling some Operator functionality
    - format output
        - currently in opsolve
"""

from warnings import warn
import numpy as np
import time
from ..system_models.string_model_parser.string_model_parser import NoiseParser
from qiskit.providers.aer.aererror import AerError
from ..qutip_extra_lite import qobj_generators as qobj_gen
from .digest_pulse_qobj import digest_pulse_qobj
from .pulse0_solvers import unitary_evolution, monte_carlo_evolution
from ..de_solvers.pulse_de_options import OPoptions
from .unitary_controller import run_unitary_experiments
from .mc_controller import run_monte_carlo_experiments
from qiskit.tools.parallel import parallel_map, CPU_COUNT

# remaining pulse0 imports
from ..qutip_extra_lite.cy.measure import occ_probabilities, write_shots_memory


def pulse_controller(qobj, system_model, backend_options):
    """Setup and run simulation, then return results
    """

    out = PulseSimDescription()

    if backend_options is None:
        backend_options = {}

    """ Note: the overriding behaviour of backend_options is currently
    broken, needs to be fixed. Maybe write a function called backend_options_override
    that gets called every time a parameter gets set"""
    # override anything in qobj_config that is present in backend_options
    #for key in backend_options.keys():
    #    qobj_config[key] = backend_options[key]

    noise_model = backend_options.get('noise_model', None)

    # post warnings for unsupported features
    _unsupported_warnings(noise_model)

    # ###############################
    # ### Extract model parameters
    # ###############################

    # Get qubit list and number
    qubit_list = system_model.subsystem_list
    if qubit_list is None:
        raise ValueError('Model must have a qubit list to simulate.')
    n_qubits = len(qubit_list)

    # get Hamiltonian
    if system_model.hamiltonian is None:
        raise ValueError('Model must have a Hamiltonian to simulate.')
    ham_model = system_model.hamiltonian

    # For now we dump this into OpSystem, though that should be refactored
    out.system = ham_model._system
    out.vars = ham_model._variables
    out.channels = ham_model._channels
    out.h_diag = ham_model._h_diag
    out.evals = ham_model._evals
    out.estates = ham_model._estates
    dim_qub = ham_model._subsystem_dims
    dim_osc = {}
    # convert estates into a Qutip qobj
    estates = [qobj_gen.state(state) for state in ham_model._estates.T[:]]
    out.initial_state = estates[0]
    out.global_data['vars'] = list(out.vars.values())
    # Need this info for evaluating the hamiltonian vars in the c++ solver
    out.global_data['vars_names'] = list(out.vars.keys())

    # Get dt
    if system_model.dt is None:
        raise ValueError('Qobj must have a dt value to simulate.')
    out.dt = system_model.dt

    # Parse noise
    noise_dict = noise_model or {}
    if noise_dict:
        noise = NoiseParser(noise_dict=noise_dict, dim_osc=dim_osc, dim_qub=dim_qub)
        noise.parse()

        out.noise = noise.compiled
        if any(out.noise):
            out.can_sample = False
    else:
        out.noise = None

    # ###############################
    # ### Parse qobj_config settings
    # ###############################

    digested_qobj = digest_pulse_qobj(qobj, out.channels, out.dt, qubit_list)

    # does this even need to be extracted here, or can the relevant info just be passed to the
    # relevant functions?
    out.global_data['shots'] = digested_qobj.shots
    out.global_data['meas_level'] = digested_qobj.meas_level
    out.global_data['meas_return'] = digested_qobj.meas_return
    out.global_data['memory_slots'] = digested_qobj.memory_slots
    out.global_data['memory'] = digested_qobj.memory
    out.global_data['n_registers'] = digested_qobj.n_registers

    out.global_data['pulse_array'] = digested_qobj.pulse_array
    out.global_data['pulse_indices'] = digested_qobj.pulse_indices
    out.pulse_to_int = digested_qobj.pulse_to_int

    out.experiments = digested_qobj.experiments

    # Handle qubit_lo_freq
    qubit_lo_freq = digested_qobj.qubit_lo_freq

    # if it wasn't specified in the PulseQobj, draw from system_model
    if qubit_lo_freq is None:
        qubit_lo_freq = system_model._qubit_freq_est

    # if still None draw from the Hamiltonian
    if qubit_lo_freq is None:
        qubit_lo_freq = system_model.hamiltonian.get_qubit_lo_from_drift()
        warn('Warning: qubit_lo_freq was not specified in PulseQobj or in PulseSystemModel, ' +
             'so it is beign automatically determined from the drift Hamiltonian.')

    out.freqs = system_model.calculate_channel_frequencies(qubit_lo_freq=qubit_lo_freq)
    out.global_data['freqs'] = list(out.freqs.values())


    # ###############################
    # ### Parse backend_options
    # # solver-specific information should be extracted in the solver
    # ###############################
    if 'seed' in backend_options:
        out.global_data['seed'] = int(backend_options.get('seed'))
    else:
        out.global_data['seed'] = None
    out.global_data['q_level_meas'] = int(backend_options.get('q_level_meas', 1))

    # solver options
    allowed_ode_options = ['atol', 'rtol', 'nsteps', 'max_step',
                           'num_cpus', 'norm_tol', 'norm_steps',
                           'rhs_reuse', 'rhs_filename']
    ode_options = backend_options.get('ode_options', {})
    for key in ode_options:
        if key not in allowed_ode_options:
            raise Exception('Invalid ode_option: {}'.format(key))
    out.ode_options = OPoptions(**ode_options)

    # Set the ODE solver max step to be the half the
    # width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in out.pulse_to_int.items():
        if key != 'pv':
            stop = out.global_data['pulse_indices'][val + 1]
            start = out.global_data['pulse_indices'][val]
            min_width = min(min_width, stop - start)
    out.ode_options.max_step = min_width / 2 * out.dt

    # ########################################
    # Determination of measurement operators.
    # ########################################
    out.global_data['measurement_ops'] = [None] * n_qubits


    for exp in out.experiments:

        # Add in measurement operators
        # Not sure if this will work for multiple measurements
        # Note: the extraction of multiple measurements works, but the simulator itself
        # implicitly assumes there is only one measurement at the end
        if any(exp['acquire']):
            for acq in exp['acquire']:
                for jj in acq[1]:
                    if jj > qubit_list[-1]:
                        continue
                    if not out.global_data['measurement_ops'][jj]:
                        out.global_data['measurement_ops'][jj] = \
                            qobj_gen.qubit_occ_oper_dressed(jj,
                                                            estates,
                                                            h_osc=dim_osc,
                                                            h_qub=dim_qub,
                                                            level=out.global_data['q_level_meas']
                                                            )

        if not exp['can_sample']:
            out.can_sample = False


    op_data_config(out)

    if out.can_sample == True:
        exp_results, exp_times = run_unitary_experiments(out)
    else:
        exp_results, exp_times = run_monte_carlo_experiments(out)

    return format_exp_results(exp_results, exp_times, out)


def op_data_config(op_system):
    """ Preps the data for the opsolver.

    This should eventually be replaced by functions that construct different types of DEs
    in standard formats

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

    # Convert inital state to flat array in global_data
    op_system.global_data['initial_state'] = \
        op_system.initial_state.full().ravel()


def format_exp_results(exp_results, exp_times, op_system):
    """format results
    """

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

        if op_system.can_sample:
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

    return all_results




def _unsupported_warnings(noise_model):
    """ Warns the user about untested/unsupported features.

    Parameters:
        noise_model (dict): backend_options for simulation
    Returns:
    Raises:
        AerError: for unsupported features
    """

    # Warnings that don't stop execution
    warning_str = '{} are an untested feature, and therefore may not behave as expected.'
    if noise_model is not None:
        warn(warning_str.format('Noise models'))


class PulseSimDescription():
    """ Place holder for "simulation description"
    """
    def __init__(self):
        # The system Hamiltonian in numerical format
        self.system = None
        # The noise (if any) in numerical format
        self.noise = None
        # System variables
        self.vars = None
        # The initial state of the system
        self.initial_state = None
        # Channels in the Hamiltonian string
        # these tell the order in which the channels
        # are evaluated in the RHS solver.
        self.channels = None
        # options of the ODE solver
        self.ode_options = None
        # time between pulse sample points.
        self.dt = None
        # Array containing all pulse samples
        self.pulse_array = None
        # Array of indices indicating where a pulse starts in the self.pulse_array
        self.pulse_indices = None
        # A dict that translates pulse names to integers for use in self.pulse_indices
        self.pulse_to_int = None
        # Holds the parsed experiments
        self.experiments = []
        # Can experiments be simulated once then sampled
        self.can_sample = True
        # holds global data
        self.global_data = {}
        # holds frequencies for the channels
        self.freqs = {}
        # diagonal elements of the hamiltonian
        self.h_diag = None
        # eigenvalues of the time-independent hamiltonian
        self.evals = None
        # eigenstates of the time-independent hamiltonian
        self.estates = None
