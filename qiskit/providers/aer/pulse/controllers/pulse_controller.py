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
# pylint: disable=invalid-name, no-name-in-module, import-error

"""
Entry/exit point for pulse simulation specified through PulseSimulator backend
"""

from warnings import warn
import numpy as np
from ..system_models.string_model_parser.string_model_parser import NoiseParser
from ..qutip_extra_lite import qobj_generators as qobj_gen
from .digest_pulse_qobj import digest_pulse_qobj
from ..qutip_extra_lite.qobj import Qobj
from .pulse_sim_options import PulseSimOptions
from .unitary_controller import run_unitary_experiments
from .mc_controller import run_monte_carlo_experiments
from .pulse_utils import get_ode_rhs_functor


def pulse_controller(qobj):
    """ Interprets PulseQobj input, runs simulations, and returns results

    Parameters:
        qobj (PulseQobj): pulse qobj containing a list of pulse schedules

    Returns:
        list: simulation results

    Raises:
        ValueError: if input is of incorrect format
        Exception: for invalid ODE options
    """
    pulse_sim_desc = PulseSimDescription()
    pulse_de_model = PulseInternalDEModel()

    config = qobj.config

    # ###############################
    # ### Extract model parameters
    # ###############################

    system_model = config.system_model

    # Get qubit list and number
    qubit_list = system_model.subsystem_list
    if qubit_list is None:
        raise ValueError('Model must have a qubit list to simulate.')
    n_qubits = len(qubit_list)

    # get Hamiltonian
    if system_model.hamiltonian is None:
        raise ValueError('Model must have a Hamiltonian to simulate.')
    ham_model = system_model.hamiltonian

    # Extract DE model information
    pulse_de_model.system = ham_model._system
    pulse_de_model.variables = ham_model._variables
    pulse_de_model.channels = ham_model._channels
    pulse_de_model.h_diag = ham_model._h_diag
    pulse_de_model.evals = ham_model._evals
    pulse_de_model.estates = ham_model._estates
    dim_qub = ham_model._subsystem_dims
    dim_osc = {}
    # convert estates into a Qutip qobj
    estates = [qobj_gen.state(state) for state in ham_model._estates.T[:]]

    # initial state set here
    if hasattr(config, 'initial_state'):
        pulse_sim_desc.initial_state = Qobj(config.initial_state)
    else:
        pulse_sim_desc.initial_state = estates[0]

    # Get dt
    if system_model.dt is None:
        raise ValueError('System model must have a dt value to simulate.')

    pulse_de_model.dt = system_model.dt

    # Parse noise
    noise_model = getattr(config, 'noise_model', None)

    # post warnings for unsupported features
    _unsupported_warnings(noise_model)

    if noise_model:
        noise = NoiseParser(noise_dict=noise_model, dim_osc=dim_osc, dim_qub=dim_qub)
        noise.parse()

        pulse_de_model.noise = noise.compiled
        if any(pulse_de_model.noise):
            pulse_sim_desc.can_sample = False

    # ###############################
    # ### Parse qobj_config settings
    # ###############################
    digested_qobj = digest_pulse_qobj(qobj,
                                      pulse_de_model.channels,
                                      system_model.dt,
                                      qubit_list)

    # extract simulation-description level qobj content
    pulse_sim_desc.shots = digested_qobj.shots
    pulse_sim_desc.meas_level = digested_qobj.meas_level
    pulse_sim_desc.meas_return = digested_qobj.meas_return
    pulse_sim_desc.memory_slots = digested_qobj.memory_slots
    pulse_sim_desc.memory = digested_qobj.memory

    # extract model-relevant information
    pulse_de_model.n_registers = digested_qobj.n_registers
    pulse_de_model.pulse_array = digested_qobj.pulse_array
    pulse_de_model.pulse_indices = digested_qobj.pulse_indices
    pulse_de_model.pulse_to_int = digested_qobj.pulse_to_int

    pulse_sim_desc.experiments = digested_qobj.experiments

    # Handle qubit_lo_freq
    qubit_lo_freq = digested_qobj.qubit_lo_freq

    # if it wasn't specified in the PulseQobj, draw from system_model
    if qubit_lo_freq is None:
        default_freq = getattr(config, 'qubit_freq_est', [np.inf])
        if default_freq != [np.inf]:
            qubit_lo_freq = default_freq

    # if still None, or is the placeholder value draw from the Hamiltonian
    if qubit_lo_freq is None:
        qubit_lo_freq = system_model.hamiltonian.get_qubit_lo_from_drift()
        warn('Warning: qubit_lo_freq was not specified in PulseQobj and there is no default, '
             'so it is beign automatically determined from the drift Hamiltonian.')

    pulse_de_model.freqs = system_model.calculate_channel_frequencies(qubit_lo_freq=qubit_lo_freq)

    # ###############################
    # ### Parse backend_options
    # # solver-specific information should be extracted in the solver
    # ###############################

    pulse_sim_desc.seed = int(config.seed) if hasattr(config, 'seed') else None
    pulse_sim_desc.q_level_meas = int(getattr(config, 'q_level_meas', 1))

    # solver options
    allowed_solver_options = ['atol', 'rtol', 'nsteps', 'max_step',
                              'num_cpus', 'norm_tol', 'norm_steps',
                              'method']
    solver_options = getattr(config, 'solver_options', {})
    for key in solver_options:
        if key not in allowed_solver_options:
            raise Exception('Invalid solver_option: {}'.format(key))
    solver_options = PulseSimOptions(**solver_options)

    # Set the ODE solver max step to be the half the
    # width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in pulse_de_model.pulse_to_int.items():
        if key != 'pv':
            stop = pulse_de_model.pulse_indices[val + 1]
            start = pulse_de_model.pulse_indices[val]
            min_width = min(min_width, stop - start)
    solver_options.de_options.max_step = min_width / 2 * pulse_de_model.dt

    # ########################################
    # Determination of measurement operators.
    # ########################################
    pulse_sim_desc.measurement_ops = [None] * n_qubits

    for exp in pulse_sim_desc.experiments:

        # Add in measurement operators
        # Not sure if this will work for multiple measurements
        # Note: the extraction of multiple measurements works, but the simulation routines
        # themselves implicitly assume there is only one measurement at the end
        if any(exp['acquire']):
            for acq in exp['acquire']:
                for jj in acq[1]:
                    if jj > qubit_list[-1]:
                        continue
                    if not pulse_sim_desc.measurement_ops[qubit_list.index(jj)]:
                        q_level_meas = pulse_sim_desc.q_level_meas
                        pulse_sim_desc.measurement_ops[qubit_list.index(jj)] = \
                            qobj_gen.qubit_occ_oper_dressed(jj,
                                                            estates,
                                                            h_osc=dim_osc,
                                                            h_qub=dim_qub,
                                                            level=q_level_meas
                                                            )

        if not exp['can_sample']:
            pulse_sim_desc.can_sample = False

    run_experiments = (run_unitary_experiments if pulse_sim_desc.can_sample
                       else run_monte_carlo_experiments)
    exp_results, exp_times = run_experiments(pulse_sim_desc, pulse_de_model, solver_options)

    output = {
        'results': format_exp_results(exp_results, exp_times, pulse_sim_desc),
        'success': True,
        'qobj_id': qobj.qobj_id
    }
    return output


def format_exp_results(exp_results, exp_times, pulse_sim_desc):
    """ format simulation results

    Parameters:
        exp_results (list): simulation results
        exp_times (list): simulation times
        pulse_sim_desc (PulseSimDescription): object containing all simulation information

    Returns:
        list: formatted simulation results
    """

    # format the data into the proper output
    all_results = []
    for idx_exp, exp in enumerate(pulse_sim_desc.experiments):

        m_lev = pulse_sim_desc.meas_level
        m_ret = pulse_sim_desc.meas_return

        # populate the results dictionary
        results = {'seed_simulator': exp['seed'],
                   'shots': pulse_sim_desc.shots,
                   'status': 'DONE',
                   'success': True,
                   'time_taken': exp_times[idx_exp],
                   'header': exp['header'],
                   'meas_level': m_lev,
                   'meas_return': m_ret,
                   'data': {}}

        if pulse_sim_desc.can_sample:
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
            if pulse_sim_desc.memory:
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


class PulseInternalDEModel:
    """Container of information required for de RHS construction
    """

    def __init__(self):
        # The system Hamiltonian in numerical format
        self.system = None
        # The noise (if any) in numerical format
        self.noise = None
        # System variables
        self.variables = None
        # Channels in the Hamiltonian string
        # these tell the order in which the channels
        # are evaluated in the RHS solver.
        self.channels = None
        # Array containing all pulse samples
        self.pulse_array = None
        # Array of indices indicating where a pulse starts in the self.pulse_array
        self.pulse_indices = None
        # A dict that translates pulse names to integers for use in self.pulse_indices
        self.pulse_to_int = None
        # dt for pulse schedules
        self.dt = None
        # holds frequencies for the channels
        self.freqs = {}
        # diagonal elements of the hamiltonian
        self.h_diag = None
        # eigenvalues of the time-independent hamiltonian
        self.evals = None
        # eigenstates of the time-independent hamiltonian
        self.estates = None

        self.n_registers = None

        # attributes used in RHS function
        self.vars = None
        self.vars_names = None
        self.num_h_terms = None
        self.c_num = None
        self.c_ops_data = None
        self.c_ops_ind = None
        self.c_ops_ptr = None
        self.n_ops_data = None
        self.n_ops_ind = None
        self.n_ops_ptr = None
        self.h_diag_elems = None

        self.h_ops_data = None
        self.h_ops_ind = None
        self.h_ops_ptr = None

        self._rhs_dict = None

    def _config_internal_data(self):
        """Preps internal data into format required by RHS function.
        """

        self.vars = list(self.variables.values())
        # Need this info for evaluating the hamiltonian vars in the c++ solver
        self.vars_names = list(self.variables.keys())

        num_h_terms = len(self.system)
        H = [hpart[0] for hpart in self.system]
        self.num_h_terms = num_h_terms

        # take care of collapse operators, if any
        self.c_num = 0
        if self.noise:
            self.c_num = len(self.noise)
            self.num_h_terms += 1

        self.c_ops_data = []
        self.c_ops_ind = []
        self.c_ops_ptr = []
        self.n_ops_data = []
        self.n_ops_ind = []
        self.n_ops_ptr = []

        self.h_diag_elems = self.h_diag

        # if there are any collapse operators
        H_noise = 0
        for kk in range(self.c_num):
            c_op = self.noise[kk]
            n_op = c_op.dag() * c_op
            # collapse ops
            self.c_ops_data.append(c_op.data.data)
            self.c_ops_ind.append(c_op.data.indices)
            self.c_ops_ptr.append(c_op.data.indptr)
            # norm ops
            self.n_ops_data.append(n_op.data.data)
            self.n_ops_ind.append(n_op.data.indices)
            self.n_ops_ptr.append(n_op.data.indptr)
            # Norm ops added to time-independent part of
            # Hamiltonian to decrease norm
            H_noise -= 0.5j * n_op

        if H_noise:
            H = H + [H_noise]

        # construct data sets
        self.h_ops_data = [-1.0j * hpart.data.data for hpart in H]
        self.h_ops_ind = [hpart.data.indices for hpart in H]
        self.h_ops_ptr = [hpart.data.indptr for hpart in H]

        self._rhs_dict = {'freqs': list(self.freqs.values()),
                          'pulse_array': self.pulse_array,
                          'pulse_indices': self.pulse_indices,
                          'vars': self.vars,
                          'vars_names': self.vars_names,
                          'num_h_terms': self.num_h_terms,
                          'h_ops_data': self.h_ops_data,
                          'h_ops_ind': self.h_ops_ind,
                          'h_ops_ptr': self.h_ops_ptr,
                          'h_diag_elems': self.h_diag_elems}

    def init_rhs(self, exp):
        """Set up and return rhs function corresponding to this model for a given
        experiment exp
        """

        # if _rhs_dict has not been set up, config the internal data
        if self._rhs_dict is None:
            self._config_internal_data()

        channels = dict(self.channels)

        # Init register
        register = np.ones(self.n_registers, dtype=np.uint8)

        ode_rhs_obj = get_ode_rhs_functor(self._rhs_dict, exp, self.system, channels, register)

        def rhs(t, y):
            return ode_rhs_obj(t, y)

        return rhs


class PulseSimDescription:
    """ Object for holding any/all information required for simulation.
    Needs to be refactored into different pieces.
    """
    def __init__(self):
        self.initial_state = None
        # Channels in the Hamiltonian string
        # these tell the order in which the channels
        # are evaluated in the RHS solver.
        self.experiments = []
        # Can experiments be simulated once then sampled
        self.can_sample = True

        self.shots = None
        self.meas_level = None
        self.meas_return = None
        self.memory_slots = None
        self.memory = None

        self.seed = None
        self.q_level_meas = None
        self.measurement_ops = None
