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
# pylint: disable=invalid-name

"""
Entry/exit point for pulse simulation specified through PulseSimulator backend
"""

from warnings import warn
import numpy as np
from ..qutip_extra_lite import qobj_generators as qobj_gen
from .digest_pulse_qobj import digest_pulse_qobj
from ..qutip_extra_lite.qobj import Qobj
from .pulse_sim_options import PulseSimOptions
from .unitary_controller import run_unitary_experiments
from .mc_controller import run_monte_carlo_experiments


def pulse_controller(qobj, system_model, backend_options):
    """ Interprets PulseQobj input, runs simulations, and returns results

    Parameters:
        qobj (qobj): pulse qobj containing a list of pulse schedules
        system_model (PulseSystemModel): contains system model information
        backend_options (dict): dict of options, which overrides other parameters

    Returns:
        list: simulation results

    Raises:
        ValueError: if input is of incorrect format
        Exception: for invalid ODE options
    """

    system_model = system_model.copy()

    pulse_sim_desc = PulseSimDescription()

    if backend_options is None:
        backend_options = {}

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

    # Extract DE model information
    dim_qub = ham_model._subsystem_dims
    dim_osc = {}
    # convert estates into a Qutip qobj
    estates = [qobj_gen.state(state) for state in ham_model._estates.T[:]]

    # initial state set here
    if 'initial_state' in backend_options:
        pulse_sim_desc.initial_state = Qobj(backend_options['initial_state'])
    else:
        pulse_sim_desc.initial_state = estates[0]

    # Check dt
    if system_model.dt is None:
        raise ValueError('System model must have a dt value to simulate.')

    # Parse noise
    noise_model = backend_options.get('noise_model', None)

    if noise_model:
        system_model.add_noise_from_dict(noise_model)

    if system_model._noise is not None:
        pulse_sim_desc.can_sample = False

    # ###############################
    # ### Parse qobj_config settings
    # ###############################
    digested_qobj = digest_pulse_qobj(qobj,
                                      ham_model._channels,
                                      system_model.dt,
                                      qubit_list,
                                      backend_options)

    # extract simulation-description level qobj content
    pulse_sim_desc.shots = digested_qobj.shots
    pulse_sim_desc.meas_level = digested_qobj.meas_level
    pulse_sim_desc.meas_return = digested_qobj.meas_return
    pulse_sim_desc.memory_slots = digested_qobj.memory_slots
    pulse_sim_desc.memory = digested_qobj.memory

    # extract model-relevant information
    system_model._n_registers = digested_qobj.n_registers
    system_model._pulse_array = digested_qobj.pulse_array
    system_model._pulse_indices = digested_qobj.pulse_indices
    system_model._pulse_to_int = digested_qobj.pulse_to_int

    pulse_sim_desc.experiments = digested_qobj.experiments

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

    system_model._freqs = system_model.calculate_channel_frequencies(qubit_lo_freq=qubit_lo_freq)

    # ###############################
    # ### Parse backend_options
    # # solver-specific information should be extracted in the solver
    # ###############################
    pulse_sim_desc.seed = int(backend_options['seed']) if 'seed' in backend_options else None
    pulse_sim_desc.q_level_meas = int(backend_options.get('q_level_meas', 1))

    # solver options
    allowed_solver_options = ['atol', 'rtol', 'nsteps', 'max_step',
                              'num_cpus', 'norm_tol', 'norm_steps',
                              'method']
    solver_options = backend_options.get('solver_options', {})
    for key in solver_options:
        if key not in allowed_solver_options:
            raise Exception('Invalid solver_option: {}'.format(key))
    solver_options = PulseSimOptions(**solver_options)

    # Set the ODE solver max step to be the half the
    # width of the smallest pulse
    min_width = np.iinfo(np.int32).max
    for key, val in system_model._pulse_to_int.items():
        if key != 'pv':
            stop = system_model._pulse_indices[val + 1]
            start = system_model._pulse_indices[val]
            min_width = min(min_width, stop - start)
    solver_options.de_options.max_step = min_width / 2 * system_model.dt

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
    exp_results, exp_times = run_experiments(pulse_sim_desc, system_model, solver_options)

    return format_exp_results(exp_results, exp_times, pulse_sim_desc)


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
